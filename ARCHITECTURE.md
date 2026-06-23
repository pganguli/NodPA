# NodPA — MSP430 Intermittent Inference Architecture

A linear, step-by-step trace of everything the board does from first power-on to producing an inference result, with source file references for every step.

---

## 0. System Overview

NodPA runs deep neural network inference on an MSP430FR5994 (or MSP432P401R) that is powered by harvested energy. Because the supply voltage can collapse at any moment, the system must survive partial inference across arbitrary power cycles without re-running from scratch. Three interlocking mechanisms make this possible:

1. **Double-buffered NVM writes** — every critical structure has two copies in external FRAM; the "newer" copy is chosen by a single version byte so a power failure during a write always leaves one consistent copy intact.
2. **HAWAII footprints** — per-layer progress counters written to NVM after every `BATCH_SIZE` output elements, enabling fine-grained restart within a layer.
3. **Slot-based intermediate storage** — a small pool of fixed-size FRAM regions ("slots") holds intermediate activation tensors between layers; the slot allocator survives power cycles.

---

## 1. Hardware Memory Map

| Region | Location | Size |
|---|---|---|
| SRAM (RAM) | On-chip | 8 KB |
| Internal FRAM | On-chip | 128 KB (two banks, 48 KB + 80 KB) |
| External FRAM | SPI (UCA3, P6.0–6.3) | 512 KB – 2 MB depending on chip |
| LEA RAM | On-chip, within SRAM | ~3784 bytes (1892 × Q15) |

The `.text` and `.const` sections (code and model weight tables baked in as C arrays) live in internal FRAM. All runtime NVM I/O goes to the external FRAM chip via SPI.

---

## 2. ONNX Model → C Arrays (`dnn-models/transform.py`)

This offline Python step must run before building the firmware. It produces `build/data.h` and `build/data.cpp`, which are compiled into the binary.

### 2.1 Input

- A trained ONNX model file (`dnn-models/`, configured via `dnn-models/configs.py`).
- A test dataset (CIFAR-10, HAR, KWS) loaded by `dnn-models/datasets.py`.

### 2.2 Graph Transformation (`dnn-models/transform.py`)

1. **Squeeze/Reshape fusion** — nodes whose sole input is a constant initializer are folded into their initializer's dimension list and removed from the graph.
2. **Gemm transpose** — weight matrices with `transB=1` are transposed offline so the firmware never needs to do it at runtime; the `TRANSPOSED` flag is set in `ParameterInfo.param_flags`.
3. **Multi-stage node insertion** (`dnn-models/utils.py: add_multi_stage_nodes`) — each `Conv` is split into a `Conv` (stage 1, produces NWHC-order partial sums) and a `ConvMerge` (stage 2, merges and re-orders to NHWC). Similarly `Gemm`/`MatMul` each get a `GemmStage2`/`MatMulStage2` merge node.
4. **Topological sort** — nodes are re-sorted so all consumers come after producers.
5. **Dynamic channel-pruning mask injection** (`dnn-models/model_utils.py: apply_dynamic_channel_pruning`) — when `DYNAMIC_DNN_APPROACH != 0`, a pruning-mask tensor and a threshold are inserted as extra inputs to the affected `Conv` nodes.

### 2.3 Scale Quantization (`dnn-models/onnx_utils.py: compute_parameter_scales`)

All floating-point weights are scaled to Q15 (signed 1.15 fixed-point, range `[-1, 1)`). For each parameter tensor:

```
q15_value = round(float_value / param_scale * 2^15)
```

`param_scale` comes from per-output annotation (`Q15_SCALE_TENSOR`) or a global config default. The scale is stored as a `Scale{fract, shift}` struct: `logical_value = q15 * fract * 2^shift / 32768`.

### 2.4 Tiling Parameter Calculation

- **Conv** (`dnn-models/layer_utils.py: determine_conv_tile_c`): the script computes `input_tile_c` and `output_tile_c` — the number of input/output channels that fit simultaneously in `lea_buffer` along with the im2col input window and the interleaved filter buffer.
- **Gemm** (`dnn-models/layer_utils.py: determine_gemm_tile_sizes`): computes `tile_a_rows`, `tile_channel`, `tile_b_cols` using the available LEA buffer space.
- These values are written into the `NodeFlags` struct for each node.

### 2.5 NVM Layout Calculation (`dnn-models/transform.py: nvm_layout`)

The script calculates how much external FRAM is consumed by fixed structures and sets `INTERMEDIATE_VALUES_SIZE` — the size of each activation slot — to evenly divide the remaining space among `num_slots` slots (aligned to 16 bytes).

Layout (growing up from address 0):
```
[0,    16)                         reserved / SPI test area
[16,   256)                        LAST_REPORTED_LAYER_IDX (unused on MCU)
[256,  256 + NUM_SLOTS × IVS)      intermediate activation slots
[...,  ... + SAMPLES_DATA_LEN)     test-set input samples (Q15)
[...,  ... + PARAMETERS_DATA_LEN)  static model weights (Q15)
```

Layout (growing down from NVM_SIZE):
```
NVM_SIZE - INFERENCE_STATS_DATA_LEN
         - 2 × MODEL_DATA_LEN           (×2 for shadow copies)
         - INTERMEDIATE_PARAMETERS_INFO_DATA_LEN
         - NODE_FLAGS_DATA_LEN          (already ×2 for shadowing)
         - NODES_DATA_LEN
         - FOOTPRINTS_DATA_LEN          (×2 for shadow copies)
         - COUNTERS_DATA_LEN
         - INFERENCE_RESULTS_DATA_LEN
```

The offsets are computed as `#define` constants in `common/platform.h` using the `*_DATA_LEN` values that `transform.py` writes into `build/data.h`.

### 2.6 Output Binary Blobs

`build/data.cpp` defines these `const uint8_t[]` arrays (each paired with a `*_DATA_LEN` constant in `data.h`):

| Variable | Content |
|---|---|
| `parameters_data` | Q15 weights + biases for all layers |
| `samples_data` | Q15-quantized test-set input samples |
| `labels_data` | Integer class labels |
| `nodes_data` | Array of `Node` structs (name, op_type, inputs list, max_output_id) |
| `node_orig_flags_data` | Array of `NodeFlags` (one per node: tile sizes, strides, pads, etc.) |
| `node_flags_data` | Two shadow copies of `NodeFlags` (for mutable debug/demo counter path) |
| `model_parameters_info_data` | Array of `ParameterInfo` for all static inputs/weights |
| `intermediate_parameters_info_data` | Zeroed `ParameterInfo` array for each node's output tensor |
| `model_data` | Initial `Model` struct (running=0, layer_idx=0, all slots=-1) |
| `footprints_data` | Zeroed footprint array (2 copies × MODEL_NODES_LEN) |
| `counters_data` | Zeroed `Counters` struct |

It also emits the `handlers[]` and `allocators[]` dispatch tables and the `INPLACE_UPDATE_OPS_MAP[]` array.

---

## 3. Boot Sequence

### 3.1 `_system_pre_init` (before C runtime init) — `msp430/main.c:141`

Runs before `.data` and `.bss` are initialized by the C runtime. Immediately calls `WDT_A_hold()` to stop the watchdog so the memory-copy initialization sequence cannot time out.

Returns `1` to tell the runtime to proceed with segment initialization.

### 3.2 `main()` — `msp430/main.c:42`

1. Calls `prvSetupHardware()`.
2. Calls `IntermittentCNNTest()` (never returns).

### 3.3 `prvSetupHardware()` — `msp430/main.c:51`

All steps execute in order:

1. **Watchdog stop** — `WDT_A_hold(...)` (second call, defensive).
2. **GPIO all-output-low** — sets every pin on P1–P4 and PJ to output and drives them low. This prevents floating inputs from leaking current into pull-up/pull-down paths, which would distort energy measurements on the harvesting board.
3. **UCA0 UART pin assignment** — P2.0 (TXD) and P2.1 (RXD) assigned to secondary module function. UART is not yet initialized (baud-rate setup happens later in `uartinit()`).
4. **LFXT crystal pins** — PJ.4 and PJ.5 assigned to primary module function (32.768 kHz crystal).
5. **Button interrupts** — P5.5 (S2) and P5.6 (S1) set as pull-up inputs with falling-edge interrupts enabled.
6. **DCO frequency** — `setFrequency(FreqLevel)` where `FreqLevel = 8` (16 MHz). This sets `FRCTL0 = FRCTLPW | NWAITS_1` (one wait state for internal FRAM at 16 MHz) and `CS_setDCOFreq(CS_DCORSEL_1, CS_DCOFSEL_4)`. `setFrequency` is in `tools/dvfs.c:46`.
7. **Clock signals** — ACLK = LFXT ÷ 1, SMCLK = DCO ÷ 1, MCLK = DCO ÷ 1.
8. **Start LFXT** — `CS_turnOnLFXT(CS_LFXT_DRIVE_0)`.
9. **Unlock LPM5** — `PMM_unlockLPM5()` exits the power-on default high-impedance mode so GPIO outputs take effect.

### 3.4 `IntermittentCNNTest()` — `common/plat-mcu.cpp:210`

This is the top-level MCU entry point, called immediately after hardware setup.

1. **GPIO counter/indicator setup** — GPIO_COUNTER_PIN (P8.0 on MSP430), P4.7 and P1.5 set as output-low (indicator GPIO pins used for oscilloscope timing of layer completions). GPIO_RESET_PIN (P5.7) and gpio_flags pins set as pull-up inputs.

2. **LED on** — P1.0 set high (comms LED signals the board is alive).

3. **Startup delay** — `our_delay_cycles(5e-3 * getFrequency(FreqLevel))` — a 5 ms busy-wait to allow the external FRAM power rail to stabilize. `our_delay_cycles` is implemented in `tools/our_misc.s` (TI assembler) or inline GCC for non-CCS builds.

4. **SPI initialization** (`tools/ext_fram/extfram.c:232: initSPI`):
   - Pin select: P6.0/P6.1/P6.2 assigned to UCA3 SPI (SIMO/SOMI/CLK).
   - CS (P6.3) set as output, driven high.
   - UCA3 configured: MSB first, 8-bit master, 3-pin, SMCLK source, baud divider = `FRAM_FREQ_DIVIDER` (set in `extfram.h`).
   - `CMD_WREN` (0x06) sent to enable writes.
   - `CMD_WRSR` (0x01) sent with 0xC0 to clear write-protection bits on the FRAM status register.

5. **SPI self-test** (`tools/ext_fram/extfram.c:544: testSPI`):
   - Reads 16 bytes from address 0.
   - Writes bytes 0x00–0x0F to address 0.
   - Reads them back and verifies.
   - If the verification fails, the code waits briefly in a loop, then forces a watchdog reset (`WDTCTL = 0`) to trigger a clean restart. This handles the case where the FRAM needs more time after power-on.

6. **Load model from NVM** (`common/platform.cpp:234: load_model_from_nvm`):
   - Calls `get_versioned_data<Model>(0)`.
   - Reads the version byte from `MODEL_OFFSET + offsetof(Model, version)` to determine which of the two shadow copies is newer.
   - Reads `sizeof(Model) - 1` bytes from the newer copy into `model_vm` in SRAM.
   - `model_vm` now contains `running`, `run_counter`, `layer_idx`, `slots_info[]`, `first_run_done`, and `version`.

7. **Load counters** (if `ENABLE_COUNTERS`) — reads the `Counters` struct from `COUNTERS_OFFSET` into SRAM.

8. **`need_reset()` check** — reads P5.7 (`GPIO_RESET_PIN`). If LOW (button held), the board is in "stable-power test mode":
   - `uartinit()` is called — configures UCA0 at 9600 baud using the `UartParams[FreqLevel-1]` table in `tools/myuart.c`.
   - `print_all_counters()` — prints any persisted counters from a previous intermittent run.
   - `first_run()` is called (see §4).
   - `notify_model_finished()` — pulses P8.0 high/low.
   - `run_cnn_tests(1)` is called `STABLE_POWER_ITERATIONS` (10) times in a loop.
   - "Done testing run" is printed over UART.
   - `model->run_counter` is zeroed, model committed.
   - Enters `while(1)` halt.

9. **Intermittent loop** (P5.7 HIGH = harvested power):
   ```c
   while (1) {
       run_cnn_tests(1);
   }
   ```
   Each iteration of this loop may span many power cycles.

---

## 4. First-Run Initialization — `common/platform.cpp:255: first_run()`

Called on the very first boot (or after a deliberate erase+reflash cycle). Wipes and repopulates the external FRAM.

1. `disable_counters()` — prevents counter overhead during initialization.
2. `my_erase()` → `eraseFRAM2(0x00)` — sends `CMD_WREN` then `CMD_WRITE` at address 0 via SPI and streams 0x00 across the entire FRAM chip (0x7FFFF bytes for 4 Mb chip, 0xFFFFF for 8 Mb). This takes several seconds at 16 MHz SPI.
3. `copy_data_to_nvm()` — writes three blobs from internal FRAM to external FRAM using `write_to_nvm_segmented`:
   - `samples_data` → `SAMPLES_OFFSET` (test-set input samples)
   - `parameters_data` → `PARAMETERS_OFFSET` (model weights)
   - `node_flags_data` → `NODE_FLAGS_OFFSET` (two shadow copies of `NodeFlags`)
4. `reset_counters(full=true)` — zeroes the `Counters` struct in SRAM and writes it to `COUNTERS_OFFSET`.
5. `intermediate_parameters_info_data` → `INTERMEDIATE_PARAMETERS_INFO_OFFSET` — the zeroed `ParameterInfo` array for each node's output tensor (one per layer, written segmented).
6. Writes `model_data` to both shadow copies: `MODEL_OFFSET` and `MODEL_OFFSET + MODEL_DATA_LEN` — sets `running=0`, `layer_idx=0`, all `slots_info[i].user = -1`.
7. `load_model_from_nvm()` — refreshes `model_vm` from the newly written NVM.
8. Sets `model->first_run_done = 1` and calls `commit_model()`.

**Note:** DNN weights (`parameters_data`), test samples (`samples_data`), and node descriptors (`nodes_data`) are read from internal FRAM (baked in as C arrays by `transform.py`) and copied to external FRAM during `first_run`. After this, all runtime NVM access goes to external FRAM.

---

## 5. Model State Persistence — `common/double_buffering.h`

### 5.1 The `Model` struct — `common/cnn_common.h:151`

```c
typedef struct Model {
    uint16_t running;          // non-zero while inference in progress
    uint16_t run_counter;      // incremented each completed inference
    uint16_t layer_idx;        // next layer to execute (recovery pointer)
    SlotInfo slots_info[NUM_SLOTS]; // slot → owning layer_idx (-1 = free)
    uint8_t first_run_done;
    uint8_t version;           // MUST be last field
} Model;
```

Two copies live at `MODEL_OFFSET` and `MODEL_OFFSET + sizeof(Model)`. The `version` byte in copy 0 names the "newer" copy (0 or 1). `get_versioned_data<Model>` reads the version byte first, then reads `sizeof(Model) - 1` bytes from the named copy.

### 5.2 `commit_model()` — `common/platform.cpp:241`

1. If `model_vm.running == 0` (full inference just completed): prints counters, resets per-inference counters, and calls `notify_model_finished()` which pulses the GPIO counter pin.
2. Calls `commit_versioned_data<Model>(0)`:
   - Determines the "older" copy id = current_newer XOR 1.
   - Writes `sizeof(Model) - 1` bytes to the older copy's NVM address.
   - Writes `older_copy_id` to the version byte in copy 0.
   - Updates the in-memory copy_id cache.

A power failure during step 2a leaves the older copy corrupt but the newer one intact. A failure during step 2b leaves the version byte unchanged, so the next boot still reads the pre-write copy.

---

## 6. Inference Loop — `common/cnn_common.cpp: run_cnn_tests()` and `run_model()`

### 6.1 `run_cnn_tests(n_samples)` — `common/cnn_common.cpp:336`

1. Calls `get_versioned_data<InferenceResults>(0)` — loads the `InferenceResults` struct (sample_idx, correct, total) from NVM.
2. If `sample_idx >= n_samples`, resets `sample_idx = 0` and commits.
3. Loops while `sample_idx < n_samples`:
   - `run_model(&predicted, &output_node)` — runs one inference.
   - Increments `sample_idx`, updates accuracy counters.
   - `commit_versioned_data<InferenceResults>(0)` — saves progress so a power failure between inferences resumes at the correct sample.

### 6.2 `run_model()` — `common/cnn_common.cpp:240`

1. **Fresh start**: if `model->running == 0`:
   - Reset `layer_idx = 0`.
   - Set all `slots_info[i].user = -1`.
   - If HAWAII: zero all footprints for every layer (`reset_hawaii_layer_footprint`).
   - Set `model->running = 1`.
   - `commit_model()`.

2. **Layer loop**:
   ```c
   for (uint16_t node_idx = model->layer_idx; node_idx < MODEL_NODES_LEN; node_idx++) {
       handle_node(model, node_idx);
       model->layer_idx++;
       commit_model();
       notify_layer_finished();
       save_model_output_data();
   }
   ```
   
   `commit_model()` after each layer writes the updated `layer_idx` to FRAM so that a power failure between layers resumes at the correct layer.

   `notify_layer_finished()` → `notify_indicator(0)` → `gpio_pulse(P4.7)` — pulses P4.7 HIGH for 5 ms. The external power-supply or oscilloscope can count these pulses to track inference progress across power cycles.

### 6.3 `handle_node(model, node_idx)` — `common/cnn_common.cpp:151`

1. Reads `cur_node = get_node(node_idx)` — pointer arithmetic into `nodes_data` in internal FRAM.
2. Reads `cur_node_flags = get_node_flags(node_idx)`:
   - Production: returns a pointer into `node_orig_flags_data` (read-only).
   - Debug/demo-counter: reads from the NVM shadow-copy pair, using a canary byte (0x55) to detect if the SRAM cache is valid.
3. Collects `input[j] = get_parameter_info(input_id[j])` for each input:
   - If `input_id < N_INPUT`: pointer into `model_parameters_info_data` (static weights, in internal FRAM).
   - If `input_id >= N_INPUT`: calls `get_intermediate_parameter_info(i)` — reads 24-byte `ParameterInfo` from `INTERMEDIATE_PARAMETERS_INFO_OFFSET + i * 24` in external FRAM, caches in `intermediate_parameters_info_vm[]`.
4. Allocates output `ParameterInfo`:
   - Copies from `input[0]` (minus `parameter_info_idx`).
   - If not an in-place op: calls `get_next_slot(model)` to assign a free slot.
5. Calls `allocators[op_type](...)` — fills in output tensor shape and `params_len`.
6. Calls `handlers[op_type](...)` — performs the computation.
7. `commit_intermediate_parameter_info(output)` — writes the 24-byte `ParameterInfo` to external FRAM.
8. `flush_intermediate_parameter_info()` — zeroes the `parameter_info_idx` fields of the SRAM cache entries, freeing them for the next layer.
9. If this was the last layer: sets `model->running = 0` and increments `model->run_counter`.
10. Resets HAWAII footprint copy_id cache.

---

## 7. Slot Allocator — `common/cnn_common.cpp:105: get_next_slot()`

The allocator scans `model->slots_info[]` round-robin starting from slot 1:

- A slot is free if `slot_user_id < 0`.
- A slot can be reclaimed if the layer that owns it (`slot_user`) has already been executed: `slot_user->max_output_id < model->layer_idx`.
- After a power failure, the previously assigned slot for the current layer is detected (`slot_user_id == model->layer_idx`) and reused without double-allocation.

`Node::max_output_id` (set by `transform.py`) is the highest layer index that reads this node's output. Once `layer_idx` exceeds `max_output_id`, the slot is no longer needed.

---

## 8. Convolution Operator (`common/conv.cpp`)

### 8.1 `alloc_conv()` — `common/conv.cpp:588`

Reads `NodeFlags.conv.*` (tile sizes, strides, pads, group) to determine:
- Output tensor dimensions (OUTPUT_H, OUTPUT_W, OUTPUT_CHANNEL).
- `n_tiles_c = ceil(CHANNEL / input_tile_c)` — number of input-channel tiles.
- `output->params_len = n_tiles_c × OUTPUT_H × OUTPUT_W × OUTPUT_CHANNEL × 2` (larger than the final output to hold all partial-sum tiles).
- Sets `TRANSPOSED` flag in `output->param_flags` — the stage-1 output is stored in **NWHC** order (not NHWC) so that each input-channel tile's output occupies a contiguous slice.

### 8.2 `handle_conv()` — `common/conv.cpp:683`

**Recovery (HAWAII):**
1. `get_versioned_data<Footprint>(layer_idx)` — reads the footprint for this layer from FRAM (2 NVM reads: version byte + footprint body).
2. `unshuffle_footprint_values()` — reassembles `NUM_COMPLETED_JOBS` from interleaved bytes.
3. Converts `NUM_COMPLETED_JOBS` to `first_unfinished_value_offset` using `job_index_to_offset()` (`common/intermittent-cnn.cpp:49`), which maps the job count back to an NWHC output tensor offset, accounting for input-channel tile boundaries.
4. Decodes the offset into `(input_tile_c_index, filter_tile_index, input_w, input_h)` — the exact position where computation must resume.

**Main loop:**
```
for each input_tile_c_offset in [0, CHANNEL) step input_tile_c:
    [optionally skip if pruning mask says so]
    compute cur_input_tile_c = min(input_tile_c, remaining)
    compute filter_offset (im2col row length including bias slot)
    for each filter_tile_index (output channel tile):
        [optionally skip if pruning mask says so]
        for each input_w (stride steps across W):
            for each input_h (tile_h rows at a time via handle_conv_inner_loop):
                call convTask(cur_input_h, ...)
```

### 8.3 `handle_conv_inner_loop()` — `common/conv.cpp:516`

Determines how many input rows (`tile_h`) fit in `lea_buffer` alongside the filter buffer and matrix-multiply output area:

```
inputs_buffer_end = LEA_BUFFER_SIZE - OUTPUT_LEN - pState_len
                    - (max_n_filters + 1) × filter_offset
tile_h = floor(inputs_buffer_end / (group × dest_offset) - 2×field_size)
         aligned to STRIDE_H
```

Fills the input area of `lea_buffer` with zeros, then writes `−0x8000` (`Q15(−1.0)`) at the bias-multiplier position of every `dest_offset`-wide row. This is the bias trick: the last column of each im2col row is forced to `−1.0`.

Calls `load_ifm_tile_row()` to DMA-copy rows of the input feature map from FRAM into `lea_buffer`. Handles the `TRANSPOSED` flag (previous layer's output was in NWHC) correctly.

Iterates `cur_input_h` across the loaded tile, calling `convTask` for each output-h position.

### 8.4 `convTask()` — `common/conv.cpp:117`

1. **Filter loading** (cached per `(filter_idx, input_tile_c_offset)`):
   - Copies `cur_output_tile_c` filter rows from FRAM into `filter_tmp` area of `lea_buffer`.
   - For standard conv (group=1): appends `−bias` at position `filter_offset − 1` of each filter row. The bias value is divided by `conv_input->scale.toFloat()` to be in the same Q15 unit as the input.
   - Interleaves filters: `my_interleave_q15(filter_tmp, channel, n_filters, filter_buffer_addr, filter_offset)` — scatters each filter's values across `n_filters` columns of the interleaved filter matrix. This makes the matrix multiply produce `n_filters` outputs per input row.

2. **Matrix multiply**:
   ```
   input_buffer_addr points into lea_buffer at the current row offset
   matrix_mpy_results = lea_buffer + LEA_BUFFER_SIZE - OUTPUT_LEN - pState_len
   A_rows = 1, A_cols = filter_offset, B_rows = filter_offset, B_cols = n_filters
   my_matrix_mpy_q15(1, filter_offset, filter_offset, n_filters,
                     input_buffer_addr, filter_buffer_addr,
                     matrix_mpy_results, output, cur_output_data_offset,
                     n_filters, pState_len)
   ```
   This internally calls `msp_matrix_mpy_q15` (TI DSPLib LEA) or `arm_mat_mult_fast_q15` (CMSIS).

3. **NVM write-back**: `my_matrix_mpy_q15` calls `my_memcpy_to_param` as a row-callback inside the TI DSPLib function, writing each output row directly to FRAM slot storage at `INTERMEDIATE_VALUES_OFFSET + slot × IVS + cur_output_data_offset × 2`.

4. **Footprint update**: `write_hawaii_layer_footprint(layer_idx, n_filters)` accumulates `n_filters` jobs; when `BATCH_SIZE` are accumulated it writes to FRAM.

5. **Output layout**: `cur_output_data_offset` is computed in NWHC order:
   ```
   input_tile_c_index × OUTPUT_W × OUTPUT_H × OUTPUT_CHANNEL
   + w × OUTPUT_H × OUTPUT_CHANNEL
   + h × OUTPUT_CHANNEL
   + channel_offset_c
   ```

### 8.5 `handle_conv_stage2()` (ConvMerge) — `common/conv.cpp:1099`

Merges the `n_tiles_c` partial-sum slices into the final NHWC-order tensor.

For each output position `(output_h, output_w)` and each channel chunk:
1. Reads the same channel slice from each tile: `n_tiles_c` reads from FRAM.
2. Accumulates with `my_add_q15`.
3. Writes the merged result to the output slot in NHWC order.
4. Updates HAWAII footprint.

ConvMerge also has its own recovery path using `run_recovery()` (the non-HAWAII variant in `common/op_utils.cpp` which scans the output slot's NVM region to find the last written value).

---

## 9. Fully-Connected Operator (`common/fc.cpp`)

### 9.1 `handle_gemm_impl()` — `common/fc.cpp:155`

ONNX `Gemm` computes `Y = A × B + C` (with the bias trick).

**Tiling layout in `lea_buffer`:**
```
[0,                    buffer_a_size)        tile of A matrix rows
[buffer_a_size,        ...-buffer_b_offset)  temp output area
[aligned,              end)                  tile of B matrix
```

**Bias trick**: the last two rows of the B tile (for the first K-tile only) hold `−bias / A_scale`. Each A row has two extra columns forced to `−0x8000` (i.e., `−1.0`). The matrix multiply automatically adds the bias.

**Weight cache**: if the B tile (same `part_offset`, `tile_b_col_offset`, `tile_channel_offset`) was loaded for a previous A-row tile, it is not reloaded from FRAM.

**Recovery** (`gemm_recovery()`): reads the HAWAII footprint (or scans the output slot), decodes `first_unfinished_value_offset` into `(tile_channel_idx, part_idx, tile_a_row_offset, tile_b_col_offset)`, and resumes the nested loop at that position.

**NVM write-back**: same as Conv — `my_matrix_mpy_q15` calls `my_memcpy_to_param` row-by-row to write output directly to FRAM.

### 9.2 `handle_gemm_stage2_impl()` — `common/fc.cpp:391`

Accumulates K-tile partial sums into the final output tensor. Reads `n_tiles` slices of `output_tile_size` elements each, adds them, writes the result to FRAM, and updates the footprint.

---

## 10. LEA (Low Energy Accelerator) Usage (`common/my_dsplib.cpp`)

The LEA is a hardware DSP co-processor on the MSP430FR5994. It requires:

1. **All operands in LEARAM** — `lea_buffer` is placed in `.leaRAM` section (`#pragma DATA_SECTION(".leaRAM")` in `common/op_utils.cpp:47`).
2. **Even-sized operands** — wrappers round down to even blockSize and handle the trailing element with scalar arithmetic.
3. **4-byte aligned start addresses** — verified by `check_buffer_address()`.
4. **Even B_cols for matrix multiply** — `my_matrix_mpy_q15` asserts this; `padding_for_lea()` rounds up where needed.

Key wrappers (all in `common/my_dsplib.cpp`):

| Wrapper | TI DSPLib | ARM CMSIS |
|---|---|---|
| `my_add_q15` | `msp_add_q15` | `arm_add_q15` |
| `my_offset_q15` | `msp_offset_q15` | `arm_offset_q15` |
| `my_scale_q15` | `msp_scale_q15` | `arm_scale_q15` |
| `my_max_q15` | `msp_max_q15` | `arm_max_q15` |
| `my_min_q15` | `msp_min_q15` | `arm_min_q15` |
| `my_matrix_mpy_q15` | `msp_matrix_mpy_q15` | `arm_mat_mult_fast_q15` |
| `my_vector_mult_q15` | `msp_mpy_q15` | `arm_mult_q15` |

`my_interleave_q15` / `my_deinterleave_q15` are scalar loops (not LEA) because the source/destination is typically at an unaligned offset within the filter tensor.

**On MSP430**, `msp_matrix_mpy_q15` (in `TI-DSPLib/source/matrix/msp_matrix_mpy_q15.c`) is the modified TI DSPLib version that accepts a callback `my_memcpy_to_param` for writing each output row directly to FRAM, bypassing a large SRAM destination buffer.

---

## 11. DMA Usage

### 11.1 SRAM-to-SRAM DMA (within `lea_buffer`) — `common/plat-mcu.cpp:73: my_memcpy()`

On MSP430:
- DMA channel 0, triggered by software (`DMA0TSEL__DMAREQ`).
- Word-width (16-bit) transfers, increment both src and dst.
- Transfer size in words = `n >> 1`.
- `DMAREQ` bit set to fire immediately; call returns synchronously (no interrupt).

On MSP432:
- µDMA channel 0, auto mode, 16-bit transfers.
- Software trigger via `MAP_DMA_requestSoftwareTransfer(0)`.
- Busy-waits on `MAP_DMA_isChannelEnabled(0)`.

### 11.2 SPI FRAM DMA Read — `tools/ext_fram/extfram.c:284: SPI_READ()`

On MSP430:
- DMA3: SMCLK-triggered (SPITXIFG), reads from RXBUF, writes dummy back to TXBUF (no increment) to keep the SPI clock running.
- DMA4: SPIRXIFG-triggered, reads from RXBUF, writes to `dst` buffer with increment.
- Both channels armed; DMA3 triggered by artificially setting `UCTXIFG`. The SPI engine drives DMA3 which feeds DMA4; the call busy-waits on DMA4 completion (`while(DMA4CTL & DMAEN__ENABLE)`).

### 11.3 SPI FRAM DMA Write — `tools/ext_fram/extfram.c:419: SPI_WRITE2()`

On MSP430:
- `CMD_WREN` then `CMD_WRITE` + 3-byte address sent by polling.
- DMA3: triggered by SPITXIFG (or optionally by Timer A1 for delayed-start writes), reads from `src` incrementally and writes to TXBUF.
- DMA ISR (`DMA_VECTOR`) clears the DMAIE flag when done.
- Caller calls `SPI_WAIT_DMA()` → waits on `DMA3CTL & DMAEN__ENABLE`, then asserts CS high.

**Timer-delayed writes** (`timer_delay > 0`): used when writing HAWAII footprints byte-by-byte. Timer A1 in UP mode provides a brief hold-off between CS assertion and the first data byte so the energy-harvesting capacitor can recover between writes.

---

## 12. NVM Read/Write Path Summary

```
read_from_nvm(vm_buffer, nvm_offset, n)
    → SPI_READ(&addr, vm_buffer, n)         extfram.c: DMA SPI read

write_to_nvm(vm_buffer, nvm_offset, n, timer_delay)
    → check_nvm_write_address(...)
    → SPI_WRITE2(&addr, vm_buffer, n, timer_delay)
    → SPI_WAIT_DMA() (if timer_delay == 0)  extfram.c: DMA SPI write
```

Higher-level paths:
- `my_memcpy_from_parameters(dest, param, offset_in_bytes, n)` → `read_from_nvm(dest, PARAMETERS_OFFSET + param->params_offset + offset_in_bytes, n)`.
- `my_memcpy_from_intermediate_values(dest, param, offset_in_word, n)` → `read_from_nvm(dest, INTERMEDIATE_VALUES_OFFSET + slot × IVS + offset_in_word × 2, n)`.
- `my_memcpy_to_param(param, offset_in_word, src, n, ...)` → `write_to_nvm(src, INTERMEDIATE_VALUES_OFFSET + slot × IVS + offset_in_word × 2, n, ...)`.
- `read_from_samples(dest, offset_in_word, n)` → reads from `SAMPLES_OFFSET + sample_idx × 2 × TOTAL_SAMPLE_SIZE + offset × 2`.

---

## 13. Double Buffering in Detail (`common/double_buffering.h`)

### 13.1 Protocol

Every type `T` that needs power-safe persistence uses two NVM copies:

```
Copy 0 lives at: nvm_addr<T>(0, data_idx)
Copy 1 lives at: nvm_addr<T>(1, data_idx)
```

The `version` byte at `nvm_addr<T>(0, data_idx) + offsetof(T, version)` in copy 0 contains the index of the **newer** copy.

**Reading** (`get_versioned_data<T>`):
1. Read version byte from copy 0.
2. Read `sizeof(T) - 1` bytes from `nvm_addr<T>(newer, data_idx)` into SRAM.

**Writing** (`commit_versioned_data<T>`):
1. `older_copy_id = newer ^ 1`.
2. Write `sizeof(T) - 1` bytes to `nvm_addr<T>(older, data_idx)`.
3. Write `older_copy_id` to the version byte.

Power-failure safety: if power fails during step 2, the older copy is corrupt but the newer is intact (version byte still points to it). If power fails during step 3, the new data is safe but the version byte still points to the old copy — the next boot reads the old copy again (at most one redundant re-computation).

### 13.2 Types Using Double Buffering

| Type | Two copies at | Cache |
|---|---|---|
| `Model` | `MODEL_OFFSET` + `MODEL_OFFSET + MODEL_DATA_LEN` | `model_vm` (SRAM) |
| `InferenceResults` | `INFERENCE_RESULTS_OFFSET` × 2 | `inference_results_vm` |
| `Footprint` / `_ExtendedFootprint` | `FOOTPRINTS_OFFSET` + both copies interleaved | `footprints_vm[layer_idx]` |
| `NodeFlags` (debug only) | `NODE_FLAGS_OFFSET` × 2 | `node_flags_vm[node_idx]` |

### 13.3 Footprint Copy-ID Cache

To avoid re-reading the version byte on every footprint write (which would be one NVM read per `BATCH_SIZE` jobs), `footprint_copy_id` in SRAM caches the current newer copy id for the active layer. It is invalidated (set to 0) at the end of each layer by `reset_footprint_copy_id_cache()`.

---

## 14. HAWAII Footprints — Fine-Grained Recovery

### 14.1 Basic Footprint (`_Footprint`)

```c
struct _Footprint {
    uint32_t value;   // NUM_COMPLETED_JOBS (in BATCH_SIZE units)
    uint8_t version;  // shadow-copy discriminator
    uint8_t dummy;
};
```

Total 6 bytes. Two copies for each of `MODEL_NODES_LEN` layers.

### 14.2 Extended Footprint (`_ExtendedFootprint`) — MULTIPLE_INDICATORS variant

```c
struct _ExtendedFootprint {
    uint8_t values[12];  // three uint32_t, interleaved byte-by-byte
    uint8_t version;
    uint8_t dummy;
};
```

The three logical fields are `NUM_COMPLETED_JOBS`, `COMPUTATION_UNIT_INDEX`, and `NUM_SKIPPED_JOBS`. They are stored interleaved so that updating the LSB of any one field requires only a 1-byte NVM write:

```
values[0]  = NUM_COMPLETED_JOBS & 0xff
values[3]  = (NUM_COMPLETED_JOBS >> 8) & 0xff
values[6]  = (NUM_COMPLETED_JOBS >> 16) & 0xff
values[9]  = (NUM_COMPLETED_JOBS >> 24) & 0xff
values[1]  = COMPUTATION_UNIT_INDEX & 0xff   etc.
values[2]  = NUM_SKIPPED_JOBS & 0xff         etc.
```

`unshuffle_footprint_values()` reassembles the three values; `split_footprint_value()` interleaves them back.

### 14.3 Write Optimization

`write_hawaii_layer_footprint()` (`common/platform.cpp:367`):
- Accumulates `n_jobs` into `unshuffled_footprint.values[NUM_COMPLETED_JOBS]`.
- If only the LSB changed (MULTIPLE_INDICATORS variant): writes a single byte to the already-current copy (no version flip required) — only 1 byte of NVM traffic.
- If higher bytes changed: calls `commit_versioned_data<Footprint>` — full shadow-copy write.

The delayed-start timer in `SPI_WRITE2` (timer_delay) is used for these 1-byte footprint writes to give the energy capacitor recovery time between writes.

### 14.4 Footprint Batching — `common/op_utils.cpp:56: hawaii_record_footprints()`

```c
static uint32_t non_recorded_jobs = 0;
void hawaii_record_footprints(Model* model, uint32_t vector_len) {
    non_recorded_jobs += vector_len;
    write_hawaii_layer_footprint(model->layer_idx,
                                 non_recorded_jobs / BATCH_SIZE * BATCH_SIZE);
    non_recorded_jobs %= BATCH_SIZE;
}
```

Only whole multiples of `BATCH_SIZE` are committed. The remainder carries to the next call. This means at most one extra BATCH_SIZE of work is re-executed after a power failure.

---

## 15. Recovery After Power Loss

On the next boot, the sequence is identical to §3, but at step 6 `load_model_from_nvm()` reads whatever state was committed before the failure:

- If `model->running == 0`: the previous inference was complete; begin a new one.
- If `model->running == 1`: the previous inference was interrupted. `run_model()` enters the layer loop at `model->layer_idx` (the first unfinished layer).
- Within each layer's handler, the recovery logic reads the HAWAII footprint for that layer (`get_versioned_data<Footprint>(layer_idx)`) and resumes computation from `first_unfinished_job_idx`.
- `job_index_to_offset()` converts the job count to a tensor offset so the loop state variables (`input_tile_c_index`, `filter_tile_index`, `input_w`, `input_h`) can be re-derived from scratch without storing them in NVM.
- The intermediate activation slots remain intact across the power failure (they are in external FRAM) — already-computed layers' outputs do not need to be recomputed.

---

## 16. Dynamic Channel Pruning (Optional)

When `DYNAMIC_DNN_APPROACH != 0`, each affected Conv layer receives an extra input `conv_channel_pruning_mask` (a Q15 tensor produced by a preceding layer).

At runtime in `handle_conv()`:
- **Input-channel pruning** (`PRUNING_INPUT_CHANNELS`): before entering each `input_tile_c` iteration, reads `channel_mask = pruning_mask[input_tile_c_offset]`. If `|channel_mask| < pruning_threshold`, the entire input-channel tile is skipped. The HAWAII footprint is updated to account for the skipped jobs (`write_hawaii_layer_two_footprints`), so that recovery after a power failure during a pruned run does not re-execute the skipped tiles.
- **Output-channel pruning** (`PRUNING_OUTPUT_CHANNELS`): for each filter tile, reads two mask values. If both are below threshold, the entire filter tile is skipped.

The pruning mask is produced at runtime by a specialized `ConvChannelGating` node earlier in the graph.

---

## 17. UART Debug Output

UART is initialized by `uartinit()` in `tools/myuart.c:275`:
- On MSP430: configures `EUSCI_A0` using `UartParams[FreqLevel - 1]` for 9600 baud at the current DCO frequency, enables RX interrupt.
- Pin: P2.0 = UCA0TXD (set as peripheral output in `prvSetupHardware`).

Debug output routing (controlled by `MY_DEBUG` level from `common/config.h`):
- `MY_DEBUG_NO_ASSERT` (0): `my_printf` is a no-op.
- `MY_DEBUG_NORMAL` (1): `my_printf` maps to `print2uart_new()` which uses `vsnprintf` into a 64-byte buffer then byte-by-byte UART TX.
- `MY_DEBUG_LAYERS` (2): also dumps per-layer tensor values.
- `MY_DEBUG_VERBOSE` (3): dumps per-element matrix values.

Intermittent-mode UART: by default UART is not initialized in the intermittent loop (to save startup energy). It can be enabled at compile time with `ENABLE_DEMO_COUNTERS`. Layer-completion notifications are printed as `"I0\n"` or `".\n"` for model completion.

---

## 18. DVFS (Dynamic Voltage/Frequency Scaling)

`tools/dvfs.c` provides `setFrequency(level)` and `getFrequency(level)`.

The system runs at a fixed `FreqLevel = 8` (16 MHz) on MSP430. DVFS can be changed by calling `setFrequency()` at runtime, but this is not done automatically in the current codebase — all inferences run at 16 MHz.

For 16 MHz (level 8): `FRCTL0 = FRCTLPW | NWAITS_1` must be set before changing the DCO, as internal FRAM requires one wait state above 8 MHz.

---

## 19. Performance Counters (`common/counters.h`, `common/counters.cpp`)

The `Counters` struct lives in SRAM (`counters_vm`) and is conditionally persisted to FRAM at `COUNTERS_OFFSET`. It tracks:

- **CPU cycles**: `progress_seeking`, `memory_layout`, `data_loading`, `embedded_values` (profiled via `msp_benchmarkStart/Stop` on MSP430, DWT on MSP432).
- **DMA**: `dma_invocations`, `dma_bytes`, `dma_vm_to_vm`.
- **NVM traffic** (bytes): `nvm_read_job_outputs`, `nvm_read_parameters`, `nvm_read_shadow_data`, `nvm_read_model`, `nvm_write_shadow_data`, `nvm_write_model`, `nvm_write_linear_jobs`, `nvm_write_non_linear_jobs`, `nvm_write_footprints`.
- **HAWAII**: `progress_preservation_bytes`, `re_execution_macs`.
- **Dynamic DNN**: `num_processed_units`, `num_processed_jobs`, `num_skipped_units`, `num_skipped_jobs`.
- `total_jobs` — persistent across power cycles (the only field written to FRAM on every model commit).

`print_all_counters()` formats and prints all counters over UART. `reset_counters(full)` zeroes all counters; `full=false` preserves `total_jobs`.

---

## 20. Source File Quick Reference

| File | Responsibility |
|---|---|
| `msp430/main.c` | Entry point, GPIO/clock/UART setup, button ISR |
| `common/plat-mcu.cpp` | `IntermittentCNNTest`, DMA memcpy, FRAM read/write, GPIO indicators, DVFS usage |
| `common/plat-mcu.h` | CPU cycle counter macros (benchmark timer / DWT) |
| `common/platform.cpp` | NVM layout, `Model` persistence, `first_run`, HAWAII footprints, `copy_data_to_nvm` |
| `common/platform.h` | NVM offset constants, all read/write/commit declarations |
| `common/double_buffering.h` | `get_versioned_data` / `commit_versioned_data` templates |
| `common/cnn_common.cpp` | `run_cnn_tests`, `run_model`, `handle_node`, slot allocator, `Scale` arithmetic |
| `common/cnn_common.h` | `Node`, `ParameterInfo`, `Model`, `SlotInfo`, `Scale` struct definitions |
| `common/intermittent-cnn.cpp` | `job_index_to_offset`, `batch_start`, `run_recovery` (HAWAII) |
| `common/layers.cpp` | `get_node`, `get_node_flags`, `get_node_orig_flags` |
| `common/conv.cpp` | `alloc_conv`, `handle_conv`, `convTask`, `handle_conv_stage2` |
| `common/fc.cpp` | `alloc_gemm_impl`, `handle_gemm_impl`, `handle_gemm_stage2_impl` (and MatMul wrappers) |
| `common/op_utils.cpp` | `lea_buffer`, `op_buffer`, `hawaii_record_footprints`, `float_to_scale_params`, `make_buffer_aligned` |
| `common/my_dsplib.cpp` | LEA/CMSIS wrapper implementations |
| `common/my_dsplib.h` | LEA/CMSIS wrapper declarations and constraints |
| `common/data_structures.h` | `NodeFlags`, `ConvNodeFlags`, `GemmNodeFlags`, `Footprint`, `Counters`, `InferenceResults` |
| `common/config.h` | `MY_DEBUG` level constants |
| `tools/dvfs.c` | `setFrequency`, `getFrequency`, `FreqLevel` |
| `tools/myuart.c` | `uartinit`, `print2uart_new`, `uart_putc`, baud-rate table |
| `tools/ext_fram/extfram.c` | `initSPI`, `testSPI`, `SPI_READ`, `SPI_WRITE2`, `SPI_WAIT_DMA`, `eraseFRAM2` |
| `tools/our_misc.s` | `our_delay_cycles_internal` (TI assembler busy-wait) |
| `dnn-models/transform.py` | Offline ONNX → C arrays pipeline |
| `dnn-models/layer_utils.py` | Conv/Gemm tile-size calculation |
| `dnn-models/onnx_utils.py` | Scale computation, tensor annotation lookup |
| `dnn-models/model_utils.py` | Dynamic channel-pruning mask injection |
| `build/data.h` / `build/data.cpp` | Generated: all constants, binary blobs, dispatch tables |
| `msp430/lnk_msp430fr5994.cmd` | Linker script: FRAM1/FRAM2 placement for .text/.const |
