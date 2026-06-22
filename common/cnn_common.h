// Common data structures and helpers for intermittent CNN inference.
//
// SYSTEM OVERVIEW
// ===============
// This codebase implements neural-network inference on ultra-low-power MCUs
// (MSP430, MSP432) that operate on harvested energy (solar, RF, thermal).
// Because the supply can fail at any moment, the system must be able to
// resume from a partially-completed inference after each power-on without
// re-running from scratch.
//
// KEY CONCEPTS
// ------------
// NVM / VM:  Non-Volatile Memory (FRAM on MSP) vs. Volatile Memory (SRAM).
//   All persistent state lives in NVM; working buffers live in VM.
//   Data must be explicitly copied between them.
//
// Shadow copies / double-buffering:  Every critical NVM structure is kept in
//   two copies. A version byte selects the "newer" copy. Writes always go to
//   the older copy first, then flip the version byte, so a power failure mid-
//   write leaves at least one consistent copy intact.  See double_buffering.h.
//
// Slots:  Fixed-size NVM regions (INTERMEDIATE_VALUES_SIZE each) used to
//   store intermediate activation tensors between layers. A small slot
//   allocator (get_next_slot) assigns slots to layers, reusing them once the
//   consumer layer has finished.
//
// Jobs / footprints (HAWAII):  Each tile computation is a "job". A footprint
//   records how many jobs have completed so that recovery after power failure
//   can skip already-finished work.  See platform.h and double_buffering.h.
//
// Q15 fixed-point:  All activation and weight values use Q15 format
//   (signed 1.15 fixed-point, range [-1, 1)).  A Scale struct carries the
//   per-tensor scale factor so that logical values can be recovered.
//   All DSP operations go through my_dsplib.h wrappers that handle LEA/CMSIS.
//
// LEA (Low Energy Accelerator):  MSP430FR5994's hardware DSP engine.  It
//   requires even-sized operands and 4-byte-aligned addresses in a dedicated
//   LEARAM region (lea_buffer).  On other targets the same wrappers call
//   ARM CMSIS-DSP or plain C.
//
// Tiling:  Convolution and GEMM are broken into tiles that fit in lea_buffer.
//   Input-channel tiling writes partial sums to the TRANSPOSED NWHC slot and
//   a subsequent ConvMerge / GemmMerge stage accumulates them into NHWC order.
//
// Dynamic DNN:  Optional channel-pruning support (DYNAMIC_DNN_APPROACH) can
//   skip input/output channels at runtime based on a pruning mask tensor,
//   reducing computation while preserving correctness up to the pruning policy.

#pragma once

#include <cstddef> /* size_t, see https://stackoverflow.com/a/26413264 */
#include <cstdint>

#include "data.h"
#include "data_structures.h"
#include "layer-defs.h"

/**********************************
 *        Data structures         *
 **********************************/

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wpadded"
#endif

#define MAX_NUM_DIMS 4

// A node in the ONNX compute graph, stored in NVM as part of the model binary.
// `inputs` holds indices into the combined parameter-info table (model weights
// come first, then intermediate tensors numbered from N_INPUT onward).
// `max_output_id` is the last layer index that consumes this node's output,
// used by the slot allocator to know when a slot can be freed.
typedef struct Node {
  char name[NODE_NAME_LEN];
  char output_name[NODE_NAME_LEN];
  uint16_t inputs_len;
  int16_t inputs[NUM_INPUTS];
  uint16_t max_output_id;
  uint16_t op_type;
  uint16_t parameters_by_importance[2];
} Node;

static_assert(sizeof(Node) == NODE_NAME_LEN * 2 + 10 + NUM_INPUTS * 2,
              "Unexpected size for Node");

// Per-tensor Q15 scale factor.  The logical floating-point value of a Q15
// integer `q` is:  q * fract * (1 << shift) / 32768.
// The struct is 4 bytes and is packed into ParameterInfo with no padding.
struct Scale {
  int16_t fract;
  uint8_t shift;
  uint8_t dummy;

  bool operator>(const Scale& other) const;
  Scale operator*(const Scale& other) const;
  Scale operator/(const Scale& other) const;
  bool operator!=(const Scale& other) const;
  float toFloat() const;
};

const Scale SCALE_ONE = {
    /* .fract = */ 16384,
    /* .shift = */ 1,
    /* .dummy = */ 0,
};

// Describes a tensor — either a static model parameter (weights/biases) or an
// intermediate activation produced by a layer.
//
// `slot` identifies where the tensor data live in NVM:
//   SLOT_TEST_SET    — input sample data (read-only)
//   SLOT_PARAMETERS  — static model weights (read-only, in NVM)
//   [0, NUM_SLOTS)   — an intermediate-values slot (writable, in NVM)
//
// `params_offset` is the byte offset within that slot's NVM region.
// `dims` holds the tensor shape in NCHW order; unused trailing dims are 0.
// `param_flags` carries bit-flags such as TRANSPOSED (data laid out in NWHC)
//   and INTEGER (values are raw integers, not Q15).
// `parameter_info_idx` indexes the combined parameter-info table (indices
//   < N_INPUT are model inputs/weights; >= N_INPUT are intermediate tensors).
//   It MUST remain the last field because my_memcpy in handle_node explicitly
//   copies sizeof(ParameterInfo) - sizeof(uint16_t) to avoid clobbering it.
typedef struct ParameterInfo {
  uint32_t params_offset;
  uint32_t params_len; /* in bytes */
  uint8_t slot;
  uint8_t param_flags;
  // uint8_t is not enough. For example, fully connected layer in MNIST has dims
  // 256x1
  uint16_t dims[MAX_NUM_DIMS];
  Scale scale;
  uint16_t parameter_info_idx;  // must be the last member of this struct
} ParameterInfo;

static_assert(sizeof(ParameterInfo) == 24, "Unexpected size for ParameterInfo");

// Tracks which layer currently "owns" an intermediate-values slot.
// `user` is the layer_idx of the owning layer, or -1 if the slot is free.
typedef struct SlotInfo {
  int16_t user;
} SlotInfo;

// Power-safe model state stored in NVM via shadow copies (see double_buffering.h).
// `running`    — non-zero while inference is in progress.
// `run_counter`— incremented each time a full inference completes.
// `layer_idx`  — index of the next layer to execute (used for recovery).
// `version`    — shadow-copy version byte; MUST be the last field so that
//               commit_versioned_data can write everything except the version
//               in one NVM write, then atomically flip the version.
typedef struct Model {
  uint16_t running;
  uint16_t run_counter;
  uint16_t layer_idx;
  SlotInfo slots_info[NUM_SLOTS];
  uint8_t first_run_done;
  uint8_t version;  // must be the last field in this struct
} Model;

static_assert(sizeof(Model) == 8 + NUM_SLOTS * 2, "Unexpected size for Model");

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

/**********************************
 *          Global data           *
 **********************************/
extern InferenceResults inference_results_vm;

/**********************************
 *         The entry point        *
 **********************************/
uint8_t run_cnn_tests(uint16_t n_samples);

/**********************************
 *          Miscellaneous         *
 **********************************/

/* MSP430 SDK already defines MIN, which means minutes */
#define MIN_VAL(x, y) ((x) < (y) ? (x) : (y))
#define MAX_VAL(x, y) ((x) > (y) ? (x) : (y))
// XXX: MSP432 driverlib requires DMA transfer size to be <= 1024. However,
// transfer size < 1024 may be broken as well - copying 1024 items works,
// copying 512 items works, copy a small number of items (e.g., 6, 10, ...)
// works, and copying 626 items (in ConvMerge of conv2 in MNIST) DOES NOT
// WORK (!?).
#define LIMIT_DMA_SIZE(x) MIN_VAL(512, (x))

/**********************************
 * Helpers for the model & nodes  *
 **********************************/
const uint8_t* get_param_base_pointer(const ParameterInfo* param,
                                      uint32_t* limit_p);
int16_t get_q15_param(Model* model, const ParameterInfo* param,
                      uint32_t offset_in_word);
void put_q15_param(ParameterInfo* param, uint32_t offset_in_word, int16_t val,
                   bool is_linear);
int64_t get_int64_param(const ParameterInfo* param, size_t i);
const ParameterInfo* get_parameter_info(uint16_t i);
const Node* get_node(size_t i);
const Node* get_node(const ParameterInfo* param);
SlotInfo* get_slot_info(Model* model, uint8_t i);
void my_memcpy_from_param(Model* model, void* dest, const ParameterInfo* param,
                          uint32_t offset_in_word, size_t n);

/**********************************
 *       Operation handlers       *
 **********************************/
typedef void (*handler)(Model* model, const ParameterInfo* input[],
                        ParameterInfo* output, const Node* node,
                        CurNodeFlags* node_flags,
                        const NodeFlags* orig_node_flags);
typedef void (*allocator)(Model* model, const ParameterInfo* input[],
                          ParameterInfo* output, const Node* node,
                          CurNodeFlags* node_flags,
                          const NodeFlags* orig_node_flags);
// below are defined in ops.c
extern const handler handlers[];
extern const allocator allocators[];
