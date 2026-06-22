// Per-node configuration flags stored in NVM alongside the model binary.
//
// NodeFlags is a tagged union: the active member depends on Node::op_type.
// The union is NODE_FLAGS_SIZE (20) bytes, sized to hold the largest member.
// Sizes are checked with a static_assert; add `dummy` padding to any new
// member struct if needed to keep members consistent and avoid NVM layout
// shifts.
//
// `general_flags`   — bitfield shared across all op types (INPUT_1_SCALE, etc.)
// `cumulative_jobs` — used by the demo-counter / progress-reporting path.
// `canary`          — non-zero (0x55) when the NodeFlags VM copy is valid.
// `version`         — shadow-copy version byte; must be the last field.

#pragma once

#include <cstdint>

struct ConvNodeFlags {
  uint16_t input_tile_c;    // Number of input channels per tile (K-tiling)
  uint16_t output_tile_c;   // Number of output channels per tile
  uint8_t pads[4];          // [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end]
  uint8_t kernel_shape[2];  // [kH, kW]
  uint8_t strides[2];       // [stride_h, stride_w]
  uint8_t group;            // Depthwise group count; 1 for standard conv
  uint8_t pruning_target;   // PRUNING_INPUT_CHANNELS or PRUNING_OUTPUT_CHANNELS
  int16_t pruning_threshold;  // Q15 magnitude below which a channel is skipped
  uint16_t pState_len;        // ARM CMSIS pState scratch size (words)
};

struct MaxPoolFlags {
  uint8_t kernel_shape[2];  // [kH, kW]
  uint8_t strides[2];       // [stride_h, stride_w]
  uint8_t ceil;       // Non-zero → ceil_mode output size (ONNX ceil_mode=1)
  uint8_t nhwc2nchw;  // Non-zero → emit output in NCHW instead of NHWC
};

struct GemmNodeFlags {
  uint16_t tile_a_rows;   // Number of A-matrix rows per tile
  uint16_t tile_channel;  // Number of shared K-dimension elements per tile
  uint16_t tile_b_cols;   // Number of B-matrix columns per tile
  uint16_t pState_len;    // ARM CMSIS pState scratch size (words)
  uint8_t input_dims;     // Number of spatial dimensions in the input tensor
  uint8_t weight_dims;    // Number of dimensions in the weight tensor
};

struct GemmMergeNodeFlags {
  uint16_t tile_length;  // Number of output elements in each GemmMerge tile
  uint8_t
      input_dims;  // Number of spatial dimensions (passed through from Gemm)
};

struct SqueezeNodeFlags {
  // Bitmap of axes to squeeze/unsqueeze (bit i → axis i).
  uint8_t axes;
};

struct ConcatNodeFlags {
  int8_t axis;  // Concatenation axis (0 = batch, 1 = channel in NHWC)
};

struct TransposeNodeFlags {
  int8_t perm[4];  // Forward permutation (ONNX perm attribute)
  int8_t
      inverse_perm[4];  // Precomputed inverse; used in element-by-element copy
};

struct SoftmaxNodeFlags {
  int8_t axis;  // Axis along which softmax is computed (usually last dim)
};

struct ArgMaxNodeFlags {
  uint8_t axis;      // Axis to reduce over
  uint8_t keepdims;  // Non-zero → keep the reduced axis with size 1
};

struct GatherNodeFlags {
  uint8_t axis;  // Data axis along which indices select elements
};

#define NODE_FLAGS_SIZE 20

struct NodeFlags {
  union {
    struct ConvNodeFlags conv;
    struct ConvNodeFlags conv_channel_gating;
    struct MaxPoolFlags max_pool;
    struct GemmNodeFlags gemm;
    struct GemmMergeNodeFlags gemm_stage2;
    struct SqueezeNodeFlags squeeze;
    struct ConcatNodeFlags concat;
    struct TransposeNodeFlags transpose;
    struct SoftmaxNodeFlags softmax;
    struct ArgMaxNodeFlags arg_max;
    struct GatherNodeFlags gather;
    uint8_t as_bytes[NODE_FLAGS_SIZE];
  };
  uint32_t cumulative_jobs;
  uint8_t general_flags;
  uint8_t dummy;
  // `canary` contains some non-zero value for detecting whether data are
  // already in VM or not
  uint8_t canary;
  uint8_t version;
};

static_assert(sizeof(struct NodeFlags) == NODE_FLAGS_SIZE + 8,
              "Unexpected size for NodeFlags");

// Standard footprint: one 32-bit "jobs done" counter per layer.
// Shadow-copied so that a power failure mid-write leaves one consistent copy.
struct _Footprint {
  uint32_t value;   // Number of completed jobs since last reset
  uint8_t version;  // Shadow-copy discriminator (flipped on each write)
  uint8_t dummy;    // Padding
};

// Extended footprint used by HAWAII when BATCH_SIZE == 1.
// Stores three uint32_t values interleaved byte-by-byte across 12 bytes so
// that each field can be updated with a single 1-byte NVM write.  See
// platform.cpp: split_footprint_value() / unshuffle_footprint_values().
struct _ExtendedFootprint {
  uint8_t values[12];  // 3 × uint32_t, interleaved (bytes[0,3,6,9] = value[0])
  uint8_t version;     // Shadow-copy discriminator
  uint8_t dummy;       // Padding
};

struct InferenceStats {
  uint32_t last_progress_indicator;  // Job index at the last reported progress
  uint32_t power_cycle_energy;  // Accumulated energy cost of this inference
  uint8_t dummy[3];
  uint8_t version;  // Shadow-copy discriminator
};

// All timing/energy/memory counters collected during one inference run.
// Members are grouped by subsystem; names are self-documenting.
// The counters struct is zeroed at the start of each inference; only
// total_jobs is "persistent" (survives across power cycles in NVM).
#define N_PERSISTENT_COUNTERS 1
struct Counters {
  uint32_t power_counters;  // CPU cycles spent in power/energy measurement

  uint32_t macs;  // Multiply-accumulate operations executed

  uint32_t progress_seeking;  // CPU cycles spent scanning footprints at startup

  uint32_t memory_layout;  // CPU cycles in slot/offset calculations

  uint32_t data_loading;  // CPU cycles spent reading parameters from NVM

  uint32_t embedded_values;  // CPU cycles accessing inline tensor data

  uint32_t dma_invocations;       // Number of DMA transfers initiated
  uint32_t dma_bytes;             // Total bytes moved by DMA
  uint32_t dma_vm_to_vm;          // DMA transfers entirely within SRAM
  uint32_t nvm_read_job_outputs;  // Bytes read back from NVM job output slots
  uint32_t nvm_read_parameters;   // Bytes read from NVM parameter storage
  uint32_t nvm_read_shadow_data;  // Bytes read from NVM shadow-copy structures
  uint32_t nvm_read_model;        // Bytes read from the NVM model descriptor
  uint32_t
      nvm_write_shadow_data;  // Bytes written to NVM shadow-copy structures
  uint32_t nvm_write_model;   // Bytes written to NVM model descriptor
  uint32_t
      nvm_write_linear_jobs;  // Bytes written for linear (FC/Conv) job outputs
  uint32_t
      nvm_write_non_linear_jobs;  // Bytes written for non-linear job outputs
  uint32_t nvm_write_footprints;  // Bytes written to HAWAII footprint records

  uint32_t num_processed_units;  // Individual output elements computed
  uint32_t num_processed_jobs;   // Jobs (tiles) actually executed
  uint32_t num_skipped_units;    // Output elements skipped (recovered from NVM)
  uint32_t num_skipped_jobs;  // Jobs skipped because footprint showed them done

  uint32_t
      progress_preservation_bytes;  // Bytes written purely for checkpointing
  uint32_t re_execution_macs;       // MACs re-executed due to partial recovery

  // Survives power cycles in NVM; counts cumulative jobs across all cycles.
  uint32_t total_jobs;
};

struct InferenceResults {
  uint16_t sample_idx;  // Index of the most recently evaluated input sample
  uint16_t correct;     // Cumulative correct predictions in current run
  uint16_t total;       // Cumulative samples evaluated in current run
  uint8_t dummy;
  uint8_t version;  // Shadow-copy discriminator
};
