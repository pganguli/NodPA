// Power-failure recovery helpers: job-index ↔ NVM-offset translation.
//
// CONCEPTS
// --------
// job_index_to_offset
//   Translates a job index (the number of completed BATCH_SIZE-element groups)
//   into the NVM offset of the LAST element of the corresponding batch.
//   For convolution the mapping is non-trivial because the output is stored
//   in NWHC (transposed) order: each input-channel tile occupies a full
//   OUTPUT_H × OUTPUT_W × OUTPUT_CHANNEL slice, and within that slice the
//   output-channel tile is the innermost dimension.
//   For all other ops the layout is linear: offset = job_index * BATCH_SIZE.
//
// batch_start
//   Given the NVM offset of the LAST element in a batch (the value stored as
//   the footprint sentinel), returns the offset of the FIRST element.
//   Inverse of the "+= BATCH_SIZE - 1" sentinel encoding.
//
// run_recovery (HAWAII variant)
//   Reads the HAWAII footprint for the current layer from NVM and divides by
//   BATCH_SIZE to convert the completed-element count into a job index.
//   The non-HAWAII variant (in op_utils.cpp) scans the output slot directly.

#include "intermittent-cnn.h"

#include <cinttypes>  // for PRId32
#include <cstdint>
#include <cstring>

#include "cnn_common.h"
#include "counters.h"
#include "data.h"
#include "data_structures.h"
#include "double_buffering.h"
#include "layers.h"
#include "my_debug.h"
#include "op_utils.h"
#include "platform.h"

#if HAWAII
uint32_t run_recovery(Model* model, ParameterInfo*) {
  const Footprint* footprint = get_versioned_data<Footprint>(model->layer_idx);
  unshuffle_footprint_values(footprint);
  return unshuffled_footprint.values[FootprintOffset::NUM_COMPLETED_JOBS] /
         BATCH_SIZE;
}
#endif

uint32_t job_index_to_offset(const ParameterInfo* output, uint32_t job_index) {
  const Node* node = get_node(output);
  const NodeFlags* node_flags =
      get_node_flags(output->parameter_info_idx - N_INPUT);
#ifdef OpConv
  uint8_t is_conv = (node->op_type == OpConv);
#else
  uint8_t is_conv = 0;
#endif

  if (!is_conv) {
    return (job_index + 1) * BATCH_SIZE - 1;
  }

  /* BEGIN constants */
  uint16_t input_tile_len, input_tile_jobs, jobs_in_a_filter_tile,
      jobs_in_an_op, output_tile_c, OUTPUT_CHANNEL;
  output_tile_c = node_flags->conv.output_tile_c;
  OUTPUT_CHANNEL = output->dims[1];

  // not taking this shortcut for approaches that use indirect recovery as
  // output padding is used in those approaches
  if (output_tile_c == OUTPUT_CHANNEL) {
    return job_index * BATCH_SIZE + BATCH_SIZE - 1;
  }

  uint16_t OUTPUT_H = output->dims[2], OUTPUT_W = output->dims[3];
  input_tile_len = OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W;
  input_tile_jobs = input_tile_len / BATCH_SIZE;
  output_tile_c = upper_gauss(output_tile_c, BATCH_SIZE) * BATCH_SIZE;
  jobs_in_a_filter_tile = OUTPUT_H * OUTPUT_W * output_tile_c / BATCH_SIZE;
  // TODO: handle cases where the following condition is not met
  MY_ASSERT(output_tile_c % BATCH_SIZE == 0);
  /* END constants */

  uint8_t input_tile_c_index = job_index / input_tile_jobs;
  job_index = job_index % input_tile_jobs;
  uint16_t channel_offset = job_index / jobs_in_a_filter_tile * output_tile_c;
  job_index %= jobs_in_a_filter_tile;
  uint32_t offset = input_tile_c_index * input_tile_len + channel_offset;
  uint16_t cur_output_tile_c =
      MIN_VAL(OUTPUT_CHANNEL - channel_offset, output_tile_c);
  jobs_in_an_op = cur_output_tile_c / BATCH_SIZE;

  if (jobs_in_an_op) {
    // an op contains at least a batch
    offset += OUTPUT_CHANNEL * (job_index / jobs_in_an_op);
    offset += (job_index % jobs_in_an_op + 1) * BATCH_SIZE - 1;
  } else {
    // TODO
    ERROR_OCCURRED();
  }
  return offset;
}

uint32_t batch_start(uint32_t batch_end_offset) {
  return batch_end_offset - (BATCH_SIZE - 1);
}
