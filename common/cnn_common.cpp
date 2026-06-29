// Core inference engine: model graph execution and helper utilities.
//
// This file owns:
//   • handle_node()   — allocates output tensor, calls alloc+handle for one op.
//   • run_model()     — iterates nodes in topological order, commits state
//   after
//                       each so recovery after a power failure restarts from
//                       the right node.
//   • run_cnn_tests() — outer loop over the test-set samples; tracks accuracy.
//   • Scale operators — fixed-point scale arithmetic (multiply, divide,
//   compare).
//
// SLOT ALLOCATOR (get_next_slot)
// Each intermediate activation requires an NVM slot.  The allocator is a
// simple round-robin that frees a slot as soon as its last consumer layer has
// run (tracked via Node::max_output_id).  Recovery after a power failure may
// find a slot already assigned from a previous power cycle; the loop detects
// this and reuses it instead of allocating a new one.

#include "cnn_common.h"

#include <cinttypes>  // for PRId32
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "config.h"
#include "counters.h"
#include "data.h"
#include "data_structures.h"
#include "double_buffering.h"
#include "intermittent-cnn.h"
#include "layer-defs.h"
#include "layers.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"
#include "platform.h"

InferenceResults inference_results_vm;

template <>
uint32_t nvm_addr<InferenceResults>(uint8_t copy_id, uint16_t) {
  return INFERENCE_RESULTS_OFFSET + copy_id * sizeof(InferenceResults);
}

template <>
InferenceResults* vm_addr<InferenceResults>(uint16_t data_idx) {
  return &inference_results_vm;
}

template <>
const char* datatype_name<InferenceResults>(void) {
  return "sample_idx";
}

const ParameterInfo* get_parameter_info(uint16_t i) {
  if (i < N_INPUT) {
    return reinterpret_cast<const ParameterInfo*>(model_parameters_info_data) +
           i;
  } else {
    return get_intermediate_parameter_info(i - N_INPUT);
  }
}

SlotInfo* get_slot_info(Model* model, uint8_t i) {
  if (i < NUM_SLOTS) {
    return model->slots_info + i;
  } else {
    return nullptr;
  }
}

int16_t get_q15_param(Model* model, const ParameterInfo* param,
                      uint32_t offset_in_word) {
  if (param->slot == SLOT_TEST_SET) {
    int16_t ret;
    read_from_samples(&ret, offset_in_word, sizeof(int16_t));
    return ret;
  } else if (param->slot == SLOT_PARAMETERS) {
    int16_t ret;
    my_memcpy_from_parameters(&ret, param, offset_in_word * sizeof(int16_t),
                              sizeof(int16_t));
    return ret;
  } else {
    int16_t ret;
    my_memcpy_from_param(model, &ret, param, offset_in_word, sizeof(int16_t));
    return ret;
  }
}

void put_q15_param(ParameterInfo* param, uint32_t offset_in_word, int16_t val,
                   bool is_linear) {
  my_memcpy_to_param(param, offset_in_word, &val, sizeof(int16_t), 0,
                     is_linear);
}

int64_t get_int64_param(const ParameterInfo* param, size_t i) {
  MY_ASSERT(param->slot == SLOT_PARAMETERS);
  int64_t ret;
  my_memcpy_from_parameters(&ret, param, i * sizeof(int64_t), sizeof(int64_t));
  return ret;
}

static uint16_t get_next_slot(Model* model) {
  /* pick a unused slot */
  uint16_t next_slot_id = 0;
  uint8_t cycle_count = 0;
  while (1) {
    next_slot_id++;
    // Fail if the loop has run a cycle
    if (next_slot_id >= NUM_SLOTS) {
      next_slot_id = 0;
      cycle_count++;
      MY_ASSERT(cycle_count <= 1);
      // Release builds (!DEBUG) disable MY_ASSERT, so the while(1) is
      // the only failsafe against looping forever when all slots are occupied.
      if (cycle_count > 1) {
        while (1);
      }
    }
    int16_t slot_user_id = get_slot_info(model, next_slot_id)->user;
    if (slot_user_id < 0) {
      break;
    }
    // previously allocated, most likely in a previous power cycle
    if (slot_user_id == model->layer_idx) {
      break;
    }
    const Node* slot_user = get_node(slot_user_id);
    if (slot_user->max_output_id < model->layer_idx) {
      break;
    }
  }
  my_printf_debug("next_slot_id = %d" NEWLINE, next_slot_id);
  get_slot_info(model, next_slot_id)->user = model->layer_idx;
  return next_slot_id;
}

void my_memcpy_from_param(Model* model, void* dest, const ParameterInfo* param,
                          uint32_t offset_in_word, size_t n) {
  if (param->slot == SLOT_TEST_SET) {
    read_from_samples(dest, offset_in_word, n);
  } else if (param->slot == SLOT_PARAMETERS) {
    my_memcpy_from_parameters(dest, param, offset_in_word * sizeof(int16_t), n);
  } else {
    my_memcpy_from_intermediate_values(dest, param, offset_in_word, n);
  }
}

static void handle_node(Model* model, uint16_t node_idx) {
  const Node* cur_node = get_node(node_idx);
  CurNodeFlags* cur_node_flags = get_node_flags(node_idx);
  const NodeFlags* cur_orig_node_flags = get_node_orig_flags(node_idx);
#if VERBOSE
  my_printf("Current node: %d, ", node_idx);
  my_printf("name = %.*s, ", NODE_NAME_LEN, cur_node->name);
  my_printf("output_name = %s, ", cur_node->output_name);
  my_printf("op_type = %d" NEWLINE, cur_node->op_type);
#endif

  int16_t input_id[NUM_INPUTS];
  const ParameterInfo* input[NUM_INPUTS];
  for (uint16_t j = 0; j < cur_node->inputs_len; j++) {
    input_id[j] = cur_node->inputs[j];
    my_printf_debug("input_id[%d] = %d" NEWLINE, j, input_id[j]);
    input[j] = get_parameter_info(input_id[j]);
    // dump_params(input[j]);
  }
  for (uint16_t j = cur_node->inputs_len; j < NUM_INPUTS; j++) {
    input[j] = nullptr;
  }
  my_printf_debug(NEWLINE);

  /* Allocate an ParameterInfo for output. Details are filled by
   * individual operation handlers */
  ParameterInfo* output = get_intermediate_parameter_info(node_idx);
  my_memcpy(output, input[0],
            sizeof(ParameterInfo) -
                sizeof(uint16_t));  // don't overwrite parameter_info_idx
  output->params_offset = 0;
  if (!INPLACE_UPDATE_OPS_MAP[cur_node->op_type]) {
    output->slot = get_next_slot(model);
  }

  allocators[cur_node->op_type](model, input, output, cur_node, cur_node_flags,
                                cur_orig_node_flags);
#if DEBUG
  my_printf_debug("Needed mem = %d, dims=(", output->params_len);
  uint32_t minimum_params_len = sizeof(int16_t);
  for (uint8_t dim_idx = 0; dim_idx < 4; dim_idx++) {
    if (!output->dims[dim_idx]) {
      break;
    }
    my_printf_debug("%d, ", output->dims[dim_idx]);
    minimum_params_len *= output->dims[dim_idx];
  }
  my_printf_debug(")" NEWLINE);
  MY_ASSERT(minimum_params_len <= output->params_len &&
            output->params_len <= INTERMEDIATE_VALUES_SIZE);
#endif

  handlers[cur_node->op_type](model, input, output, cur_node, cur_node_flags,
                              cur_orig_node_flags);

#if ENABLE_COUNTERS
  MY_ASSERT(counters_cleared());
#endif
#ifndef __arm__
  // For some operations (e.g., ConvMerge), scale is determined in the handlers
  my_printf_debug("Output scale = %f" NEWLINE, output->scale.toFloat());
#endif

  commit_intermediate_parameter_info(output);
  flush_intermediate_parameter_info();  // flush intermediate ParameterInfo
                                        // loaded to VM for the next layer

  if (node_idx == MODEL_NODES_LEN - 1) {
    model->running = 0;
    model->run_counter++;
#if ENABLE_COUNTERS
    if (need_reset() && !model->run_counter) {
      uint32_t total_jobs = get_counter(offsetof(Counters, total_jobs));
      my_printf("total_jobs=%d" NEWLINE, total_jobs);
    }
#endif
  }

#if HAWAII
  // Different layers use different footprint copies, so reset the cache at the
  // end of a layer
  reset_footprint_copy_id_cache();
#endif
}

#if DEBUG
const float first_sample_outputs[] = FIRST_SAMPLE_OUTPUTS;
#endif

static void run_model(uint16_t* ansptr, const ParameterInfo** output_node_ptr) {
  my_printf_debug("N_INPUT = %d" NEWLINE, N_INPUT);

  Model* model = get_model();
  if (!model->running) {
    // reset model
    model->layer_idx = 0;
    for (uint8_t idx = 0; idx < NUM_SLOTS; idx++) {
      SlotInfo* cur_slot_info = get_slot_info(model, idx);
      cur_slot_info->user = -1;
    }
#if HAWAII
    for (uint16_t node_idx = 0; node_idx < MODEL_NODES_LEN; node_idx++) {
      reset_hawaii_layer_footprint(node_idx);
    }
#endif
    model->running = 1;
    commit_model();
  }

  for (uint16_t node_idx = model->layer_idx; node_idx < MODEL_NODES_LEN;
       node_idx++) {
    handle_node(model, node_idx);
    model->layer_idx++;

    commit_model();

    notify_layer_finished();

    save_model_output_data();
  }

  // the parameter info for the last node should also be refreshed in release
  // builds (DEBUG == 0); otherwise the model is not correctly re-initialized
  const ParameterInfo* output_node =
      get_parameter_info(MODEL_NODES_LEN + N_INPUT - 1);
  if (output_node_ptr) {
    *output_node_ptr = output_node;
  }
#if DEBUG
  if (inference_results_vm.sample_idx == 0) {
    my_printf("OUT slot=%d poff=%lu plen=%lu dims=%d,%d scale=%f" NEWLINE,
              (int)output_node->slot, (unsigned long)output_node->params_offset,
              (unsigned long)output_node->params_len,
              (int)output_node->dims[0], (int)output_node->dims[1],
              output_node->scale.toFloat());
  }
  int16_t max = INT16_MIN;
  uint16_t u_ans;
  uint16_t ans_len = sizeof(first_sample_outputs) / sizeof(float);
  ans_len = MIN_VAL(output_node->dims[1], ans_len);

  float output_max = 0;
  if (inference_results_vm.sample_idx == 0) {
    for (uint16_t buffer_idx = 0; buffer_idx < ans_len; buffer_idx++) {
      output_max =
          MAX_VAL(std::fabs(first_sample_outputs[buffer_idx]), output_max);
    }
  }

  uint16_t buffer_len = LIMIT_DMA_SIZE(ans_len);

  for (uint16_t buffer_offset = 0; buffer_offset < ans_len;
       buffer_offset += buffer_len) {
    uint16_t cur_buffer_len = MIN_VAL(buffer_len, ans_len - buffer_offset);

    my_memcpy_from_param(model, lea_buffer, output_node, buffer_offset,
                         cur_buffer_len * sizeof(int16_t));

#if DYNAMIC_DNN_APPROACH != DYNAMIC_DNN_FINE_GRAINED
    if (inference_results_vm.sample_idx == 0) {
      bool all_ok = true;
      uint16_t first_bad_idx = 0;
      float first_bad_got = 0, first_bad_expected = 0;
      uint16_t ofm_idx = buffer_offset;
      for (uint16_t buffer_idx = 0; buffer_idx < cur_buffer_len; buffer_idx++) {
        int16_t got_q15 = lea_buffer[buffer_idx];
        {
          float got_real =
              q15_to_float(got_q15, ValueInfo(output_node), nullptr);
          float expected = first_sample_outputs[ofm_idx];
          float error = fabs((got_real - expected) / output_max);
          my_printf("DBG idx=%d q15=%d got=%f exp=%f" NEWLINE,
                    buffer_offset + buffer_idx, (int)got_q15, got_real, expected);
          if (all_ok && error > 0.1) {
            all_ok = false;
            first_bad_idx = buffer_offset + buffer_idx;
            first_bad_got = got_real;
            first_bad_expected = expected;
          }
          ofm_idx++;
        }
      }
      // Errors in CIFAR-10 are quite large...
      MY_ASSERT(
          all_ok,
          "Value error too large at index %d: got=%f, expected=%f" NEWLINE,
          first_bad_idx, first_bad_got, first_bad_expected);
    }
#endif

    int16_t cur_max = INT16_MIN;
    uint16_t cur_ans = 0;
    my_max_q15(lea_buffer, cur_buffer_len, &cur_max, &cur_ans);
    if (cur_max > max) {
      max = cur_max;
      u_ans = buffer_offset + cur_ans;
    }
  }

  *ansptr = u_ans;
#endif
}

uint8_t run_cnn_tests(uint16_t n_samples) {
  // -1 wraps to 0xFFFF, used as an invalid/uninitialized sentinel.
  uint16_t predicted = static_cast<uint16_t>(-1);
  const ParameterInfo* output_node;
#if DEBUG
  uint16_t label = static_cast<uint16_t>(-1);
  if (!n_samples) {
    n_samples = LABELS_DATA_LEN / sizeof(int16_t);
  }
  const uint16_t* labels = reinterpret_cast<const uint16_t*>(labels_data);
#endif

  get_versioned_data<InferenceResults>(0);
  if (inference_results_vm.sample_idx >= n_samples) {
    inference_results_vm.sample_idx = 0;
    commit_versioned_data<InferenceResults>(0);
  }

  for (; inference_results_vm.sample_idx < n_samples;) {
    my_printf_debug("Running sample %d" NEWLINE,
                    inference_results_vm.sample_idx);
    run_model(&predicted, &output_node);
#if DEBUG
    label = labels[inference_results_vm.sample_idx];
    inference_results_vm.total++;
    if (label == predicted) {
      inference_results_vm.correct++;
    }
    if (inference_results_vm.sample_idx % 100 == 99) {
      my_printf("Finished %d/%d inputs" NEWLINE,
                inference_results_vm.sample_idx + 1, n_samples);
      // stdout is not flushed at \n if it is not a terminal
      my_flush();
    }
    my_printf_debug("idx=%d label=%d predicted=%d correct=%d" NEWLINE,
                    inference_results_vm.sample_idx, label, predicted,
                    label == predicted);
#endif
    inference_results_vm.sample_idx++;
    commit_versioned_data<InferenceResults>(0);
  }
#if DEBUG
  if (n_samples == 1) {
    dump_params(get_model(), output_node, "Output", "Output");
  }
  my_printf("correct=%" PRId32 " ", (int32_t)inference_results_vm.correct);
  my_printf("total=%" PRId32 " ", (int32_t)inference_results_vm.total);
#ifndef __arm__
  my_printf("accuracy=%.2f%%" NEWLINE,
            100.0 * inference_results_vm.correct / inference_results_vm.total);
#else
  my_printf(NEWLINE);
#endif

#endif
  return 0;
}

bool Scale::operator>(const Scale& other) const {
  return this->toFloat() > other.toFloat();
}

Scale Scale::operator*(const Scale& other) const {
  Scale newScale(*this);
  int32_t newFract = static_cast<int32_t>(newScale.fract) * other.fract;
  while (newFract < 32768) {
    newFract *= 2;
    newScale.shift--;
  }
  newScale.fract = (newFract / 32768);
  newScale.shift += other.shift;
  return newScale;
}

Scale Scale::operator/(const Scale& other) const {
  Scale newScale(*this);
  int32_t newFract =
      (static_cast<int32_t>(newScale.fract) * 32768 / other.fract);
  while (newFract >= 32768 || newScale.shift < other.shift) {
    newFract /= 2;
    newScale.shift++;
  }
  newScale.fract = newFract;
  newScale.shift -= other.shift;
  return newScale;
}

bool Scale::operator!=(const Scale& other) const {
  // XXX: missing accuracy?
  return this->toFloat() != other.toFloat();
}

float Scale::toFloat() const {
  // use static_cast<int32_t>(1)<<shift instead of 1<<shift, otherwise
  // 1 is interpreted as int16_t on MSP430 and overflow occurs for shift >= 16
  return 1.0f * fract * (static_cast<int32_t>(1) << shift) / 32768;
}
