// Platform abstraction: NVM I/O, model persistence, and HAWAII footprints.
//
// This file provides the implementation of all NVM read/write operations and
// the higher-level helpers (get_model, commit_model, first_run, etc.) that
// are shared between the MCU and PC targets.  Low-level I/O (SPI for MCU,
// mmap for PC) is provided by plat-mcu.cpp / plat-pc.cpp respectively.
//
// INTERMEDIATE PARAMETER INFO CACHE
//   A tiny VM-side cache (intermediate_parameters_info_vm) holds the
//   ParameterInfo structs for the current and adjacent layers so that NVM is
//   not re-read on every tensor access.  flush_intermediate_parameter_info()
//   resets the cache at the end of each layer by zeroing the parameter_info_idx
//   fields (idx == 0 is the "free slot" sentinel).
//
// FOOTPRINT ENCODING (HAWAII)
//   The basic _Footprint holds a single uint32_t progress counter.
//   The extended _ExtendedFootprint holds three 32-bit values packed
//   byte-interleaved across 12 bytes so that the HAWAII partial-write protocol
//   can safely update only the least-significant byte without corrupting the
//   others:
//
//     bytes[0]  = value[0] & 0xff       (NUM_COMPLETED_JOBS LSB)
//     bytes[3]  = (value[0] >> 8) & 0xff
//     bytes[6]  = (value[0] >> 16) & 0xff
//     bytes[9]  = (value[0] >> 24) & 0xff
//     bytes[1]  = value[1] & 0xff       (COMPUTATION_UNIT_INDEX LSB)
//     ... etc.
//
//   unshuffle_footprint_values() reassembles the interleaved bytes into the
//   plain uint32_t values in unshuffled_footprint before they are used.
//   split_footprint_value() does the inverse.
//
// HAWAII WRITE OPTIMISATION
//   When only the LSB of a footprint counter changes (most increments by a
//   small BATCH_SIZE stay within a single byte), the MULTIPLE_INDICATORS
//   variant writes a single byte instead of a full shadow-copy commit.  This
//   halves NVM traffic for the common case.

#include "platform.h"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "c_callbacks.h"
#include "cnn_common.h"
#include "config.h"
#include "counters.h"
#include "data.h"
#include "data_structures.h"
#include "double_buffering.h"
#include "intermittent-cnn.h"  // for sample_idx
#include "my_debug.h"

// put offset checks here as extra headers are used
// In the external-FRAM layout the NVM grows from both ends; verify the two
// regions don't overlap.  In the internal-FRAM layout PARAMETERS_OFFSET is a
// virtual alias beyond WRITABLE_NVM_SIZE, so the same check does not apply.
#if !defined(__MSP430FR5962__) || EXT_FRAM
static_assert(COUNTERS_OFFSET >= PARAMETERS_OFFSET + PARAMETERS_DATA_LEN,
              "Incorrect NVM layout");
#endif

static const uint8_t FOOTPRINT_SIZE = 2;

Model model_vm;
const uint8_t NUM_PARAMETER_INFO_SLOTS =
    1 + NUM_INPUTS;  // ParameterInfo for one output and several inputs
static ParameterInfo intermediate_parameters_info_vm[NUM_PARAMETER_INFO_SLOTS];

static uint32_t intermediate_values_offset(uint8_t slot_id) {
  return INTERMEDIATE_VALUES_OFFSET + slot_id * INTERMEDIATE_VALUES_SIZE;
}

static uint32_t intermediate_parameters_info_addr(uint16_t i) {
  return INTERMEDIATE_PARAMETERS_INFO_OFFSET + i * sizeof(ParameterInfo);
}

template <>
uint32_t nvm_addr<Model>(uint8_t i, uint16_t) {
  return MODEL_OFFSET + i * sizeof(Model);
}

template <>
Model* vm_addr<Model>(uint16_t data_idx) {
  return &model_vm;
}

template <>
const char* datatype_name<Model>(void) {
  return "model";
}

static void notify_progress(void) {
#if 0
    // indicate there is some progress in this power cycle
    static bool notified = false;
    if (!notified) {
        notify_indicator(1);
        notified = true;
    }
#endif
}

void my_memcpy_to_param(ParameterInfo* param, uint32_t offset_in_word,
                        const void* src, size_t n, uint16_t timer_delay,
                        bool is_linear) {
  MY_ASSERT(param->slot < NUM_SLOTS);
  uint32_t total_offset =
      param->params_offset + offset_in_word * sizeof(int16_t);

#if DYNAMIC_DNN_APPROACH != DYNAMIC_DNN_FINE_GRAINED
  MY_ASSERT(total_offset + n <= param->params_len);
#else
  if (total_offset + n > param->params_len) {
    return;
  }
#endif

#if ENABLE_COUNTERS && !ENABLE_DEMO_COUNTERS
  uint32_t n_jobs;
  n_jobs = n;
  if (is_linear) {
    add_counter(offsetof(Counters, nvm_write_linear_jobs), n_jobs);
    my_printf_debug("Recorded %u bytes of linear jobs written to NVM" NEWLINE,
                    n_jobs);
  } else {
    add_counter(offsetof(Counters, nvm_write_non_linear_jobs), n_jobs);
    my_printf_debug(
        "Recorded %u bytes of non-linear jobs written to NVM" NEWLINE, n_jobs);
  }
#endif

  write_to_nvm(src, intermediate_values_offset(param->slot) + total_offset, n,
               timer_delay);

  notify_progress();
}

void my_memcpy_from_intermediate_values(void* dest, const ParameterInfo* param,
                                        uint32_t offset_in_word, size_t n) {
#if ENABLE_COUNTERS
  if (counters_enabled) {
    add_counter(offsetof(Counters, nvm_read_job_outputs), n);
    my_printf_debug(
        "Recorded %lu bytes of job outputs fetched from NVM, "
        "accumulated=%" PRIu32 NEWLINE,
        n, get_counter(offsetof(Counters, nvm_read_job_outputs)));
  }
#endif

  read_from_nvm(dest,
                intermediate_values_offset(param->slot) +
                    offset_in_word * sizeof(int16_t),
                n);
}

void read_from_samples(void* dest, uint32_t offset_in_word, size_t n) {
#if ENABLE_COUNTERS
  if (counters_enabled) {
    add_counter(offsetof(Counters, nvm_read_job_outputs), n);
    my_printf_debug(
        "Recorded %lu bytes of samples fetched from NVM, accumulated=%" PRIu32
            NEWLINE,
        n, get_counter(offsetof(Counters, nvm_read_job_outputs)));
  }
#endif

  read_from_nvm(dest,
                SAMPLES_OFFSET +
                    (inference_results_vm.sample_idx % LABELS_DATA_LEN) * 2 *
                        TOTAL_SAMPLE_SIZE +
                    offset_in_word * sizeof(int16_t),
                n);
}

static uint8_t get_available_parameter_info_slot() {
  for (uint8_t parameter_info_slot_idx = 0;
       parameter_info_slot_idx < NUM_PARAMETER_INFO_SLOTS;
       parameter_info_slot_idx++) {
    if (intermediate_parameters_info_vm[parameter_info_slot_idx]
            .parameter_info_idx == 0) {
      return parameter_info_slot_idx;
    }
  }

  MY_ASSERT(false);

  return UINT8_MAX;
}

ParameterInfo* get_intermediate_parameter_info(uint16_t i) {
#if ENABLE_COUNTERS
  if (counters_enabled) {
    add_counter(offsetof(Counters, nvm_read_model), sizeof(ParameterInfo));
    my_printf_debug(
        "Recorded %lu bytes of ParameterInfo fetched from NVM" NEWLINE,
        sizeof(ParameterInfo));
  }
#endif
  ParameterInfo* dst =
      intermediate_parameters_info_vm + get_available_parameter_info_slot();
  read_from_nvm(dst, intermediate_parameters_info_addr(i),
                sizeof(ParameterInfo));
  my_printf_debug("Load intermediate parameter info %d from NVM" NEWLINE, i);
  MY_ASSERT(dst->parameter_info_idx == i + N_INPUT,
            "Expect parameter index %d but got %d" NEWLINE, i + N_INPUT,
            dst->parameter_info_idx);
  return dst;
}

void commit_intermediate_parameter_info(const ParameterInfo* param) {
#if ENABLE_COUNTERS
  if (counters_enabled) {
    add_counter(offsetof(Counters, nvm_write_model), sizeof(ParameterInfo));
    my_printf_debug("Recorded %lu bytes of ParameterInfo written NVM" NEWLINE,
                    sizeof(ParameterInfo));
  }
#endif
  uint16_t node_idx = param->parameter_info_idx - N_INPUT;
  write_to_nvm(param, intermediate_parameters_info_addr(node_idx),
               sizeof(ParameterInfo));
  my_printf_debug("Committing intermediate parameter info %d to NVM" NEWLINE,
                  node_idx);
}

void flush_intermediate_parameter_info() {
  for (uint8_t parameter_info_slot_idx = 0;
       parameter_info_slot_idx < NUM_PARAMETER_INFO_SLOTS;
       parameter_info_slot_idx++) {
    intermediate_parameters_info_vm[parameter_info_slot_idx]
        .parameter_info_idx = 0;
  }
}

Model* load_model_from_nvm(void) {
  Model* ret = get_versioned_data<Model>(0);
  return ret;
}

Model* get_model(void) { return &model_vm; }

void commit_model(void) {
  if (!model_vm.running) {
    print_all_counters();
    reset_counters(/*full=*/false);
  }
  commit_versioned_data<Model>(0);
  // send finish signals only after the whole network has really finished
  add_counter(offsetof(Counters, power_counters), 1);

  if (!model_vm.running) {
    notify_model_finished();
  }
}

void first_run(void) {
  my_printf_debug("First run, resetting everything..." NEWLINE);
  disable_counters();
  my_erase();
  copy_data_to_nvm();
  reset_counters(/*full=*/true);

  write_to_nvm_segmented(
      intermediate_parameters_info_data, intermediate_parameters_info_addr(0),
      INTERMEDIATE_PARAMETERS_INFO_DATA_LEN, sizeof(ParameterInfo));
  write_to_nvm(model_data, nvm_addr<Model>(0, 0), MODEL_DATA_LEN);
  write_to_nvm(model_data, nvm_addr<Model>(1, 0), MODEL_DATA_LEN);

  Model* model = load_model_from_nvm();  // refresh model_vm
  model->first_run_done = 1;
  commit_model();

  my_printf_debug("Init for " CONFIG "/" METHOD " with batch size=%d" NEWLINE,
                  BATCH_SIZE);
  enable_counters();
}

void read_from_nvm_segmented(uint8_t* vm_buffer, uint32_t nvm_offset,
                             uint32_t total_len, uint16_t segment_size) {
  for (uint32_t idx = 0; idx < total_len; idx += segment_size) {
    read_from_nvm(vm_buffer + idx, nvm_offset + idx,
                  MIN_VAL(total_len - idx, segment_size));
  }
}

void write_to_nvm_segmented(const uint8_t* vm_buffer, uint32_t nvm_offset,
                            uint32_t total_len, uint16_t segment_size) {
  for (uint32_t idx = 0; idx < total_len; idx += segment_size) {
    write_to_nvm(vm_buffer + idx, nvm_offset + idx,
                 MIN_VAL(total_len - idx, segment_size));
  }
}

#if HAWAII
NOINIT static Footprint footprints_vm[MODEL_NODES_LEN];
NOINIT UnshuffledFootprint unshuffled_footprint;
#if DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS
NOINIT UnshuffledFootprint unshuffled_footprint_mirror[2];
#endif
static uint8_t footprint_copy_id = 0;

template <>
uint32_t nvm_addr<Footprint>(uint8_t copy_id, uint16_t layer_idx) {
  return FOOTPRINTS_OFFSET +
         (copy_id * MODEL_NODES_LEN + layer_idx) * sizeof(Footprint);
}

template <>
Footprint* vm_addr<Footprint>(uint16_t layer_idx) {
  return &footprints_vm[layer_idx];
}

template <>
const char* datatype_name<Footprint>(void) {
  return "footprint";
}

template <>
uint8_t* copy_id_cache_addr<Footprint>(void) {
  return &footprint_copy_id;
}

#if DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS_BASIC || \
    DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS
static uint32_t combine_footprint_value(const Footprint* footprint,
                                        uint8_t footprint_offset) {
  uint32_t footprint_value = (footprint->values[footprint_offset + 0] << 0) +
                             (footprint->values[footprint_offset + 3] << 8) +
                             (footprint->values[footprint_offset + 6] << 16) +
                             (footprint->values[footprint_offset + 9] << 24);
  my_printf_debug("Combined footprint %d value = %d" NEWLINE, footprint_offset,
                  footprint_value);
  return footprint_value;
}

static inline void split_footprint_value(Footprint* footprint,
                                         const uint32_t* footprint_values,
                                         uint8_t footprint_offset) {
  uint32_t footprint_value = footprint_values[footprint_offset];
  my_printf_debug("Split value %d to footprint %d" NEWLINE, footprint_value,
                  footprint_offset);
  footprint->values[footprint_offset + 0] = (footprint_value >> 0) & 0xff;
  footprint->values[footprint_offset + 3] = (footprint_value >> 8) & 0xff;
  footprint->values[footprint_offset + 6] = (footprint_value >> 16) & 0xff;
  footprint->values[footprint_offset + 9] = (footprint_value >> 24) & 0xff;
}

void unshuffle_footprint_values(const Footprint* footprint) {
  for (uint8_t footprint_offset = 0;
       footprint_offset < FootprintOffset::NUM_FOOTPRINTS; footprint_offset++) {
    unshuffled_footprint.values[footprint_offset] =
        combine_footprint_value(footprint, footprint_offset);
  }
}
#else
static inline void split_footprint_value(Footprint* footprint,
                                         const uint32_t* footprint_values,
                                         uint8_t footprint_offset) {
  MY_ASSERT(footprint_offset == 0);
  footprint->value = footprint_values[0];
}

void unshuffle_footprint_values(const Footprint* footprint) {
  unshuffled_footprint.values[0] = footprint->value;
}
#endif

void write_hawaii_layer_footprint(uint16_t layer_idx, int16_t n_jobs) {
#if DYNAMIC_DNN_APPROACH != DYNAMIC_DNN_COARSE_GRAINED
  uint32_t old_value =
      unshuffled_footprint.values[FootprintOffset::NUM_COMPLETED_JOBS];
  uint32_t new_value = old_value + n_jobs;

  unshuffled_footprint.values[FootprintOffset::NUM_COMPLETED_JOBS] = new_value;

  Footprint* footprint = vm_addr<Footprint>(layer_idx);

#if DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS
  constexpr uint32_t mask = std::numeric_limits<uint32_t>::max() - 0xff;

  if ((old_value & mask) == (new_value & mask)) {
    my_printf_debug("Only LSB is changed, preseving one byte..." NEWLINE);

    uint8_t newer_copy_id = get_newer_copy_id<Footprint>(layer_idx);

    footprint->values[0] = new_value & 0xff;
    write_to_nvm(footprint->values,
                 nvm_addr<Footprint>(newer_copy_id, layer_idx),
                 sizeof(uint8_t));

    add_demo_counter(offsetof(Counters, progress_preservation_bytes),
                     sizeof(uint8_t));

    unshuffled_footprint_mirror[newer_copy_id].values[0] = new_value;
  } else
#endif
  {
    my_printf_debug("More than LSB is changed, preseving all bytes..." NEWLINE);

    split_footprint_value(footprint, unshuffled_footprint.values,
                          FootprintOffset::NUM_COMPLETED_JOBS);
    commit_versioned_data<Footprint>(layer_idx);

    // see commit_versioned_data: sizeof(Footprint) - sizeof(uint8_t) for data +
    // sizeof(uint8_t) for the pointer
    add_demo_counter(offsetof(Counters, progress_preservation_bytes),
                     FOOTPRINT_SIZE);

#if DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS
    my_memcpy(
        &unshuffled_footprint_mirror[get_newer_copy_id<Footprint>(layer_idx)],
        &unshuffled_footprint, sizeof(UnshuffledFootprint));
#endif
  }

  my_printf_debug("After committing one footprint" NEWLINE);
  dump_footprints_debug(layer_idx);
#endif
}

#if DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS_BASIC || \
    DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS
void write_hawaii_layer_two_footprints(uint16_t layer_idx,
                                       FootprintOffset footprint_offset1,
                                       int16_t increment1,
                                       FootprintOffset footprint_offset2,
                                       int16_t increment2) {
  my_printf_debug("Increment footprint at offset %d by %d" NEWLINE,
                  footprint_offset1, increment1);
  my_printf_debug("Increment footprint at offset %d by %d" NEWLINE,
                  footprint_offset2, increment2);

  MY_ASSERT(std::abs(footprint_offset1 - footprint_offset2) == 1);

  my_printf_debug("Before committing two footprints" NEWLINE);
  dump_footprints_debug(layer_idx);

  Footprint* footprint = vm_addr<Footprint>(layer_idx);

  unshuffled_footprint.values[footprint_offset1] += increment1;
  unshuffled_footprint.values[footprint_offset2] += increment2;

#if DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS
  uint8_t newer_copy_id = get_newer_copy_id<Footprint>(layer_idx);
  uint8_t older_copy_id = newer_copy_id ^ 1;

  const UnshuffledFootprint& mirror =
      unshuffled_footprint_mirror[older_copy_id];

  // Comparing old and new values. If only the least significant bytes are
  // changed, update them only. Otherwise, update the whole value.
  constexpr uint32_t mask = std::numeric_limits<uint32_t>::max() - 0xff;

  uint8_t ok_values = 0;
  for (uint8_t footprint_offset = 0;
       footprint_offset < sizeof(UnshuffledFootprint) / sizeof(uint32_t);
       footprint_offset++) {
    // The condition for partial preservation: for updated footprints, only LSBs
    // are changed, and for other footprints, the value remains exactly the same
    if (footprint_offset == footprint_offset1 ||
        footprint_offset == footprint_offset2) {
      if ((mirror.values[footprint_offset] & mask) ==
          (unshuffled_footprint.values[footprint_offset] & mask)) {
        ok_values++;
      }
    } else {
      if (mirror.values[footprint_offset] ==
          unshuffled_footprint.values[footprint_offset]) {
        ok_values++;
      }
    }
  }

  if (ok_values == FootprintOffset::NUM_FOOTPRINTS) {
    my_printf_debug(
        "Only LSB in either footprint is changed, preseving two "
        "bytes..." NEWLINE);

    footprint->values[footprint_offset1] =
        unshuffled_footprint.values[footprint_offset1] & 0xff;
    footprint->values[footprint_offset2] =
        unshuffled_footprint.values[footprint_offset2] & 0xff;

    commit_versioned_data<Footprint>(
        layer_idx,
        /*commit_offset=*/MIN_VAL(footprint_offset1, footprint_offset2),
        /*num_bytes=*/2);

    add_demo_counter(offsetof(Counters, progress_preservation_bytes), 2);
  } else
#endif
  {
    my_printf_debug(
        "More than LSB is changed in some footprint, preseving all "
        "bytes..." NEWLINE);

    split_footprint_value(footprint, unshuffled_footprint.values,
                          footprint_offset1);
    split_footprint_value(footprint, unshuffled_footprint.values,
                          footprint_offset2);

    commit_versioned_data<Footprint>(layer_idx);

    add_demo_counter(offsetof(Counters, progress_preservation_bytes),
                     FOOTPRINT_SIZE);
  }

#if DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS
  // The newer copy id is flipped in commit_versioned_data above
  newer_copy_id ^= 1;
  my_memcpy(&unshuffled_footprint_mirror[newer_copy_id], &unshuffled_footprint,
            sizeof(UnshuffledFootprint));
#endif

  my_printf_debug("After committing two footprints" NEWLINE);
  dump_footprints_debug(layer_idx);
}
#endif

void reset_hawaii_layer_footprint(uint16_t layer_idx) {
  Footprint footprint;
  memset(&footprint, 0, sizeof(Footprint));
  write_to_nvm(&footprint, nvm_addr<Footprint>(0, layer_idx),
               sizeof(Footprint));
  write_to_nvm(&footprint, nvm_addr<Footprint>(1, layer_idx),
               sizeof(Footprint));
  my_printf_debug("Reset HAWAII layer footprint for layer %d" NEWLINE,
                  layer_idx);
}

void reset_footprint_copy_id_cache(void) {
  *copy_id_cache_addr<Footprint>() = 0;
}

#endif
