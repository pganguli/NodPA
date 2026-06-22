// Performance counter storage, update, and reporting.
//
// The Counters struct lives in both SRAM (counters_data_vm[]) for fast
// in-place increment and NVM (at COUNTERS_OFFSET) for power-cycle survival.
// load_counters() reads NVM into SRAM at startup; _add_counter() increments
// in SRAM then writes the single 32-bit word back to NVM immediately.
// counters_enabled is set to 0 during those NVM accesses to prevent the NVM
// write itself from being counted (which would cause infinite recursion).
//
// Per-layer mode (ENABLE_PER_LAYER_COUNTERS): counters_data_vm is indexed by
// model_vm.layer_idx so each layer accumulates independently.  The NVM
// write offset is also layer-indexed via counter_offset().
//
// reset_counters(full=false) zeroes all fields except the last
// N_PERSISTENT_COUNTERS uint32_t (currently total_jobs).  This lets the
// power-cycle count accumulate across many inference runs.
//
// start_cpu_counter / stop_cpu_counter implement a two-level nesting stack
// (current_counter + prev_counter).  Nesting is needed because e.g.
// "progress seeking" is measured around a loop that may also trigger NVM
// reads that are separately timed under "data loading".

#include "counters.h"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "cnn_common.h"
#include "data.h"
#include "data_structures.h"
#include "my_debug.h"
#include "platform.h"

uint8_t counters_enabled = 1;

#if ENABLE_COUNTERS
uint8_t current_counter = INVALID_POINTER;
uint8_t prev_counter = INVALID_POINTER;
uint32_t num_skipped_jobs_since_boot = 0;

Counters counters_data_vm[COUNTERS_LEN];

Counters* counters() {
#if ENABLE_PER_LAYER_COUNTERS
  return counters_data_vm + model_vm.layer_idx;
#else
  return counters_data_vm;
#endif
}

static uint32_t counter_offset(uint8_t counter) {
#if ENABLE_PER_LAYER_COUNTERS
  return COUNTERS_OFFSET + model_vm.layer_idx * sizeof(Counters) + counter;
#else
  return COUNTERS_OFFSET + counter;
#endif
}

template <uint32_t Counters::* MemPtr>
static uint32_t print_counters() {
  uint32_t total = 0;
  for (uint16_t i = 0; i < MODEL_NODES_LEN; i++) {
    total += counters_data_vm[i].*MemPtr;
#if ENABLE_PER_LAYER_COUNTERS
    my_printf("%12" PRIu32, counters_data_vm[i].*MemPtr);
#else
    break;
#endif
  }
  my_printf(" total=%12" PRIu32, total);
  return total;
}

void print_all_counters() {
#if !ENABLE_DEMO_COUNTERS
  my_printf("op types:                ");
#if ENABLE_PER_LAYER_COUNTERS
  for (uint16_t i = 0; i < MODEL_NODES_LEN; i++) {
    my_printf("% 12d", get_node(i)->op_type);
  }
#endif
  uint32_t total_dma_bytes = 0, total_macs = 0, total_overhead = 0;
  uint32_t total_num_skipped_jobs = 0, total_num_processed_jobs = 0,
           total_num_skipped_units = 0, total_num_processed_units = 0;
  my_printf(NEWLINE "Power counters:          ");
  print_counters<&Counters::power_counters>();
  my_printf(NEWLINE "MACs:                    ");
  total_macs = print_counters<&Counters::macs>();
  // recovery overheads
  my_printf(NEWLINE "Progress seeking:        ");
  total_overhead += print_counters<&Counters::progress_seeking>();
  // misc
  my_printf(NEWLINE "Memory layout:           ");
  total_overhead += print_counters<&Counters::memory_layout>();
  my_printf(NEWLINE "DMA invocations:         ");
  print_counters<&Counters::dma_invocations>();
  my_printf(NEWLINE "DMA bytes:               ");
  total_dma_bytes = print_counters<&Counters::dma_bytes>();
  my_printf(NEWLINE "DMA (VM to VM):          ");
  print_counters<&Counters::dma_vm_to_vm>();
  my_printf(NEWLINE "NVM read (job outputs):  ");
  print_counters<&Counters::nvm_read_job_outputs>();
  my_printf(NEWLINE "NVM read (parameters):   ");
  print_counters<&Counters::nvm_read_parameters>();
  my_printf(NEWLINE "NVM read (shadow data):  ");
  print_counters<&Counters::nvm_read_shadow_data>();
  my_printf(NEWLINE "NVM read (model data):   ");
  print_counters<&Counters::nvm_read_model>();
  my_printf(NEWLINE "NVM write (shadow data): ");
  print_counters<&Counters::nvm_write_shadow_data>();
  my_printf(NEWLINE "NVM write (model data):  ");
  print_counters<&Counters::nvm_write_model>();

  my_printf(NEWLINE "NVM write (L jobs):      ");
  total_overhead += print_counters<&Counters::nvm_write_linear_jobs>();
  my_printf(NEWLINE "NVM write (NL jobs):     ");
  total_overhead += print_counters<&Counters::nvm_write_non_linear_jobs>();
  my_printf(NEWLINE "NVM write (footprints):  ");
  total_overhead += print_counters<&Counters::nvm_write_footprints>();

  my_printf(NEWLINE "Processed units:         ");
  total_num_processed_units = print_counters<&Counters::num_processed_units>();
  my_printf(NEWLINE "Skipped units:           ");
  total_num_skipped_units = print_counters<&Counters::num_skipped_units>();
  my_printf(NEWLINE "Processed jobs:          ");
  total_num_processed_jobs = print_counters<&Counters::num_processed_jobs>();
  my_printf(NEWLINE "Skipped jobs:            ");
  total_num_skipped_jobs = print_counters<&Counters::num_skipped_jobs>();

  my_printf(NEWLINE "Ratio of skipped jobs: %f",
            1.0 * total_num_skipped_jobs /
                (total_num_skipped_jobs + total_num_processed_jobs));
  my_printf(NEWLINE "Ratio of skipped units: %f",
            1.0 * total_num_skipped_units /
                (total_num_skipped_units + total_num_processed_units));

  my_printf(NEWLINE "Total DMA bytes: %d", total_dma_bytes);
  my_printf(NEWLINE "Total MACs: %d", total_macs);
  my_printf(NEWLINE "Communication-to-computation ratio: %f",
            1.0f * total_dma_bytes / total_macs);
  my_printf(NEWLINE "Total overhead: %" PRIu32, total_overhead);
  my_printf(NEWLINE "run_counter: %d" NEWLINE, get_model()->run_counter);
#endif
}

static uint32_t* get_counter_ptr(uint8_t counter) {
  return reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(counters()) +
                                     counter);
}

void load_counters(void) {
  // Temporarily disable counters as NVM access may also update counters,
  // resulting in infinite recursion
  counters_enabled = 0;
  read_from_nvm_segmented(reinterpret_cast<uint8_t*>(counters_data_vm),
                          COUNTERS_OFFSET, sizeof(Counters) * COUNTERS_LEN,
                          sizeof(Counters));
  counters_enabled = 1;
}

void _add_counter(uint8_t counter, uint32_t value) {
  // Disable counters for a similar reason
  counters_enabled = 0;
  *get_counter_ptr(counter) += value;
  write_to_nvm(get_counter_ptr(counter), counter_offset(counter),
               sizeof(uint32_t), 0);
  my_printf_debug("Increment counter %d to %d" NEWLINE, counter,
                  *get_counter_ptr(counter));
  counters_enabled = 1;
}

uint32_t get_counter(uint8_t counter) { return *get_counter_ptr(counter); }

void reset_counters(bool full) {
#if ENABLE_COUNTERS
  // Disable counters for a similar reason
  counters_enabled = 0;

  num_skipped_jobs_since_boot = 0;

  uint32_t reset_size = sizeof(Counters) * COUNTERS_LEN;
  if (!full) {
    reset_size -= sizeof(uint32_t) * N_PERSISTENT_COUNTERS;
  }
  memset(counters_data_vm, 0, reset_size);
  write_to_nvm_segmented(reinterpret_cast<uint8_t*>(counters_data_vm),
                         COUNTERS_OFFSET, reset_size, sizeof(Counters));

  counters_enabled = 1;
#endif
}

bool counters_cleared() {
  return (current_counter == INVALID_POINTER) &&
         (prev_counter == INVALID_POINTER);
}

#if !ENABLE_DEMO_COUNTERS
void start_cpu_counter(uint8_t mem_ptr) {
  MY_ASSERT(
      prev_counter == INVALID_POINTER,
      "There is already two counters - prev_counter=%d, current_counter=%d",
      prev_counter, current_counter);

  if (current_counter != INVALID_POINTER) {
    prev_counter = current_counter;
    add_counter(prev_counter, plat_stop_cpu_counter());
    my_printf_debug("Stopping outer CPU counter %d" NEWLINE, prev_counter);
  }
  my_printf_debug("Start CPU counter %d" NEWLINE, mem_ptr);
  current_counter = mem_ptr;
  plat_start_cpu_counter();
}

void stop_cpu_counter(void) {
  MY_ASSERT(current_counter != INVALID_POINTER);

  my_printf_debug("Stop inner CPU counter %d" NEWLINE, current_counter);
  add_counter(current_counter, plat_stop_cpu_counter());
  if (prev_counter != INVALID_POINTER) {
    current_counter = prev_counter;
    my_printf_debug("Restarting outer CPU counter %d" NEWLINE, current_counter);
    plat_start_cpu_counter();
    prev_counter = INVALID_POINTER;
  } else {
    current_counter = INVALID_POINTER;
  }
}
#endif

void report_progress(uint32_t num_jobs) {
#if ENABLE_DEMO_COUNTERS
  static uint8_t last_progress = 0;

  if (!model_vm.run_counter) {
    return;
  }

  uint32_t total_jobs = get_counter(offsetof(Counters, total_jobs));

  my_printf_debug("num_jobs=%d" NEWLINE, num_jobs);

  uint8_t cur_progress = 100 * num_jobs / total_jobs;
  uint32_t progress_preservation_bytes =
      get_counter(offsetof(Counters, progress_preservation_bytes));
  // report only when the percentage is changed to avoid high UART overheads
  if (cur_progress != last_progress) {
    my_printf("P,%d,%d,", cur_progress, progress_preservation_bytes / 1024);

    uint32_t re_execution_macs =
        get_counter(offsetof(Counters, re_execution_macs));
    if (re_execution_macs >= 1024) {
      my_printf("%dK" NEWLINE, re_execution_macs / 1024);
    } else {
      my_printf("%d" NEWLINE, re_execution_macs);
    }

    last_progress = cur_progress;
  }
#endif
}

#endif
