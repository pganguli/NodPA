// Performance counter infrastructure for profiling inference on MCU.
//
// ENABLE_COUNTERS = 0 strips all counter code to no-ops so production builds
// pay zero cost.  When enabled, the Counters struct (data_structures.h) holds
// one 32-bit bucket per metric; the struct is mirrored in SRAM (fast) and
// periodically flushed to NVM so values survive power cycles.
//
// Counter addressing uses offsetof(Counters, field) rather than C++ pointer-
// to-member for compatibility with MSP430's 20-bit address space, where
// ordinary pointer arithmetic can generate 32-bit loads even for small
// structs.  The add_counter / start_cpu_counter / stop_cpu_counter macros
// accept these byte offsets and call into the implementation via
// get_counter_ptr().
//
// ENABLE_PER_LAYER_COUNTERS: maintains one Counters instance per DNN layer
// so overhead can be attributed to individual ops.  Mutually exclusive with
// ENABLE_DEMO_COUNTERS (which accumulates globally for a live progress bar).
//
// start_cpu_counter / stop_cpu_counter support one level of nesting: an
// inner counter can be started while an outer one is running; the outer
// counter is suspended and resumed automatically.  Deeper nesting triggers
// MY_ASSERT.
//
// report_progress: called after each job to emit a "P,<pct>,<bytes>,<macs>"
// line over UART when running in demo-counter mode.  Prints only when the
// percentage changes to keep UART traffic low.

#pragma once

#include <cstdint>

#include "cnn_common.h"
#include "data.h"
#include "data_structures.h"
#include "my_debug.h"

#define ENABLE_COUNTERS 0

// Counter pointers have the form offsetof(Counter, field_name). I use
// offsetof() instead of pointers to member fields like
// https://stackoverflow.com/questions/670734/pointer-to-class-data-member as
// the latter involves pointer arithmetic and is slower for platforms with
// special pointer bitwidths (ex: MSP430)
#if ENABLE_COUNTERS

// Some demo codes assume counters are accumulated across layers
static_assert(!ENABLE_PER_LAYER_COUNTERS || !ENABLE_DEMO_COUNTERS,
              "ENABLE_PER_LAYER_COUNTERS and ENABLE_DEMO_COUNTERS are mutually "
              "exclusive");

extern uint8_t current_counter;
extern uint8_t prev_counter;
extern uint32_t num_skipped_jobs_since_boot;
const uint8_t INVALID_POINTER = 0xff;

void load_counters(void);
void _add_counter(uint8_t counter, uint32_t value);
#define add_demo_counter _add_counter
uint32_t get_counter(uint8_t counter);
#if !ENABLE_DEMO_COUNTERS
#define add_counter _add_counter
void start_cpu_counter(uint8_t mem_ptr);
void stop_cpu_counter(void);
#else
#define add_counter(counter, value)
#define start_cpu_counter(mem_ptr)
#define stop_cpu_counter()
#endif

void print_all_counters();
void reset_counters(bool full);
bool counters_cleared();
void report_progress(uint32_t num_jobs);

#else
#define add_demo_counter(counter, value)
#define add_counter(counter, value)
#define start_cpu_counter(mem_ptr)
#define stop_cpu_counter()
#define print_all_counters()
#define reset_counters(full)
#define report_progress(num_jobs)
#endif

// A global switch for disabling counters temporarily
extern uint8_t counters_enabled;

static inline void enable_counters() { counters_enabled = 1; }

static inline void disable_counters() { counters_enabled = 0; }
