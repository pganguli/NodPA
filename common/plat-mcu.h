// MCU-specific platform header — not included directly; use "platform.h".
//
// Provides plat_start_cpu_counter / plat_stop_cpu_counter, which return
// the number of CPU cycles elapsed between calls.  Used by the counter
// subsystem (counters.h) to attribute overhead to each profiling bucket.
//
//   MSP430: TI DSPLib benchmark timer (msp_benchmarkStart/Stop).
//           Resolution is one MCLK cycle; wraps at 2^32 cycles (~100s at 8MHz).
//
//   MSP432: ARM Cortex-M4 DWT cycle counter (DWT->CYCCNT).
//           The counter must be enabled before use (DWT->CTRL |= 1).
//           last_cyccnt stores the snapshot; stop returns the delta.
//
//   PC:     Both macros are no-ops; stop returns 1 so callers don't divide by
//   zero.
//
// IntermittentCNNTest() is the top-level entry point called from main().
// button_pushed() is the interrupt-context callback for the two buttons on
// the LaunchPad, used to cycle through test modes or trigger manual resets.

// IWYU pragma: private, include "platform.h"
#pragma once

#include <stdint.h>

#include "data.h"
#include "tools/ext_fram/extfram.h"

#ifdef __MSP430__
#include <DSPLib.h>
static inline void plat_start_cpu_counter(void) {
  msp_benchmarkStart(MSP_BENCHMARK_BASE, 1);
}

static inline uint32_t plat_stop_cpu_counter(void) {
  return msp_benchmarkStop(MSP_BENCHMARK_BASE);
}
#elif defined(__MSP432__)
#include <msp.h>
extern uint32_t last_cyccnt;
// ARM Cortex-M4 DWT cycle counter — must enable via DWT->CTRL |= 1 first.
static inline void plat_start_cpu_counter(void) {
  DWT->CTRL |= 1;
  last_cyccnt = DWT->CYCCNT;
}

static inline uint32_t plat_stop_cpu_counter(void) {
  uint32_t ret = DWT->CYCCNT - last_cyccnt;
  DWT->CTRL &= 0XFFFFFFFE;
  return ret;
}
#else
#define plat_start_cpu_counter()
#define plat_stop_cpu_counter() 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

void IntermittentCNNTest(void);
void button_pushed(uint16_t button1_status, uint16_t button2_status);

#ifdef __cplusplus
}
#endif
