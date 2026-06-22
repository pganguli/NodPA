// Global compile-time configuration for the intermittent-inference firmware.
//
// MY_DEBUG controls how much runtime checking and logging is compiled in:
//   MY_DEBUG_NO_ASSERT (0) — no assertions, no debug output.  Production mode.
//   MY_DEBUG_NORMAL   (1) — assertions enabled; per-sample accuracy printed.
//   MY_DEBUG_LAYERS   (2) — also prints per-layer tensor dumps.
//   MY_DEBUG_VERBOSE  (3) — maximum output, including per-element matrix dumps.
//
// ENABLE_DEMO_COUNTERS — when set, accumulates progress counters across power
//   cycles for demonstration purposes.  Mutually exclusive with
//   ENABLE_PER_LAYER_COUNTERS (see counters.h).

#pragma once

#define ENABLE_DEMO_COUNTERS 0

// Debugging
#define MY_DEBUG_NO_ASSERT 0
#define MY_DEBUG_NORMAL 1
#define MY_DEBUG_LAYERS 2
#define MY_DEBUG_VERBOSE 3

#ifndef MY_DEBUG
#define MY_DEBUG MY_DEBUG_NO_ASSERT
#endif
