// PC-simulation stubs for MCU-specific timer and SPI macros.
//
// On a real MCU, plat-mcu.h provides hardware-specific definitions (e.g.
// SysTick for cpu counter, actual FRAM clock dividers).  On the PC target
// these are no-ops or dummy values so the shared code in platform.cpp
// compiles without changes.
//
// Do not include this file directly; include "platform.h" instead.

// IWYU pragma: private, include "platform.h"
#pragma once

#include "data.h"

#define plat_start_cpu_counter()
#define plat_stop_cpu_counter() 1

// Mimics the MSP432 SPI clock divider; not functionally significant on PC.
#define FRAM_FREQ_DIVIDER 6  // the value for MSP432, for simulation
