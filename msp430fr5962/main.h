/*
 * MSP430 tick-timer configuration (legacy FreeRTOS defines, kept because
 * common/plat-mcu.cpp includes this header).  NodPA does not run FreeRTOS on
 * the Riotee build, but these symbols are referenced in a few places.
 */

#define configTICK_VECTOR TIMER0_A0_VECTOR
#define configTICK_RATE_HZ (1000)
