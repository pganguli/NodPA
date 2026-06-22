/*
 * MSP430 tick-timer configuration for FreeRTOS (if used).
 *
 * configTICK_VECTOR: the hardware timer interrupt that drives the RTOS
 *   tick — Timer0_A0 compare output on MSP430FR5994.
 * configTICK_RATE_HZ: 1000 Hz (1 ms tick).  The comment about "non-real time
 *   simulated environment" is a relic from FreeRTOS Windows port boilerplate;
 *   on the MSP430 the tick runs at the real DCO frequency.
 */

#define configTICK_VECTOR TIMER0_A0_VECTOR
#define configTICK_RATE_HZ (1000)
