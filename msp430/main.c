/*
 * MSP430FR5994 board bring-up and inference entry point.
 *
 * prvSetupHardware:
 *   - Disables the watchdog (must happen before any significant delay).
 *   - Sets all GPIO pins to output/low to prevent floating inputs from
 *     drawing current — critical for accurate energy harvesting experiments.
 *   - Configures P2.0/P2.1 as UART TX/RX (UCA0) for debug output.
 *   - Starts the 32.768 kHz LFXT crystal on PJ.4/PJ.5 (used for ACLK and
 *     the power-supply's timing reference).
 *   - Sets DCO to FreqLevel MHz; MCLK and SMCLK both run at DCO.
 *   - Enables falling-edge interrupts on P5.5 (S2) and P5.6 (S1) for the
 *     two LaunchPad buttons.
 *   - Calls PMM_unlockLPM5() to exit the default LPM5 high-impedance state.
 *
 * _system_pre_init runs before C runtime initialisation.  The watchdog is
 * stopped here too so that the initialisation of .data/.bss does not time out.
 *
 * Port_5 ISR: reads the interrupt status of P5.5/P5.6 and forwards button
 * identity to button_pushed(), which is implemented in plat-mcu.cpp.
 */

#include <tools/dvfs.h>
#include <tools/myuart.h>

#include "plat-mcu.h"

/* Standard demo includes, used so the tick hook can exercise some FreeRTOS
functionality in an interrupt. */
#include <driverlib.h>

#include "main.h"

/*-----------------------------------------------------------*/
/*
 * Configure the hardware as necessary.
 */
static void prvSetupHardware(void);

/*-----------------------------------------------------------*/

int main(void) {
  /* Configure the hardware ready to run the demo. */
  prvSetupHardware();

  IntermittentCNNTest();

  return 0;
}

static void prvSetupHardware(void) {
  /* Stop Watchdog timer. */
  WDT_A_hold(__MSP430_BASEADDRESS_WDT_A__);

  /* Set all GPIO pins to output and low. */
  GPIO_setOutputLowOnPin(GPIO_PORT_P1, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 |
                                           GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 |
                                           GPIO_PIN6 | GPIO_PIN7);
  GPIO_setOutputLowOnPin(GPIO_PORT_P2, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 |
                                           GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 |
                                           GPIO_PIN6 | GPIO_PIN7);
  GPIO_setOutputLowOnPin(GPIO_PORT_P3, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 |
                                           GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 |
                                           GPIO_PIN6 | GPIO_PIN7);
  GPIO_setOutputLowOnPin(GPIO_PORT_P4, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 |
                                           GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 |
                                           GPIO_PIN6 | GPIO_PIN7);
  GPIO_setOutputLowOnPin(
      GPIO_PORT_PJ, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 |
                        GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 | GPIO_PIN8 |
                        GPIO_PIN9 | GPIO_PIN10 | GPIO_PIN11 | GPIO_PIN12 |
                        GPIO_PIN13 | GPIO_PIN14 | GPIO_PIN15);
  GPIO_setAsOutputPin(GPIO_PORT_P1, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 |
                                        GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 |
                                        GPIO_PIN6 | GPIO_PIN7);
  GPIO_setAsOutputPin(GPIO_PORT_P2, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 |
                                        GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 |
                                        GPIO_PIN6 | GPIO_PIN7);
  GPIO_setAsOutputPin(GPIO_PORT_P3, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 |
                                        GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 |
                                        GPIO_PIN6 | GPIO_PIN7);
  GPIO_setAsOutputPin(GPIO_PORT_P4, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 |
                                        GPIO_PIN3 | GPIO_PIN4 | GPIO_PIN5 |
                                        GPIO_PIN6 | GPIO_PIN7);
  GPIO_setAsOutputPin(
      GPIO_PORT_PJ, GPIO_PIN0 | GPIO_PIN1 | GPIO_PIN2 | GPIO_PIN3 | GPIO_PIN4 |
                        GPIO_PIN5 | GPIO_PIN6 | GPIO_PIN7 | GPIO_PIN8 |
                        GPIO_PIN9 | GPIO_PIN10 | GPIO_PIN11 | GPIO_PIN12 |
                        GPIO_PIN13 | GPIO_PIN14 | GPIO_PIN15);

  /* Configure P2.0 - UCA0TXD and P2.1 - UCA0RXD. */
  GPIO_setOutputLowOnPin(GPIO_PORT_P2, GPIO_PIN0);
  GPIO_setAsOutputPin(GPIO_PORT_P2, GPIO_PIN0);
  GPIO_setAsPeripheralModuleFunctionInputPin(GPIO_PORT_P2, GPIO_PIN1,
                                             GPIO_SECONDARY_MODULE_FUNCTION);
  GPIO_setAsPeripheralModuleFunctionOutputPin(GPIO_PORT_P2, GPIO_PIN0,
                                              GPIO_SECONDARY_MODULE_FUNCTION);

  /* Set PJ.4 and PJ.5 for LFXT. */
  GPIO_setAsPeripheralModuleFunctionInputPin(
      GPIO_PORT_PJ, GPIO_PIN4 + GPIO_PIN5, GPIO_PRIMARY_MODULE_FUNCTION);

  // Configure button S1 (P5.6) interrupt and S2 P(5.5)
  GPIO_selectInterruptEdge(GPIO_PORT_P5, GPIO_PIN6,
                           GPIO_HIGH_TO_LOW_TRANSITION);
  GPIO_setAsInputPinWithPullUpResistor(GPIO_PORT_P5, GPIO_PIN6);
  GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN6);
  GPIO_enableInterrupt(GPIO_PORT_P5, GPIO_PIN6);
  GPIO_selectInterruptEdge(GPIO_PORT_P5, GPIO_PIN5,
                           GPIO_HIGH_TO_LOW_TRANSITION);
  GPIO_setAsInputPinWithPullUpResistor(GPIO_PORT_P5, GPIO_PIN5);
  GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN5);
  GPIO_enableInterrupt(GPIO_PORT_P5, GPIO_PIN5);

  GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN6);
  GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN5);

  /* Set DCO frequency to 16 MHz. */
  setFrequency(FreqLevel);

  /* Set external clock frequency to 32.768 KHz. */
  CS_setExternalClockSource(32768, 0);

  /* Set ACLK = LFXT. */
  CS_initClockSignal(CS_ACLK, CS_LFXTCLK_SELECT, CS_CLOCK_DIVIDER_1);

  /* Set SMCLK = DCO with frequency divider of 1. */
  CS_initClockSignal(CS_SMCLK, CS_DCOCLK_SELECT, CS_CLOCK_DIVIDER_1);

  /* Set MCLK = DCO with frequency divider of 1. */
  CS_initClockSignal(CS_MCLK, CS_DCOCLK_SELECT, CS_CLOCK_DIVIDER_1);

  /* Start XT1 with no time out. */
  CS_turnOnLFXT(CS_LFXT_DRIVE_0);

  /* Disable the GPIO power-on default high-impedance mode. */
  PMM_unlockLPM5();
}
/*-----------------------------------------------------------*/

int _system_pre_init(void) {
  /* Stop Watchdog timer. */
  WDT_A_hold(__MSP430_BASEADDRESS_WDT_A__);

  /* Return 1 for segments to be initialised. */
  return 1;
}

/*
 * port 5 interrupt service routine to handle s1 and s2 button press
 *
 */

#pragma vector = PORT5_VECTOR
__interrupt void Port_5(void) {
  GPIO_disableInterrupt(GPIO_PORT_P5, GPIO_PIN6);
  GPIO_disableInterrupt(GPIO_PORT_P5, GPIO_PIN5);

  /* Button pushed, do something if you need to */
  uint16_t status_5_5 = GPIO_getInterruptStatus(GPIO_PORT_P5, GPIO_PIN5),
           status_5_6 = GPIO_getInterruptStatus(GPIO_PORT_P5, GPIO_PIN6);
  button_pushed(status_5_5, status_5_6);

  GPIO_enableInterrupt(GPIO_PORT_P5, GPIO_PIN6);
  GPIO_enableInterrupt(GPIO_PORT_P5, GPIO_PIN5);
  GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN6);
  GPIO_clearInterrupt(GPIO_PORT_P5, GPIO_PIN5);
}

/*-----------------------------------------------------------*/
