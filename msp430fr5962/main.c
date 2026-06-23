/*
 * MSP430FR5962 (Riotee board) board bring-up and inference entry point for
 * NodPA.  Adapted from the MSP430FR5994 LaunchPad main.c.
 *
 * Differences from the 5994 LaunchPad:
 *   - No 32.768 kHz LFXT crystal on the Riotee module.  ACLK is sourced from
 *     the internal VLO (~9.4 kHz); MCLK/SMCLK run from the DCO at 16 MHz.
 *   - Debug UART is UCA1 on D1/TX = P2.5 (and D0/RX = P2.6), bridged to USB by
 *     the board's RP2040.  (eUSCI_A0 / P2.0-P2.1 is reserved for the C2C link
 *     to the nRF52 and must not be reconfigured here.)  Pin function selection
 *     for the UART is done in tools/myuart.c:uartinit().
 *   - The external SPI FRAM is on eUSCI_B1 / P5.0-P5.3; those pins are set up
 *     in tools/ext_fram/extfram.c:initSPI(), not here.
 *   - PWRGD_L (P5.4) and PWRGD_H (P5.5) are the capacitor-voltage comparator
 *     outputs; left as inputs for the (future) intermittent-power path.
 *
 * _system_pre_init runs before C runtime initialisation; the watchdog is
 * stopped there so initialisation of .data/.bss cannot time out.
 */

#include <tools/dvfs.h>
#include <tools/myuart.h>

#include "plat-mcu.h"

#include <driverlib.h>

/*-----------------------------------------------------------*/
static void prvSetupHardware(void);
/*-----------------------------------------------------------*/

int main(void) {
  prvSetupHardware();

  /* Initialize UART before IntermittentCNNTest() so that any failure inside
   * (e.g. testSPI triggering a watchdog reset) produces visible output rather
   * than a silent reset loop. */
#if MY_DEBUG
  uartinit();
  print2uart("NodPA FR5962: starting\r\n");
#endif

  IntermittentCNNTest();

  return 0;
}

static void prvSetupHardware(void) {
  /* Stop Watchdog timer. */
  WDT_A_hold(__MSP430_BASEADDRESS_WDT_A__);

  /* Set all GPIO pins to output and low to prevent floating inputs from
   * drawing current.  NOTE: the C2C SPI pins to the nRF52 (P2.0/P2.1 = SIMO/
   * SOMI, P1.5 = CLK, P1.4 = CS) are driven here too; that is harmless while
   * the MSP430 runs NodPA standalone (the nRF52 is idle in the stable-power
   * milestone).  initSPI()/uartinit() reclaim the pins they need afterwards. */
  GPIO_setOutputLowOnPin(GPIO_PORT_P1, 0xFF);
  GPIO_setOutputLowOnPin(GPIO_PORT_P2, 0xFF);
  GPIO_setOutputLowOnPin(GPIO_PORT_P3, 0xFF);
  GPIO_setOutputLowOnPin(GPIO_PORT_P4, 0xFF);
  GPIO_setOutputLowOnPin(GPIO_PORT_PJ, 0xFFFF);
  GPIO_setAsOutputPin(GPIO_PORT_P1, 0xFF);
  GPIO_setAsOutputPin(GPIO_PORT_P2, 0xFF);
  GPIO_setAsOutputPin(GPIO_PORT_P3, 0xFF);
  GPIO_setAsOutputPin(GPIO_PORT_P4, 0xFF);
  GPIO_setAsOutputPin(GPIO_PORT_PJ, 0xFFFF);

  /* Debug UART on UCA1: D1/TX = P2.5 (UCA1TXD), D0/RX = P2.6 (UCA1RXD),
   * secondary module function.  Baud configuration happens in uartinit(). */
  GPIO_setAsPeripheralModuleFunctionOutputPin(GPIO_PORT_P2, GPIO_PIN5,
                                              GPIO_SECONDARY_MODULE_FUNCTION);
  GPIO_setAsPeripheralModuleFunctionInputPin(GPIO_PORT_P2, GPIO_PIN6,
                                             GPIO_SECONDARY_MODULE_FUNCTION);

  /* Capacitor-voltage comparator outputs as inputs (used by the intermittent
   * power path; harmless for stable-power bring-up). */
  GPIO_setAsInputPin(GPIO_PORT_P5, GPIO_PIN4);  /* PWRGD_L */
  GPIO_setAsInputPin(GPIO_PORT_P5, GPIO_PIN5);  /* PWRGD_H */

  /* Set DCO frequency to 16 MHz (also sets the FRAM wait state). */
  setFrequency(FreqLevel);

  /* No LFXT crystal on the Riotee module: ACLK = VLO (~9.4 kHz). */
  CS_initClockSignal(CS_ACLK, CS_VLOCLK_SELECT, CS_CLOCK_DIVIDER_1);
  CS_initClockSignal(CS_SMCLK, CS_DCOCLK_SELECT, CS_CLOCK_DIVIDER_1);
  CS_initClockSignal(CS_MCLK, CS_DCOCLK_SELECT, CS_CLOCK_DIVIDER_1);

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
/*-----------------------------------------------------------*/
