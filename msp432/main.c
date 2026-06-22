// MSP432P401R board bring-up and inference entry point.
//
// prvSetupHardware configures P1.1 and P1.4 as pull-up inputs with falling-
// edge interrupts for the two LaunchPad buttons (S1 and S2).  Master
// interrupts are enabled via MAP_Interrupt_enableMaster().
//
// The clock setup is done by setFrequency(FreqLevel) before prvSetupHardware;
// see tools/dvfs.c for the 11-level frequency table.
//
// PORT1_IRQHandler: ARM Cortex-M4 GPIO ISR.  Reads the status register,
// clears the interrupt flag, then forwards button identity to button_pushed(),
// which is implemented in plat-mcu.cpp (plat-mcu.h declares it extern "C"
// so it can be called from this C file).

#include <driverlib.h>

#include "msp.h"
#include "plat-mcu.h"
#include "tools/dvfs.h"
#include "tools/myuart.h"

static void prvSetupHardware(void);

void main(void) {
  WDT_A->CTL = WDT_A_CTL_PW | WDT_A_CTL_HOLD;  // stop watchdog timer

  setFrequency(FreqLevel);

  prvSetupHardware();

  IntermittentCNNTest();
}

// See timer_a_upmode_gpio_toggle.c in MSP432 examples for code below

#define TIMER_PERIOD 375

static void prvSetupHardware(void) {
  // Ref: MSP432 example gpio_input_interrupt.c

  /* Configuring P1.1 as an input and enabling interrupts */
  MAP_GPIO_setAsInputPinWithPullUpResistor(GPIO_PORT_P1, GPIO_PIN1 | GPIO_PIN4);
  MAP_GPIO_clearInterruptFlag(GPIO_PORT_P1, GPIO_PIN1 | GPIO_PIN4);
  MAP_GPIO_enableInterrupt(GPIO_PORT_P1, GPIO_PIN1 | GPIO_PIN4);
  MAP_Interrupt_enableInterrupt(INT_PORT1);

  /* Enabling MASTER interrupts */
  MAP_Interrupt_enableMaster();
}

/* GPIO ISR */
void PORT1_IRQHandler(void) {
  uint32_t status;

  status = MAP_GPIO_getEnabledInterruptStatus(GPIO_PORT_P1);
  MAP_GPIO_clearInterruptFlag(GPIO_PORT_P1, status);

  button_pushed(status & GPIO_PIN1, status & GPIO_PIN4);
}
