// MCU hardware backend: DMA memcpy, SPI-FRAM I/O, GPIO indicators, and DVFS.
//
// This file provides the platform-specific implementations of read_from_nvm,
// write_to_nvm, my_memcpy, and related functions for MSP430FR5994,
// MSP430FR5962 (Riotee), and MSP432P401R.
//
// DMA MEMCPY
//   On MSP430: DMA channel 0 is used for word-width (16-bit) transfers,
//     software-triggered (DMAREQ).  The transfer fires synchronously — code
//     does not check a done flag because the channel idles immediately after.
//   On MSP432: UDMA channel 0 uses auto-mode with interrupt on completion.
//     DMA_INT1_IRQHandler (defined in startup_msp432p401r_gcc.c) fires on
//     completion; a busy-wait on MAP_DMA_isChannelEnabled() makes the call
//     synchronous from the caller's point of view.
//
// EXTERNAL FRAM (NVM)
//   All NVM I/O goes through SPI to an external FRAM chip via extfram.h.
//   SPI_READ / SPI_WRITE2 are the low-level byte-transfer primitives.
//   SPI_WAIT_DMA() waits for an in-flight DMA SPI transfer to complete.
//
// GPIO INDICATORS
//   notify_layer_finished() / notify_model_finished() pulse GPIO pins that an
//   external oscilloscope or the intermittent power supply hardware can detect.
//   This is the mechanism used by measure-intermittent.py to count inference
//   completions across power cycles.
//
// IntermittentCNNTest()
//   The top-level MCU entry point (called from msp430/main.c and
//   msp432/main.c). Initialises SPI/UART, runs a fixed number of stable-power
//   test iterations when the reset button is held, then enters the infinite
//   intermittent loop.

#include <driverlib.h>
#ifdef __MSP430__
#include <DSPLib.h>
#include <msp430.h>

#include "main.h"
#elif defined(__MSP432__)
#include <msp432.h>
#endif
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "cnn_common.h"
#include "double_buffering.h"
#include "counters.h"
#include "data.h"
#include "intermittent-cnn.h"
#include "my_debug.h"
#include "platform.h"
#include "tools/dvfs.h"
#include "tools/ext_fram/extfram.h"
#include "tools/myuart.h"
#include "tools/our_misc.h"

#ifdef __MSP430__
#define DATA_SECTION_NVM _Pragma("DATA_SECTION(\".nvm\")")
#else
#define DATA_SECTION_NVM
#endif

#ifdef __MSP432__
uint32_t last_cyccnt = 0;
#endif

#ifdef __MSP430__

#define MY_DMA_CHANNEL DMA_CHANNEL_0

#endif

void my_memcpy(void* dest, const void* src, size_t n) {
#ifdef __MSP430__
  DMA0CTL = 0;

  DMACTL0 &= 0xFF00;
  // set DMA transfer trigger for channel 0
  DMACTL0 |= DMA0TSEL__DMAREQ;

  DMA_setSrcAddress(MY_DMA_CHANNEL, (uint32_t)src, DMA_DIRECTION_INCREMENT);
  DMA_setDstAddress(MY_DMA_CHANNEL, (uint32_t)dest, DMA_DIRECTION_INCREMENT);
  /* transfer size is in words (2 bytes) */
  DMA0SZ = n >> 1;
  DMA0CTL |= DMAEN + DMA_TRANSFER_BLOCK + DMA_SIZE_SRCWORD_DSTWORD;
  DMA0CTL |= DMAREQ;
#elif defined(__MSP432__)
  MAP_DMA_enableModule();
  MAP_DMA_setControlBase(controlTable);
  MAP_DMA_setChannelControl(
      DMA_CH0_RESERVED0 | UDMA_PRI_SELECT,  // Channel 0, PRImary channel
      // re-arbitrate after 1024 (maximum) items
      // an item is 16-bit
      UDMA_ARB_1024 | UDMA_SIZE_16 | UDMA_SRC_INC_16 | UDMA_DST_INC_16);
  // Use the first configurable DMA interrupt handler DMA_INT1_IRQHandler,
  // which is defined below (overriding weak symbol in startup*.c)
  MAP_DMA_assignInterrupt(DMA_INT1, 0);
  MAP_Interrupt_enableInterrupt(INT_DMA_INT1);
  MAP_Interrupt_disableSleepOnIsrExit();
  MAP_DMA_setChannelTransfer(
      DMA_CH0_RESERVED0 | UDMA_PRI_SELECT,
      UDMA_MODE_AUTO,  // Set as auto mode with no need to retrigger after each
                       // arbitration
      const_cast<void*>(src), dest,
      n >> 1  // transfer size in items
  );
  curDMATransmitChannelNum = 0;
  MAP_DMA_enableChannel(0);
  MAP_DMA_requestSoftwareTransfer(0);
  while (MAP_DMA_isChannelEnabled(0)) {
  }
#endif
}

void my_memcpy_from_parameters(void* dest, const ParameterInfo* param,
                               uint32_t offset_in_bytes, size_t n) {
  MY_ASSERT(offset_in_bytes + n <= PARAMETERS_DATA_LEN);
#if ENABLE_COUNTERS
  if (counters_enabled) {
    add_counter(offsetof(Counters, nvm_read_parameters), n);
    my_printf_debug(
        "Recorded %lu bytes fetched from parameters, accumulated=%" PRIu32
            NEWLINE,
        n, get_counter(offsetof(Counters, nvm_read_parameters)));
  }
#endif

  read_from_nvm(dest,
                PARAMETERS_OFFSET + param->params_offset + offset_in_bytes, n);
}

void read_from_nvm(void* vm_buffer, uint32_t nvm_offset, size_t n) {
  SPI_ADDR addr;
  addr.L = nvm_offset;
  MY_ASSERT(n <= 1024);
  SPI_READ(&addr, reinterpret_cast<uint8_t*>(vm_buffer), n);
}

struct GPIOPin {
  uint16_t port;
  uint16_t pin;
};

static const GPIOPin indicators[] = {
#if defined(__MSP430FR5962__)
    // D3 = P2.4 → TA1.0 timer output (layer-finished pulse, zero CPU cost).
    {GPIO_PORT_P2, GPIO_PIN4},  // indicators[0]: notify_layer_finished()
#elif defined(__MSP430__)
    {GPIO_PORT_P4, GPIO_PIN7},  // used in notify_layer_finished()
    {GPIO_PORT_P1, GPIO_PIN5},  // TODO: check if it works
#else
    {GPIO_PORT_P5, GPIO_PIN4},  // used in notify_layer_finished()
    {GPIO_PORT_P4, GPIO_PIN7},
#endif
};

static const GPIOPin gpio_flags[] = {
#if defined(__MSP430FR5962__)
    // Riotee: D4 = P4.6 (first_run trigger), plus the two comparator outputs.
    {GPIO_PORT_P4, GPIO_PIN6},
    {GPIO_PORT_P5, GPIO_PIN4},  // PWRGD_L
    {GPIO_PORT_P5, GPIO_PIN5},  // PWRGD_H
#elif defined(__MSP430__)
    // TODO: check if these work on MSP430
    {GPIO_PORT_P3, GPIO_PIN7},
    {GPIO_PORT_P3, GPIO_PIN6},
    {GPIO_PORT_P3, GPIO_PIN5},
#else
    {GPIO_PORT_P2, GPIO_PIN7},
    {GPIO_PORT_P2, GPIO_PIN6},
    {GPIO_PORT_P2, GPIO_PIN4},
#endif
};

void write_to_nvm(const void* vm_buffer, uint32_t nvm_offset, size_t n,
                  uint16_t timer_delay) {
  SPI_ADDR addr;
  addr.L = nvm_offset;
  check_nvm_write_address(nvm_offset, n);
  MY_ASSERT(n <= 1024);
  SPI_WRITE2(&addr, reinterpret_cast<const uint8_t*>(vm_buffer), n,
             timer_delay);
  if (!timer_delay) {
    SPI_WAIT_DMA();
  }
}

void my_erase() { eraseFRAM2(0x00); }

void copy_data_to_nvm(void) {
  write_to_nvm_segmented(samples_data, SAMPLES_OFFSET, SAMPLES_DATA_LEN);
  write_to_nvm_segmented(parameters_data, PARAMETERS_OFFSET,
                         PARAMETERS_DATA_LEN);
  write_to_nvm_segmented(node_flags_data, NODE_FLAGS_OFFSET,
                         NODE_FLAGS_DATA_LEN);
}

[[noreturn]] void ERROR_OCCURRED(void) { while (1); }

#if defined(__MSP430FR5962__)
// Riotee: model-finished pulse on D2 = P2.3 (TA0.0 timer compare output).
//         layer-finished  pulse on D3 = P2.4 (TA1.0 timer compare output).
//         first_run trigger  on D4 = P4.6: bridge D4 to GND at boot to force first_run().
// (D6 / PJ.6 is shared with nRF52 P1.03 which pulls it permanently low.)
#define GPIO_COUNTER_PORT GPIO_PORT_P2
#define GPIO_COUNTER_PIN GPIO_PIN3
#define GPIO_RESET_PORT GPIO_PORT_P4
#define GPIO_RESET_PIN GPIO_PIN6
#elif defined(__MSP430__)
#define GPIO_COUNTER_PORT GPIO_PORT_P8
#define GPIO_COUNTER_PIN GPIO_PIN0
#define GPIO_RESET_PORT GPIO_PORT_P5
#define GPIO_RESET_PIN GPIO_PIN7
#else
#define GPIO_COUNTER_PORT GPIO_PORT_P5
#define GPIO_COUNTER_PIN GPIO_PIN5
#define GPIO_RESET_PORT GPIO_PORT_P2
#define GPIO_RESET_PIN GPIO_PIN5
#endif

#define STABLE_POWER_ITERATIONS 10

#if defined(__MSP430FR5962__)
/* nRF52 P1.03 and MSP430 PJ.6 share the D6 net.  The nRF52 may drive PJ.6
 * low (output-low or internal pull-down), making need_reset() permanently
 * true and triggering first_run() on every power cycle.  Use a magic byte
 * in the internal FRAM .nvm section (type=NOINIT) instead: the linker drops
 * any .cinit record for it so the value survives power cycles.  On a
 * factory-fresh or erased chip the byte is 0xFF; first_run() writes 0xA5.
 * Subsequent power cycles find 0xA5 and go directly to the resume path. */
__attribute__((section(".nvm"))) static uint8_t first_run_done;
#define FIRST_RUN_DONE_MAGIC 0xA5u
#endif

bool need_reset() {
  return !GPIO_getInputPinValue(GPIO_RESET_PORT, GPIO_RESET_PIN);
}

void IntermittentCNNTest() {
  GPIO_setAsOutputPin(GPIO_COUNTER_PORT, GPIO_COUNTER_PIN);
  GPIO_setOutputLowOnPin(GPIO_COUNTER_PORT, GPIO_COUNTER_PIN);
  for (size_t idx = 0; idx < sizeof(indicators) / sizeof(indicators[0]);
       idx++) {
    GPIO_setAsOutputPin(indicators[idx].port, indicators[idx].pin);
    GPIO_setOutputLowOnPin(indicators[idx].port, indicators[idx].pin);
  }
  GPIO_setAsInputPinWithPullUpResistor(GPIO_RESET_PORT, GPIO_RESET_PIN);
  for (size_t idx = 0; idx < sizeof(gpio_flags) / sizeof(gpio_flags[0]);
       idx++) {
    GPIO_setAsInputPinWithPullUpResistor(gpio_flags[idx].port,
                                         gpio_flags[idx].pin);
  }

#if defined(__MSP430FR5962__)
  // Riotee module LED is LED_CTRL = PJ.0 (active high), shared with the nRF52.
  GPIO_setAsOutputPin(GPIO_PORT_PJ, GPIO_PIN0);
  GPIO_setOutputHighOnPin(GPIO_PORT_PJ, GPIO_PIN0);

  // P2.3 (D2, GPIO_COUNTER_PIN) → TA0.0 timer compare output: zero-CPU inference pulse.
  P2SEL0 |= BIT3;  P2SEL1 &= ~BIT3;  P2DIR |= BIT3;
  // P2.4 (D3, indicators[0]) → TA1.0 timer compare output: zero-CPU layer pulse.
  P2SEL0 |= BIT4;  P2SEL1 &= ~BIT4;  P2DIR |= BIT4;
  // Both timers: continuous mode, ACLK (VLO ~9.4 kHz). 5 ms pulse ≈ 47 ACLK ticks.
  TA0CTL = TASSEL__ACLK | MC__CONTINUOUS | TACLR;
  TA1CTL = TASSEL__ACLK | MC__CONTINUOUS | TACLR;
#else
  GPIO_setAsOutputPin(GPIO_PORT_P1, GPIO_PIN0);
  GPIO_setOutputHighOnPin(GPIO_PORT_P1, GPIO_PIN0);
#endif

  // sleep to wait for external FRAM
  // 5ms / (1/f)
  our_delay_cycles(5E-3 * getFrequency(FreqLevel));

  initSPI();
#if MY_DEBUG >= MY_DEBUG_NORMAL
  if (testSPI() != 0) {
    uartinit();
    print2uart("testSPI FAILED - check FRAM wiring\r\n");
    volatile uint16_t counter = 1000;
    while (counter--);
    WDTCTL = 0;
  }
#endif

  load_model_from_nvm();

#if ENABLE_COUNTERS
  load_counters();
#endif

#if defined(__MSP430FR5962__)
  if (need_reset() || first_run_done != FIRST_RUN_DONE_MAGIC) {
#else
  if (need_reset()) {
#endif
#if MY_DEBUG >= MY_DEBUG_NORMAL
    uartinit();
    // To get counters in NVM after intermittent tests
    print_all_counters();
#endif

    first_run();

#if defined(__MSP430FR5962__)
    first_run_done = FIRST_RUN_DONE_MAGIC;
#endif

    notify_model_finished();

    for (uint8_t idx = 0; idx < STABLE_POWER_ITERATIONS; idx++) {
      run_cnn_tests(1);
    }

#if MY_DEBUG >= MY_DEBUG_NORMAL
    my_printf("Done testing run" NEWLINE);
    // For platforms where counters are recorded in VM (ex: MSP432)
    print_all_counters();
#endif

    get_model()->run_counter = 0;
    commit_model();

    inference_results_vm.correct    = 0;
    inference_results_vm.total      = 0;
    inference_results_vm.sample_idx = 0;
    commit_versioned_data<InferenceResults>(0);

    while (1);
  }

#if ENABLE_DEMO_COUNTERS
  uartinit();
#endif
  while (1) {
    run_cnn_tests(1);
  }
}

void button_pushed(uint16_t button1_status, uint16_t button2_status) {
  my_printf_debug("button1_status=%d button2_status=%d" NEWLINE, button1_status,
                  button2_status);
}

static void gpio_pulse(uint8_t port, uint16_t pin) {
#if defined(__MSP430FR5962__)
  if (port == GPIO_PORT_P2 && pin == GPIO_PIN3) {
    // P2.3 → TA0.0: set HIGH via Mode 0, arm Reset compare ~5 ms out.
    TA0CCTL0 = OUT;
    TA0CCR0  = TA0R + 47;
    TA0CCTL0 = OUTMOD_5;
  } else if (port == GPIO_PORT_P2 && pin == GPIO_PIN4) {
    // P2.4 → TA1.0: same pattern using Timer_A1 CCR0.
    TA1CCTL0 = OUT;
    TA1CCR0  = TA1R + 47;
    TA1CCTL0 = OUTMOD_5;
  } else {
    GPIO_setOutputHighOnPin(port, pin);
    our_delay_cycles(5E-3 * getFrequency(FreqLevel));
    GPIO_setOutputLowOnPin(port, pin);
  }
#else
  // Trigger a short peak so that multiple inferences in long power cycles are
  // correctly recorded
  GPIO_setOutputHighOnPin(port, pin);
  our_delay_cycles(5E-3 * getFrequency(FreqLevel));
  GPIO_setOutputLowOnPin(port, pin);
#endif
}

void notify_layer_finished(void) { notify_indicator(0); }

void notify_model_finished(void) {
#if MY_DEBUG >= MY_DEBUG_NORMAL
  my_printf("." NEWLINE);
#endif
  gpio_pulse(GPIO_COUNTER_PORT, GPIO_COUNTER_PIN);
}

void notify_indicator(uint8_t idx) {
#if !ENABLE_DEMO_COUNTERS && MY_DEBUG >= MY_DEBUG_NORMAL
  my_printf("I%d" NEWLINE, idx);
#endif
  gpio_pulse(indicators[idx].port, indicators[idx].pin);
}

bool read_gpio_flag(GPIOFlag flag) {
  uint8_t idx = static_cast<uint8_t>(flag);
  return !GPIO_getInputPinValue(gpio_flags[idx].port, gpio_flags[idx].pin);
}

void save_model_output_data() {}
