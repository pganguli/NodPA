# NodPA firmware build for the MSP432P401R (arm-none-eabi-gcc).
#
# Build-only: no flash target (program via CCS, UniFlash, or OpenOCD).
#
# ⚠  MISSING DEPENDENCY — this Makefile will FAIL at compile.
#    The MSP432P4xx driverlib (SimpleLink MSP432 SDK, typically at
#    ~/ti/simplelink_msp432p4_sdk_*/driverlib/MSP432P4xx/) is NOT installed
#    on this system and is NOT in the repository.  It provides driverlib.h and
#    the peripheral source files required by plat-mcu.cpp and msp432/main.c.
#    Add it and set DRIVERLIB= on the command line, e.g.:
#      make DRIVERLIB=~/ti/simplelink_msp432p4_sdk_3_40_00_12/driverlib/MSP432P4xx release
#
# Named targets:
#   release   DEBUG=0, VERBOSE=0, -O3              [default]
#   debug     DEBUG=1, VERBOSE=0, no opt + -g
#   clean     remove build/
#
# Independent build variables:
#   EXT_FRAM=1  Use the external SPI FRAM chip (default: 1 per config.h).
#   VERBOSE=1   Per-layer/element tensor dumps (implies DEBUG=1).
#   DRIVERLIB   Path to MSP432P4xx driverlib source directory (not set).

ROOT     := $(abspath $(CURDIR)/..)
PRJ      := $(CURDIR)
TARGET   := intermittent-cnn-msp432p401r
OUT      ?= $(PRJ)/build/release

# Toolchain
CC  := arm-none-eabi-gcc
CXX := arm-none-eabi-g++
LD  := arm-none-eabi-g++
HEX := arm-none-eabi-objcopy

# MSP432 driverlib (SimpleLink SDK) — set or export before building.
DRIVERLIB ?=

# CCS ARM include path for msp432p401r.h (device CMSIS header).
CCS_ROOT  ?= $(lastword $(sort $(wildcard $(HOME)/ti/ccs*/ccs)))
CCS_ARM_INC ?= $(CCS_ROOT)/ccs_base/arm/include

DSPLIB   := $(ROOT)/TI-DSPLib
CMSIS    := $(ROOT)/ARM-CMSIS/CMSIS

# ---- source lists -------------------------------------------------------------
COMMON_CPP := \
  intermittent-cnn.cpp op_handlers.cpp op_utils.cpp conv.cpp counters.cpp \
  fc.cpp pooling.cpp cnn_common.cpp my_debug.cpp plat-mcu.cpp platform.cpp \
  my_dsplib.cpp layers.cpp softmax.cpp slice.cpp dropout.cpp argmax.cpp \
  gather.cpp broadcasted_binary_op.cpp
SRC_CPP := $(addprefix $(ROOT)/common/,$(COMMON_CPP)) $(ROOT)/build/data.cpp

EXT_FRAM ?= 1
ifeq ($(EXT_FRAM),1)
SRC_C   := $(ROOT)/tools/dvfs.c $(ROOT)/tools/myuart.c \
           $(ROOT)/tools/ext_fram/extfram.c
else
SRC_C   := $(ROOT)/tools/dvfs.c $(ROOT)/tools/myuart.c
endif
SRC_ASM :=   # no assembler sources on MSP432 (arm-none-eabi)

# CMSIS-DSP (mirrors root CMakeLists.txt arm_cmsis_dsp target)
CMSIS_C := \
  $(CMSIS)/DSP/Source/BasicMathFunctions/arm_add_q15.c \
  $(CMSIS)/DSP/Source/BasicMathFunctions/arm_dot_prod_q15.c \
  $(CMSIS)/DSP/Source/BasicMathFunctions/arm_mult_q15.c \
  $(CMSIS)/DSP/Source/BasicMathFunctions/arm_offset_q15.c \
  $(CMSIS)/DSP/Source/BasicMathFunctions/arm_scale_q15.c \
  $(CMSIS)/DSP/Source/MatrixFunctions/arm_mat_init_q15.c \
  $(CMSIS)/DSP/Source/MatrixFunctions/arm_mat_mult_fast_q15.c \
  $(CMSIS)/DSP/Source/StatisticsFunctions/arm_max_q15.c \
  $(CMSIS)/DSP/Source/StatisticsFunctions/arm_min_q15.c \
  $(CMSIS)/DSP/Source/SupportFunctions/arm_fill_q15.c

SRC_C += $(CMSIS_C)

# MSP432P4xx driverlib (set DRIVERLIB to the SDK driverlib directory)
ifneq ($(DRIVERLIB),)
SRC_C += $(wildcard $(DRIVERLIB)/*.c)
endif

# Board entry + startup
SRC_C += $(PRJ)/main.c $(PRJ)/system_msp432p401r.c $(PRJ)/startup_msp432p401r_gcc.c
SRC_CPP += $(ROOT)/build/data.cpp

# ---- objects -----------------------------------------------------------------
OBJ := $(addprefix $(OUT)/,$(notdir $(SRC_CPP:.cpp=.o))) \
       $(addprefix $(OUT)/,$(notdir $(SRC_C:.c=.o)))

vpath %.cpp $(sort $(dir $(SRC_CPP)))
vpath %.c   $(sort $(dir $(SRC_C)))

# ---- flags -------------------------------------------------------------------
CPU_FLAGS := -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb

INCLUDES := \
  -I$(ROOT)/common \
  -I$(ROOT)/build \
  -I$(ROOT) \
  -I$(PRJ) \
  -I$(DSPLIB)/include \
  -I$(ROOT)/msp430-compat \
  -I$(CMSIS)/Core/Include \
  -I$(CMSIS)/DSP/Include \
  -I$(CCS_ARM_INC)

ifneq ($(DRIVERLIB),)
INCLUDES += -I$(DRIVERLIB)
endif

DEBUG      ?= 0
VERBOSE    ?= 0
STACK_SIZE ?= 0x800
OPT        ?= -O3

DEFINES := \
  -D__MSP432__ -D__MSP432P401R__ \
  -DDEBUG=$(DEBUG) -DVERBOSE=$(VERBOSE) \
  -DUSE_ARM_CMSIS=1 -DARM_MATH_MATRIX_CHECK \
  -DEXT_FRAM=$(EXT_FRAM)

COMMON_FLAGS := $(CPU_FLAGS) $(OPT) -ffunction-sections -fdata-sections \
                -Wall -Wextra -Wno-unused-parameter \
                $(INCLUDES) $(DEFINES)
CFLAGS   := $(COMMON_FLAGS) -std=c11
CXXFLAGS := $(COMMON_FLAGS) -std=c++14 -fno-exceptions -fno-rtti

LDFLAGS := $(CPU_FLAGS) -T$(PRJ)/msp432p401r.lds \
           -Wl,--gc-sections -Wl,-Map=$(OUT)/$(TARGET).map \
           -Wl,--defsym,__STACK_SIZE=$(STACK_SIZE) \
           -lm -lstdc++

.PHONY: all release debug _build clean

all: release

release:
	$(MAKE) _build \
	  DEBUG=0 VERBOSE=0 EXT_FRAM=$(EXT_FRAM) \
	  OPT=-O3 STACK_SIZE=0x800 \
	  OUT=$(PRJ)/build/release

debug:
	$(MAKE) _build \
	  DEBUG=1 VERBOSE=$(VERBOSE) EXT_FRAM=$(EXT_FRAM) \
	  OPT="-O0 -g3" STACK_SIZE=0x1000 \
	  OUT=$(PRJ)/build/debug

_build:
ifeq ($(DRIVERLIB),)
	$(error MSP432P4xx driverlib not found. \
	  Install the SimpleLink MSP432 SDK and pass DRIVERLIB= e.g.: \
	  make DRIVERLIB=~/ti/simplelink_msp432p4_sdk_3_40_00_12/driverlib/MSP432P4xx release)
endif
	$(MAKE) $(OUT)/$(TARGET).hex

# ---- compile rules -----------------------------------------------------------
$(OUT)/%.o: %.cpp $(PRJ)/Makefile | $(OUT)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUT)/%.o: %.c $(PRJ)/Makefile | $(OUT)
	$(CC) $(CFLAGS) -c $< -o $@

# ---- link + hex --------------------------------------------------------------
$(OUT)/$(TARGET).elf: $(OBJ)
ifeq ($(DRIVERLIB),)
	$(error MSP432P4xx driverlib not found. \
	  Install the SimpleLink MSP432 SDK and set DRIVERLIB= to the \
	  driverlib/MSP432P4xx directory, e.g.: \
	  make DRIVERLIB=~/ti/simplelink_msp432p4_sdk_3_40_00_12/driverlib/MSP432P4xx release)
endif
	$(LD) $(LDFLAGS) -o $@ $^

$(OUT)/$(TARGET).hex: $(OUT)/$(TARGET).elf
	$(HEX) -O ihex $< $@

$(OUT):
	mkdir -p $(OUT)

clean:
	rm -rf $(PRJ)/build
