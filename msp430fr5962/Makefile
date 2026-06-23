# NodPA firmware build for the MSP430FR5962 on the Riotee board.
#
# Mirrors the known-good Riotee dummy build (cl430/lnk430/hex430 from CCS,
# flashed via riotee-probe over SBW) but compiles the full NodPA source tree:
# the common inference engine, the tools layer (UART/DVFS/external-FRAM SPI),
# the TI DSPLib LEA subset, MSP430 driverlib, and the generated model data.
#
# Override CCS_ROOT to match your install, e.g.:
#   make CCS_ROOT=~/ti/ccs2051/ccs release
#
# Targets:
#   debug    no optimization + symbolic debug
#   release  highest optimization (default)
#   clean    remove the Debug/ output directory
#   flash    program the device over SBW via riotee-probe

CCS_ROOT  ?= $(lastword $(sort $(wildcard $(HOME)/ti/ccs*/ccs)))
COMPILER  ?= $(lastword $(sort $(wildcard $(CCS_ROOT)/tools/compiler/ti-cgt-msp430_*)))

ifeq ($(CCS_ROOT),)
  $(error CCS_ROOT not found under $(HOME)/ti/ccs*/ccs. Set CCS_ROOT explicitly.)
endif
ifeq ($(COMPILER),)
  $(error No ti-cgt-msp430_* compiler found. Set COMPILER explicitly.)
endif

CL   := $(COMPILER)/bin/cl430
LNK  := $(COMPILER)/bin/lnk430
HEX  := $(COMPILER)/bin/hex430
LIB  := $(COMPILER)/lib
INC  := $(CCS_ROOT)/ccs_base/msp430/include
CINC := $(COMPILER)/include

PRJ      := $(CURDIR)
ROOT     := $(abspath $(CURDIR)/..)
OUT      := $(PRJ)/Debug
TARGET   := intermittent-cnn-msp430fr5962
DRIVERLIB := $(ROOT)/msp430/driverlib/MSP430FR5xx_6xx
DSPLIB   := $(ROOT)/TI-DSPLib

# ---- source lists -----------------------------------------------------------

# Common inference engine (MCU platform -> plat-mcu.cpp, NOT plat-pc.cpp).
COMMON_CPP := \
  intermittent-cnn.cpp op_handlers.cpp op_utils.cpp conv.cpp counters.cpp \
  fc.cpp pooling.cpp cnn_common.cpp my_debug.cpp plat-mcu.cpp platform.cpp \
  my_dsplib.cpp layers.cpp softmax.cpp slice.cpp dropout.cpp argmax.cpp \
  gather.cpp broadcasted_binary_op.cpp
SRC_CPP := $(addprefix $(ROOT)/common/,$(COMMON_CPP)) $(ROOT)/build/data.cpp

# Tools: C sources + the delay-loop assembler routine.
SRC_C   := $(ROOT)/tools/dvfs.c $(ROOT)/tools/myuart.c \
           $(ROOT)/tools/ext_fram/extfram.c
SRC_ASM := $(ROOT)/tools/our_misc.s

# TI DSPLib subset actually referenced by my_dsplib.cpp + the LEA runtime.
DSP_C := \
  source/matrix/msp_matrix_mpy_q15.c \
  source/vector/msp_add_q15.c source/vector/msp_offset_q15.c \
  source/vector/msp_mac_q15.c source/vector/msp_max_q15.c \
  source/vector/msp_min_q15.c source/vector/msp_mpy_q15.c \
  source/vector/msp_scale_q15.c \
  source/utility/msp_deinterleave_q15.c source/utility/msp_interleave_q15.c \
  source/utility/msp_fill_q15.c \
  source/lea/msp_lea_init.c source/lea/msp_lea_isr.c \
  source/lea/msp_lea_flags.c source/lea/msp_lea_patch.c
SRC_C += $(addprefix $(DSPLIB)/,$(DSP_C))

# MSP430 driverlib (whole FR5xx_6xx family library).
SRC_C += $(wildcard $(DRIVERLIB)/*.c)

# The board entry point.
SRC_C += $(PRJ)/main.c

# ---- objects ----------------------------------------------------------------
OBJ := $(addprefix $(OUT)/,$(notdir $(SRC_CPP:.cpp=.obj))) \
       $(addprefix $(OUT)/,$(notdir $(SRC_C:.c=.obj))) \
       $(addprefix $(OUT)/,$(notdir $(SRC_ASM:.s=.obj)))

vpath %.cpp $(sort $(dir $(SRC_CPP)))
vpath %.c   $(sort $(dir $(SRC_C)))
vpath %.s   $(sort $(dir $(SRC_ASM)))

# ---- flags ------------------------------------------------------------------
INCLUDES := \
  --include_path=$(INC) --include_path=$(CINC) \
  --include_path=$(PRJ) \
  --include_path=$(ROOT) \
  --include_path=$(ROOT)/common \
  --include_path=$(ROOT)/build \
  --include_path=$(DSPLIB)/include \
  --include_path=$(DRIVERLIB)

# MY_DEBUG=0: no UART, no debug prints (production).
# MY_DEBUG=1: UART enabled, basic progress output.
MY_DEBUG ?= 0

DEFINES = \
  --define=__MSP430FR5962__ --define=MSP_DISABLE_DIAGNOSTICS \
  --define=MY_DEBUG=$(MY_DEBUG) --define=USE_ARM_CMSIS=0 --define=LEA_BUFFER_SIZE=1884

OPT ?= --opt_level=3

# Use recursive (=) so that target-specific MY_DEBUG / OPT overrides take effect
# at recipe evaluation time rather than at parse time.
COMMON_FLAGS = -vmspx --abi=eabi --data_model=large --code_model=large \
  --use_hw_mpy=F5 $(OPT) \
  --silicon_errata=CPU21 --silicon_errata=CPU22 --silicon_errata=CPU40 \
  --advice:power=none --gen_func_subsections=on \
  $(INCLUDES) $(DEFINES)

CFLAGS_C   = $(COMMON_FLAGS) --c99
CFLAGS_CPP = $(COMMON_FLAGS) --c++14

.PHONY: all debug release build clean flash

all: release
debug:   MY_DEBUG := 1
debug:   OPT := --opt_level=off --symdebug:dwarf
debug:   build
release: OPT := --opt_level=3
release: build

build: $(OUT)/$(TARGET).hex

# ---- compile rules ----------------------------------------------------------
$(OUT)/%.obj: %.cpp $(PRJ)/Makefile | $(OUT)
	$(CL) $(CFLAGS_CPP) --obj_directory=$(OUT) -c $<
$(OUT)/%.obj: %.c $(PRJ)/Makefile | $(OUT)
	$(CL) $(CFLAGS_C) --obj_directory=$(OUT) -c $<
$(OUT)/%.obj: %.s $(PRJ)/Makefile | $(OUT)
	$(CL) $(CFLAGS_C) --obj_directory=$(OUT) -c $<

# ---- link + hex -------------------------------------------------------------
# Drive the link through cl430 (--run_linker) so the MSP430X/large-model ISA
# attributes from the objects are propagated to the linker and the correct
# large-model RTS variant is selected from the libc.a index.
$(OUT)/$(TARGET).hex: $(OBJ)
	$(CL) -vmspx --abi=eabi --data_model=large --code_model=large \
	  --use_hw_mpy=F5 --run_linker \
	  --define=__LARGE_CODE_MODEL__ --define=__LARGE_DATA_MODEL__ \
	  --heap_size=160 --stack_size=0x200 \
	  -i$(LIB) -i$(INC) \
	  --output_file=$(OUT)/$(TARGET).out --map_file=$(OUT)/$(TARGET).map \
	  $(OBJ) $(PRJ)/lnk_msp430fr5962.cmd \
	  --library=libc.a --rom_model
	$(HEX) --intel --romwidth=8 --memwidth=8 --fill=0xFF \
	  -o $(OUT)/$(TARGET).hex $(OUT)/$(TARGET).out

$(OUT):
	mkdir -p $(OUT)

clean:
	rm -rf $(OUT)

# SBW programming: bypass --on (not target-power) during programming, then
# power the target on afterwards.
flash: $(OUT)/$(TARGET).hex
	riotee-probe bypass --on
	sleep 1
	riotee-probe program -d msp430 -f $(OUT)/$(TARGET).hex
	riotee-probe target-power --on
