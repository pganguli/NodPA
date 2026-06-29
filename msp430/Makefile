# NodPA firmware build for the MSP430FR5994 LaunchPad.
#
# Uses the same TI cl430/lnk430/hex430 toolchain as the msp430fr5962 build.
# Build-only: no flash target (program via CCS or UniFlash).
#
# Override CCS_ROOT to match your install, e.g.:
#   make CCS_ROOT=~/ti/ccs2051/ccs release
#
# Named targets:
#   release   DEBUG=0, VERBOSE=0, -O3              [default]
#   debug     DEBUG=1, VERBOSE=0, no opt + DWARF
#   clean     remove build/
#
# Independent build variables (can be combined freely):
#   EXT_FRAM=1  Use the external SPI FRAM chip (default: 1 per config.h).
#   VERBOSE=1   Per-layer/element tensor dumps (implies DEBUG=1).
#   DEBUG=1     Assertions, testSPI, accuracy prints.
#
# build/data.h and build/data.cpp must be generated first:
#   make -C .. data-<config>   (from the repo root)

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
OUT      ?= $(PRJ)/build/release
TARGET   := intermittent-cnn-msp430fr5994
DRIVERLIB := $(PRJ)/driverlib/MSP430FR5xx_6xx
DSPLIB   := $(ROOT)/TI-DSPLib

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
SRC_ASM := $(ROOT)/tools/our_misc.s

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

SRC_C += $(wildcard $(DRIVERLIB)/*.c)
SRC_C += $(PRJ)/main.c

# ---- objects -----------------------------------------------------------------
OBJ := $(addprefix $(OUT)/,$(notdir $(SRC_CPP:.cpp=.obj))) \
       $(addprefix $(OUT)/,$(notdir $(SRC_C:.c=.obj))) \
       $(addprefix $(OUT)/,$(notdir $(SRC_ASM:.s=.obj)))

vpath %.cpp $(sort $(dir $(SRC_CPP)))
vpath %.c   $(sort $(dir $(SRC_C)))
vpath %.s   $(sort $(dir $(SRC_ASM)))

# ---- flags -------------------------------------------------------------------
INCLUDES := \
  --include_path=$(INC) --include_path=$(CINC) \
  --include_path=$(PRJ) \
  --include_path=$(ROOT) \
  --include_path=$(ROOT)/common \
  --include_path=$(ROOT)/build \
  --include_path=$(DSPLIB)/include \
  --include_path=$(DRIVERLIB)

DEBUG      ?= 0
VERBOSE    ?= 0
STACK_SIZE ?= 0x200

DEFINES = \
  --define=__MSP430__ --define=__MSP430FR5994__ \
  --define=MSP_DISABLE_DIAGNOSTICS \
  --define=DEBUG=$(DEBUG) --define=VERBOSE=$(VERBOSE) \
  --define=USE_ARM_CMSIS=0 --define=LEA_BUFFER_SIZE=4096 \
  --define=EXT_FRAM=$(EXT_FRAM)

OPT ?= --opt_level=3

COMMON_FLAGS = -vmspx --abi=eabi --data_model=large --code_model=large \
  --use_hw_mpy=F5 $(OPT) \
  --silicon_errata=CPU21 --silicon_errata=CPU22 --silicon_errata=CPU40 \
  --advice:power=none --gen_func_subsections=on \
  $(INCLUDES) $(DEFINES)

CFLAGS_C   = $(COMMON_FLAGS) --c99
CFLAGS_CPP = $(COMMON_FLAGS) --c++14

.PHONY: all debug release _build clean

all: release

release:
	$(MAKE) _build \
	  DEBUG=0 VERBOSE=0 EXT_FRAM=$(EXT_FRAM) \
	  OPT="--opt_level=3" STACK_SIZE=0x200 \
	  OUT=$(PRJ)/build/release

debug:
	$(MAKE) _build \
	  DEBUG=1 VERBOSE=$(VERBOSE) EXT_FRAM=$(EXT_FRAM) \
	  OPT="--opt_level=off --symdebug:dwarf" STACK_SIZE=0x500 \
	  OUT=$(PRJ)/build/debug

_build: $(OUT)/$(TARGET).hex

# ---- compile rules -----------------------------------------------------------
$(OUT)/%.obj: %.cpp $(PRJ)/Makefile | $(OUT)
	$(CL) $(CFLAGS_CPP) --obj_directory=$(OUT) -c $<
$(OUT)/%.obj: %.c $(PRJ)/Makefile | $(OUT)
	$(CL) $(CFLAGS_C) --obj_directory=$(OUT) -c $<
$(OUT)/%.obj: %.s $(PRJ)/Makefile | $(OUT)
	$(CL) $(CFLAGS_C) --obj_directory=$(OUT) -c $<

# ---- link + hex --------------------------------------------------------------
$(OUT)/$(TARGET).hex: $(OBJ)
	$(CL) -vmspx --abi=eabi --data_model=large --code_model=large \
	  --use_hw_mpy=F5 --run_linker \
	  --define=__LARGE_CODE_MODEL__ --define=__LARGE_DATA_MODEL__ \
	  --heap_size=160 --stack_size=$(STACK_SIZE) \
	  -i$(LIB) -i$(INC) \
	  --output_file=$(OUT)/$(TARGET).out --map_file=$(OUT)/$(TARGET).map \
	  $(OBJ) $(PRJ)/lnk_msp430fr5994.cmd \
	  --library=libc.a --rom_model
	$(HEX) --intel --romwidth=8 --memwidth=8 --fill=0xFF \
	  -o $(OUT)/$(TARGET).hex $(OUT)/$(TARGET).out

$(OUT):
	mkdir -p $(OUT)

clean:
	rm -rf $(PRJ)/build
