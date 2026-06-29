# NodPA PC (native host) inference build — GNU make / g++.
#
# Builds the same inference engine as the MCU targets but using plat-pc.cpp
# instead of plat-mcu.cpp, so inference can be run on the host for testing
# and model validation without hardware.
#
# Named targets:
#   release   DEBUG=0, VERBOSE=0, -O3              [default]
#   debug     DEBUG=1, VERBOSE=0, -O0 -g3 + ASan
#   clean     remove build/
#
# Independent build variables:
#   VERBOSE=1   Per-layer/element tensor dumps (implies DEBUG=1).
#   DATA_DIR    Directory containing generated data.h / data.cpp.
#               Default: $(ROOT)/build  (shared with MCU targets).

ROOT    := $(abspath $(CURDIR)/..)
PRJ     := $(CURDIR)
TARGET  := intermittent-cnn
OUT     ?= $(PRJ)/build/release

CC  := gcc
CXX := g++

DATA_DIR ?= $(ROOT)/build
CMSIS    := $(ROOT)/ARM-CMSIS/CMSIS
DSPLIB   := $(ROOT)/TI-DSPLib

# ---- source lists ------------------------------------------------------------
COMMON_CPP := \
  intermittent-cnn.cpp op_handlers.cpp op_utils.cpp conv.cpp counters.cpp \
  fc.cpp pooling.cpp cnn_common.cpp my_debug.cpp plat-pc.cpp platform.cpp \
  my_dsplib.cpp layers.cpp softmax.cpp slice.cpp dropout.cpp argmax.cpp \
  gather.cpp broadcasted_binary_op.cpp
SRC_CPP := $(addprefix $(ROOT)/common/,$(COMMON_CPP)) $(DATA_DIR)/data.cpp

# CMSIS-DSP (same subset as CMakeLists.txt arm_cmsis_dsp target)
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

# TI DSPLib subset (MSP430-compat shim provides MSP430 types on the host)
DSP_C := \
  $(DSPLIB)/source/matrix/msp_matrix_mpy_q15.c \
  $(DSPLIB)/source/vector/msp_add_q15.c \
  $(DSPLIB)/source/vector/msp_offset_q15.c \
  $(DSPLIB)/source/vector/msp_mac_q15.c \
  $(DSPLIB)/source/vector/msp_max_q15.c \
  $(DSPLIB)/source/vector/msp_min_q15.c \
  $(DSPLIB)/source/vector/msp_mpy_q15.c \
  $(DSPLIB)/source/vector/msp_scale_q15.c \
  $(DSPLIB)/source/utility/msp_deinterleave_q15.c \
  $(DSPLIB)/source/utility/msp_interleave_q15.c \
  $(DSPLIB)/source/utility/msp_fill_q15.c

SRC_C := $(CMSIS_C) $(DSP_C)

# ---- objects -----------------------------------------------------------------
OBJ := $(addprefix $(OUT)/,$(notdir $(SRC_CPP:.cpp=.o))) \
       $(addprefix $(OUT)/,$(notdir $(SRC_C:.c=.o)))

vpath %.cpp $(sort $(dir $(SRC_CPP)))
vpath %.c   $(sort $(dir $(SRC_C)))

# ---- flags -------------------------------------------------------------------
INCLUDES := \
  -I$(ROOT)/common \
  -I$(DATA_DIR) \
  -I$(ROOT) \
  -I$(ROOT)/msp430-compat \
  -I$(CMSIS)/Core/Include \
  -I$(CMSIS)/DSP/Include \
  -I$(DSPLIB)/include

DEBUG   ?= 0
VERBOSE ?= 0
OPT     ?= -O3

DEFINES := -DPC_BUILD -DDEBUG=$(DEBUG) -DVERBOSE=$(VERBOSE) \
           -DUSE_ARM_CMSIS=1 -DARM_MATH_MATRIX_CHECK

WARN    := -Wall -Wextra -Wshadow -Wundef -Wno-unused-parameter \
           -Werror=undef -pedantic

COMMON_FLAGS := $(OPT) $(WARN) $(INCLUDES) $(DEFINES)
CFLAGS   := $(COMMON_FLAGS) -std=c11 -Wstrict-prototypes
CXXFLAGS := $(COMMON_FLAGS) -std=c++14 \
            -Wold-style-cast -Wsuggest-override -Wnon-virtual-dtor

ifeq ($(DEBUG),1)
CFLAGS   += -fsanitize=address
CXXFLAGS += -fsanitize=address
LDFLAGS  += -fsanitize=address
endif

LDFLAGS += -lm

.PHONY: all release debug _build clean

all: release

release:
	$(MAKE) _build \
	  DEBUG=0 VERBOSE=0 \
	  OPT=-O3 \
	  OUT=$(PRJ)/build/release

debug:
	$(MAKE) _build \
	  DEBUG=1 VERBOSE=$(VERBOSE) \
	  OPT="-O0 -g3" \
	  OUT=$(PRJ)/build/debug

_build: $(OUT)/$(TARGET)

# ---- compile rules -----------------------------------------------------------
$(OUT)/%.o: %.cpp $(PRJ)/Makefile | $(OUT)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUT)/%.o: %.c $(PRJ)/Makefile | $(OUT)
	$(CC) $(CFLAGS) -c $< -o $@

# ---- link --------------------------------------------------------------------
$(OUT)/$(TARGET): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^

$(OUT):
	mkdir -p $(OUT)

clean:
	rm -rf $(PRJ)/build
