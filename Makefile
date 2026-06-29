# NodPA root Makefile — orchestrates all sub-builds and model-data generation.
#
# Model-data targets (write shared build/data.h + build/data.cpp):
#   data-cifar10-dnp    CIFAR-10 ResNet-10, external-FRAM layout
#   data-kws-dnp        Keyword Spotting CNN, external-FRAM layout
#   data-har-dnp        HAR CNN, external-FRAM layout
#   data-har-int        HAR CNN, internal-FRAM layout (required for this config)
#
# Firmware / PC build targets:
#   release             → release-msp430          [default]
#   release-msp430      MSP430FR5994 LaunchPad (cl430, build-only)
#   release-riotee      MSP430FR5962 Riotee module (cl430, flash via make -C msp430fr5962 flash)
#   release-msp432      MSP432P401R (arm-none-eabi-gcc; needs driverlib — see msp432/Makefile)
#   release-pc          PC host binary (g++, for model validation without hardware)
#   debug               → debug-msp430
#   debug-msp430        As release-msp430 with DEBUG=1
#   debug-riotee        As release-riotee with DEBUG=1
#   debug-msp432        As release-msp432 with DEBUG=1
#   debug-pc            As release-pc with DEBUG=1 + ASan
#
# Build variables (passed through to all sub-builds):
#   EXT_FRAM=0/1   NVM backend (default 1 for all targets).
#   VERBOSE=0/1    Tensor dumps on top of DEBUG (implies DEBUG=1).
#   TARGET=msp430  transform.py --target for data-* (default: msp430).
#
# Other:
#   clean           Remove build/ (data.h/data.cpp + PC artifacts) and all
#                   firmware build dirs.
#
# Typical workflows:
#   make data-har-dnp && make release-riotee
#   make data-har-dnp && make debug-riotee EXT_FRAM=1
#   make data-har-int && make release-msp430

ROOT := $(CURDIR)

# Pass-through vars with defaults.
EXT_FRAM ?= 0
VERBOSE  ?= 0
TARGET   ?= msp430

# ---- data-generation targets -------------------------------------------------
# transform.py is run from the repo root (it expects to find dnn-models/ here).
# data-har-int uses --internal-fram because that config exists solely for builds
# that must fit in the device's internal FRAM (pruning_threshold=0).

.PHONY: data-cifar10-dnp data-kws-dnp data-har-dnp data-har-int

data-cifar10-dnp:
	python dnn-models/transform.py --target $(TARGET) --hawaii cifar10-dnp

data-kws-dnp:
	python dnn-models/transform.py --target $(TARGET) --hawaii kws-dnp

data-har-dnp:
	python dnn-models/transform.py --target $(TARGET) --hawaii har-dnp

data-har-int:
	python dnn-models/transform.py --target $(TARGET) --hawaii --internal-fram har-int

# ---- firmware / PC build targets ---------------------------------------------

.PHONY: all release debug \
        release-msp430 debug-msp430 \
        release-riotee debug-riotee \
        release-msp432 debug-msp432 \
        release-pc debug-pc \
        clean

all: release

release: release-msp430

debug: debug-msp430

release-msp430:
	$(MAKE) -C msp430 release EXT_FRAM=$(EXT_FRAM) VERBOSE=$(VERBOSE)

debug-msp430:
	$(MAKE) -C msp430 debug EXT_FRAM=$(EXT_FRAM) VERBOSE=$(VERBOSE)

release-riotee:
	$(MAKE) -C msp430fr5962 release EXT_FRAM=$(EXT_FRAM) VERBOSE=$(VERBOSE)

debug-riotee:
	$(MAKE) -C msp430fr5962 debug EXT_FRAM=$(EXT_FRAM) VERBOSE=$(VERBOSE)

release-msp432:
	$(MAKE) -C msp432 release EXT_FRAM=$(EXT_FRAM) VERBOSE=$(VERBOSE)

debug-msp432:
	$(MAKE) -C msp432 debug EXT_FRAM=$(EXT_FRAM) VERBOSE=$(VERBOSE)

release-pc:
	$(MAKE) -C pc release DATA_DIR=$(ROOT)/build VERBOSE=$(VERBOSE)

debug-pc:
	$(MAKE) -C pc debug DATA_DIR=$(ROOT)/build VERBOSE=$(VERBOSE)

# ---- clean -------------------------------------------------------------------
# Removes generated data files (build/data.h, build/data.cpp) and all firmware
# + PC build artifacts.  The bare build/ directory (CMake artifacts, model
# data) is removed entirely; each sub-build's own build/ is cleaned via their
# respective Makefiles.
clean:
	rm -rf $(ROOT)/build
	$(MAKE) -C msp430 clean
	$(MAKE) -C msp430fr5962 clean
	$(MAKE) -C msp432 clean
	$(MAKE) -C pc clean
