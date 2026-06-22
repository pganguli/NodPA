# Capture Non-determinism If You Can: Intermittent Inference on Dynamic Neural Networks

<!-- ABOUT THE PROJECT -->
## Overview

This project develops a middleware module (referred to as NodPA) that accumulates non-deterministic inference progress to enable correct and efficient dynamic neural network inference on intermittent systems. 
NodPA strategically selects additional progress information to capture the non-determinism of the power-interrupted computation while preserving only the changed portions of the progress information to maintain low runtime overhead.

We implemented NodPA on the Texas Instruments device MSP-EXP432P401R. It is an ARM-based 32-bit MCU with 64KB SRAM and single instruction multiple data (SIMD) instructions for accelerated computation. An external NVM module (Cypress CY15B116QN serial FRAM) was integrated to the platform. 

NodPA was integrated with the [HAWAII](https://ieeexplore.ieee.org/document/9211553) intermittent inference engine for evalution purposes, although it is compatible with other engines. 

NodPA contains two main components which interacts with the inference engine at runtime:

* Non-determinism accumulator: determines the sufficient set of progress indicators to track, ensuring non-deterministic progress information capture of the target dynamic network.
* Preservation minimizer: ensures low progress preservation overhead, by reducing both the amount of data and number of transfers when preserving progress to NVM. 

We evaluate NodPA on three networks, ResNet, HAR, and KWS, using both static and dynamic variants, trained on the CIFAR-10 dataset, an accelerometer sensor dataset, and the Google Speech Commands dataset, respectively.
We compare NodPA with two existing baselines, CD and FD, in terms of model accuracy, inference latency, and runtime overhead, which includes both preservation and recovery overhead.


Demo video: [https://youtu.be/_1qVoG4aCxY](https://youtu.be/_1qVoG4aCxY)

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Directory/File Structure](#directory/file-structure)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Python Environment](#python-environment)
  * [Generating Embedded Data Files](#generating-embedded-data-files)
  * [Building and Running the PC Simulator](#building-and-running-the-pc-simulator)
  * [Automated Pipeline](#automated-pipeline)
  * [Testing Intermittent Execution](#testing-intermittent-execution)
  * [Hardware Wiring](#hardware-wiring)
  * [Setup and Build for MSP430FR5994](#setup-and-build-for-msp430fr5994)
  * [Setup and Build for MSP432P401R](#setup-and-build-for-msp432p401r)

## Directory/File Structure

Below is an explanation of the directories/files found in this repo.

* `common/conv.cpp`, `common/fc.cpp`, `common/pooling.cpp`, `common/op_handlers.cpp`, `common/op_utils.*`: functions implementing various neural network layers and auxiliary functions shared among different layers.
* `common/cnn_common.*`, `common/intermittent-cnn.*`: main components of the HAWAII intermittent inference engine.
* `common/platform.*`, `common/plat-mcu.*` and `common/plat-pc.*`: high-level wrappers for handling platform-specific peripherals.
* `common/my_dsplib.*`: high-level wrappers for accessing different vendor-specific library calls performing accelerated computations.
* `common/counters.*` : helper functions for measuring runtime overhead.
* `dnn-models/`: pre-trained ONNX models and python scripts for converting an ONNX model into a custom format recognized by the lightweight inference engine.
* `exp_data/exp.xlsx`: experimental results.
* `msp432/`: platform-speicific hardware initialization functions.
* `tools/`: helper functions for various system peripherals (e.g., UART, system clocks and external FRAM)
* `tools/intermittent-power-supply/`: codes for controlling the power supply to simulate intermittent power traces.
* `train/`: codes for training neural networks with dynamic pruning and exporting trained models to ONNX.

## Getting Started

### Prerequisites

Here are basic software and hardware requirements to build NodPA along with the HAWAII intermittent inference engine:

**Software:**

* Python 3.11 or 3.12 (3.12 is recommended; the pinned packages in `requirements-base.txt` are not yet compatible with 3.13+)
* [Code composer studio](https://www.ti.com/tool/CCSTUDIO) 12.8
* [MSP432 driverlib](https://www.ti.com/tool/MSPDRIVERLIB) 3.21.00.05

**Hardware (for real-device deployment):**

* [MSP-EXP432P401R LaunchPad](https://www.ti.com/tool/MSP-EXP432P401R) (abbreviated MSP432) — the intermittent system under test
* LAUNCHXL-CC1352R1 LaunchPad (abbreviated CC1352) — monitoring board
* Cypress CY15B116QN serial FRAM — external NVM

### Python Environment

Create and activate a virtual environment, then install the inference and transform dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-base.txt
```

`requirements-dev.txt` adds matplotlib and Jupyter for exploratory work. For training new models from scratch (GPU required), use `train/requirements.txt` instead. The power-supply control scripts have their own `tools/intermittent-power-supply/requirements.txt`.

### Generating Embedded Data Files

`dnn-models/transform.py` must be run before building. It reads a pre-trained ONNX model, quantizes weights to Q15 fixed-point, and writes `build/data.h` and `build/data.cpp` which are compiled into the firmware.

```bash
python dnn-models/transform.py --target msp432 --hawaii --all-samples cifar10-dnp
```

Available configs: `cifar10-dnp`, `har-dnp`, `kws-dnp`. Add `--model-variant static` for static (non-dynamic) variants. Outputs go to `build/` by default (override with `--data-output-dir`).

### Building and Running the PC Simulator

```bash
cmake -B build -S .
make -C build intermittent-cnn   # build only the PC simulator target
./build/intermittent-cnn          # run all samples (full dataset, continuous power)
./build/intermittent-cnn 1        # run 1 sample
./build/intermittent-cnn 100      # run 100 samples
```

> **Note:** If CCS is installed, cmake will also attempt to build the `plat-mcu` library for
> cross-compilation. This will fail if MSP432 driverlib is not yet installed, but does not affect
> the PC simulator. Specifying `intermittent-cnn` as the make target avoids this error.

The simulator writes (and reads back) NVM state to `nvm.bin` in the current directory by default.
Use `-n PATH` to redirect it (recommended when running inside Docker or WSL to avoid slow cross-filesystem I/O):

```bash
./build/intermittent-cnn -n /tmp/nvm.bin      # use /tmp for fast local storage
./build/intermittent-cnn -r -n /tmp/nvm.bin   # read-only: discard state changes (useful for debugging)
```

Debug output levels (set at cmake configure time):

* `cmake -B build -S . -D MY_DEBUG=0` — no debug output (default for size-optimized builds)
* `cmake -B build -S . -D MY_DEBUG=1` — basic progress output
* `cmake -B build -S . -D MY_DEBUG=2` — verbose layer outputs (needed for protobuf feature-map saving)
* `cmake -B build -S . -D MY_DEBUG=3` — maximum verbosity (many power-cycle points for intermittency testing)

For more details about the simulator, see `common/simulator.md`.

### Automated Pipeline

`run.py` orchestrates the full pipeline: transform → cmake configure → make → run.

```bash
# Run all model/power combinations in the simulator
python run.py --platform simulator

# Run a specific model and power level
python run.py --platform simulator --model dynamic-cifar10 --power continuous
python run.py --platform simulator --model static-har --power 4mW
```

Valid model names: `dynamic-{cifar10,har,kws}`, `static-{cifar10,har,kws}` (append `-nopres` to disable preservation).
Valid power levels: `4mW`, `12mW`, `continuous`.

### Testing Intermittent Execution

`exp/run-intermittently.py` runs the simulator in a loop, injecting simulated power failures (via the `-c` flag) and restarting until one complete inference finishes.

```bash
# Run 3 complete inferences, shutting down after a random number of NVM writes in [3100, 4200]
# (these bounds correspond to the cifar10 model at ~4 mW harvested power)
rm -f /tmp/nvm.bin
python exp/run-intermittently.py \
    --rounds 3 \
    --n-samples 1 \
    --shutdown-after-writes 3100,4200 \
    --nvm-bin-path /tmp/nvm.bin \
    ./build/intermittent-cnn
```

Each round prints the number of power cycles needed and running accuracy. Successful output looks like:

```text
correct=1 total=1 accuracy=100.00%
Number of power cycles: 214
```

For power-cycle bounds for other models and power levels, see `SimulatorRunner.POWER_CYCLE_BOUNDS` in `run.py`. For more options (time-based failures, log compression), see `common/simulator.md`.

### Hardware Wiring

#### EHM (Energy Harvesting Module)

Connect BAT and VBAT_OK of CJMCU-25504 to 5LP01SP PMOS, 5LN01SP NMOS and a large resistor (e.g., 10 MΩ) as shown in `EHM_circuit.png`.

#### MSP432 ↔ CC1352 (monitoring board)

| CC1352 pin | MSP432 pin | Purpose                |
| ---------- | ---------- | ---------------------- |
| GND        | GND        | common ground          |
| DIO16      | P5.5       | count inferences       |
| DIO11      | 3V3        | detect power on/off    |

#### MSP432 ↔ EHM

| EHM pin | MSP432 pin | Purpose                  |
| ------- | ---------- | ------------------------ |
| GND     | GND        | common ground            |
| VOUT    | 3V3        | supply energy to MSP432  |

#### MSP432 ↔ CY15B116QN (external FRAM)

| CY15B116QN pin | MSP432 pin | Purpose                    |
| -------------- | ---------- | -------------------------- |
| GND            | GND        | common ground              |
| 3V3            | 3V3        | supply energy to FRAM      |
| CS             | P1.5       | chip select                |
| SDI            | P2.3       | MOSI (MSP432 → FRAM)       |
| SDO            | P2.2       | MISO (FRAM → MSP432)       |
| SCK            | P2.1       | SPI clock                  |

### Setup and Build for MSP430FR5994

The `msp430/` folder contains a CCS project for the MSP430FR5994 LaunchPad.

#### MSP430 Prerequisites

* **CCS 12.8** with the MSP430 compiler toolchain (v21.6 LTS) — bundled with CCS, no separate download
* **MSP430 driverlib**: download [MSP430Ware](https://www.ti.com/tool/MSPDRIVERLIB) from TI, then copy
  `driverlib/MSP430FR5xx_6xx/` into `msp430/driverlib/MSP430FR5xx_6xx/` inside this repo.
  If you already installed MSP430Ware via CCS Resource Explorer, the folder is typically at
  `~/ti/msp430/MSP430ware_*/driverlib/MSP430FR5xx_6xx/`.
* **Submodules** (TI-DSPLib): `git submodule update --init --recursive`

#### Step 1: Generate embedded data files

```bash
source .venv/bin/activate
python dnn-models/transform.py --target msp430 --hawaii --all-samples cifar10-dnp
```

Use `har-dnp` or `kws-dnp` for the other models. The generated `build/data.cpp` is linked
directly into the CCS project via a path-linked resource.

#### Step 2: Import the CCS project

1. **File → Import → Code Composer Studio → CCS Projects**
2. Browse to the `msp430/` directory — CCS will find the project named `intermittent-cnn-msp430`
3. Import **in place** (do not copy into workspace)

#### Step 3: Select build configuration and build

The project has two build configurations:

| Configuration  | Device        | Compiler   | Use when                      |
| -------------- | ------------- | ---------- | ----------------------------- |
| **Large_Data** | MSP430FR5994  | v21.6 LTS  | Normal use — use this one     |
| Small_Data     | MSP430FR5969  | v4.4       | Legacy; smaller FRAM device   |

Select **Large_Data**, then **Project → Build Project** (Ctrl+B).

Key compiler settings applied automatically by the `.cproject`:

* `-D __MSP430FR5994__` and large data model (20-bit addresses)
* `-O3 --opt_for_speed=5` (maximum speed optimization)
* `--use_hw_mpy=F5` (LEA hardware multiplier)
* Silicon errata workarounds: CPU21, CPU22, CPU40

#### Step 4: Flash and run

Connect the MSP-EXP430FR5994 LaunchPad via USB, then **Run → Debug** in CCS. This flashes the
`.out` file and halts at `main`. Press **Run → Resume** (F8) to start inference.

Inference results are printed over UART at **9600 baud** (configured in `tools/myuart.h`).
Use the CCS built-in serial terminal or run:

```bash
python tools/minicom-launcher.py
```

### Setup and Build for MSP432P401R

The `msp432/` folder contains a CCS project for the MSP-EXP432P401R LaunchPad.

#### MSP432 Prerequisites

* **CCS 12.8**
* **MSP432 driverlib** ([MSPDRIVERLIB](https://www.ti.com/tool/MSPDRIVERLIB) v3.21.00.05):
  copy `driverlib/MSP432P4xx/` into `msp432/driverlib/MSP432P4xx/`
* **Submodules**: `git submodule update --init --recursive`

#### MSP432 Steps

1. Generate data files: `python dnn-models/transform.py --target msp432 --hawaii --all-samples cifar10-dnp`
2. **File → Import → Code Composer Studio → CCS Projects**, browse to `msp432/`, import in place
3. Build the project (**Project → Build Project**)
4. Flash via **Run → Debug**, then resume
