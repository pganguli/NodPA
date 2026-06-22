#!/usr/bin/env python3
"""
End-to-end pipeline runner for NodPA experiments.

Converts a pre-trained ONNX model, builds the C simulator or real-device firmware,
and runs inference — either on the PC simulator or on the physical MSP432 board.

Usage: python run.py [--platform simulator|real_device] [--model MODEL] [--power POWER]
"""

from abc import ABC, abstractmethod
import argparse
import datetime
import itertools
import os
import pathlib
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time


class Runner(ABC):
    ALL_MODELS = ("resnet", "har", "kws")
    ALL_VARIANTS = ("static", "dynamic")
    ALL_MODEL_COMBINATIONS = list(
        itertools.product(["", "-nopres"], ALL_VARIANTS, ALL_MODELS)
    )
    ALL_MODEL_COMBINATIONS_STR = [
        f"{variant}-{model}{suffix}"
        for suffix, variant, model in ALL_MODEL_COMBINATIONS
    ]
    ALL_POWERS = ("4mW", "12mW", "continuous")

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
    CCS_ROOT = pathlib.Path(r"C:\ti\ccs1281\ccs")
    CCS_WORKSPACE = (
        r"C:\Users\yen\Documents\Local\School\newslab\ccstudio-workspace-v12-win"
    )

    def check_call(self, args, *popenargs, **kwargs) -> None:
        print("Running: " + shlex.join(args))
        subprocess.check_call(args, *popenargs, **kwargs)

    def convert_model(self, model: str, variant: str, with_preservation: bool) -> None:
        cmd = [
            sys.executable,
            "dnn-models/transform.py",
            "--target",
            "msp432",
            "--all-samples",
            "--hawaii",
            f"{model}-dnp",
        ]

        if variant == "static":
            cmd.extend(["--model-variant", "static"])

        if with_preservation:
            cmd.extend(["--dynamic-dnn-approach", "multiple-indicators"])
        else:
            cmd.extend(["--dynamic-dnn-approach", "coarse-grained"])

        self.check_call(cmd)

    @abstractmethod
    def configure_project(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_project(self, model: str, variant: str, with_preservation: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def deploy(
        self, model: str, variant: str, with_preservation: bool, power: str
    ) -> None:
        raise NotImplementedError

    def run(self, model: str, power: str) -> None:
        for model in model.split(","):
            with_preservation = True
            variant, model = model.split("-", maxsplit=1)
            if model.endswith("-nopres"):
                with_preservation = False
                model = model.removesuffix("-nopres")
            if model == "resnet":
                model = "cifar10"

            self.convert_model(model, variant, with_preservation)
            self.configure_project()
            self.build_project(model, variant, with_preservation)

            for power in power.split(","):
                self.deploy(model, variant, with_preservation, power)


class SimulatorRunner(Runner):
    POWER_CYCLE_BOUNDS = {
        "har-4mW": [3000, 4100],
        "har-12mW": [8300, 10000],
        "cifar10-4mW": [3100, 4200],
        "cifar10-12mW": [7350, 9000],
        "kws-4mW": [600, 1500],
        "kws-12mW": [6300, 7500],
    }
    N_SAMPLES = {
        "har": 2947,
        "cifar10": 10000,
        "kws": 4890,
    }

    def configure_project(self) -> None:
        self.check_call(["cmake", "-B", "build", "-S", ".", "-D", "MY_DEBUG=1"])

    def build_project(self, model: str, variant: str, with_preservation: bool) -> None:
        self.check_call(["make", "-C", "build"])

    def deploy(
        self, model: str, variant: str, with_preservation: bool, power: str
    ) -> None:
        # Use a path on a temp dir, to avoid slow file access on Docker/WSL
        nvm_bin_path = "/tmp/nvm.bin"

        try:
            os.remove(nvm_bin_path)
        except FileNotFoundError:
            pass

        start_time = time.time()

        num_samples = self.N_SAMPLES[model]

        if power != "continuous":
            power_cycle_lower_bound, power_cycle_upper_bound = self.POWER_CYCLE_BOUNDS[
                f"{model}-{power}"
            ]

            self.check_call(
                [
                    sys.executable,
                    "exp/run-intermittently.py",
                    "--rounds",
                    "1",
                    "--n-samples",
                    str(num_samples),
                    "--shutdown-after-writes",
                    f"{power_cycle_lower_bound},{power_cycle_upper_bound}",
                    "--nvm-bin-path",
                    nvm_bin_path,
                    "./build/intermittent-cnn",
                ]
            )

        else:
            self.check_call(
                [
                    "./build/intermittent-cnn",
                    "-r",
                    str(num_samples),
                    "-n",
                    nvm_bin_path,
                ]
            )

        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_per_inference_in_milliseconds = elapsed_time / num_samples * 1000
        print(
            f"Time taken: {elapsed_time:.3f} seconds ({elapsed_time_per_inference_in_milliseconds:.3f} milliseconds per inference)"
        )


class RealDeviceRunner(Runner):
    TERATERM_PATH = r"C:\Program Files (x86)\teraterm\ttermpro.exe"
    MONITORING_BOARD_BAUD_RATE = 115200
    MONITORING_BOARD_DESCRIPTION = "XDS110 Class Application/User UART"
    MONITORING_BOARD_SERIAL_NUMBER = "L4100AFL"

    NUM_INFERENCES = {
        "continuous": 10,
        "4mW": 10,
        "12mW": 25,
    }

    def __init__(self, regenerate_binaries: bool = False):
        self.regenerate_binaries = regenerate_binaries

    def get_prebuilt_real_device_binary_filename(
        self, model, variant, with_preservation
    ) -> str:
        suffix = "" if with_preservation else "-nopres"
        return str(
            self.PROJECT_ROOT
            / "msp432-binaries"
            / f"intermittent-cnn-msp432-{variant}-{model}{suffix}.out"
        )

    def convert_model(self, model: str, variant: str, with_preservation: bool) -> None:
        if (
            os.path.exists(
                self.get_prebuilt_real_device_binary_filename(
                    model, variant, with_preservation
                )
            )
            and not self.regenerate_binaries
        ):
            return
        super().convert_model(model, variant, with_preservation)

    def configure_project(self) -> None:
        pass

    def build_project(self, model: str, variant: str, with_preservation: bool) -> None:
        os.makedirs(self.PROJECT_ROOT / "msp432-binaries", exist_ok=True)

        # Build the project if needed
        output_file = self.get_prebuilt_real_device_binary_filename(
            model, variant, with_preservation
        )

        if not os.path.exists(output_file) or self.regenerate_binaries:
            self.check_call(
                [
                    str(self.CCS_ROOT / "eclipse" / "eclipsec.exe"),
                    "-noSplash",
                    "-data",
                    self.CCS_WORKSPACE,
                    "-application",
                    "com.ti.ccstudio.apps.projectBuild",
                    "-ccs.projects",
                    "intermittent-cnn-msp432",
                ]
            )
            shutil.copy2(
                self.PROJECT_ROOT
                / "msp432"
                / "Debug__GNU"
                / "intermittent-cnn-msp432.out",
                output_file,
            )

    def deploy(
        self, model: str, variant: str, with_preservation: bool, power: str
    ) -> None:
        from serial.tools.list_ports import comports

        if self.regenerate_binaries:
            return

        input("Set jumpers... Hit ENTER to continue")

        env = os.environ.copy()
        env["EXIT_LOADTI_WITH_ERRORLEVEL"] = "1"
        # Deploy the program
        self.check_call(
            [
                str(
                    self.CCS_ROOT
                    / "ccs_base"
                    / "scripting"
                    / "examples"
                    / "loadti"
                    / "loadti.bat"
                ),
                "--cfg-file",
                str(
                    self.PROJECT_ROOT / "msp432" / "targetConfigs" / "MSP432P401R.ccxml"
                ),
                "--load",
                "--reset",
                self.get_prebuilt_real_device_binary_filename(
                    model, variant, with_preservation
                ),
            ],
            env=env,
        )

        print("Programming completed")

        input("Set jumpers... Hit ENTER to continue")

        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_log_dir = self.PROJECT_ROOT / "exp_data" / "logs"
        exp_log_dir.mkdir(exist_ok=True)
        teraterm_log_path = (
            exp_log_dir / f"teraterm-{variant}-{model}-{now_str}-raw.log"
        )
        teraterm_ini = self.PROJECT_ROOT / "exp" / "TERATERM.ini"
        power_trace_log_path = exp_log_dir / f"power-trace-{power}-{now_str}.csv"

        serial_port = None
        for port in comports():
            if (
                port.serial_number == self.MONITORING_BOARD_SERIAL_NUMBER
                and self.MONITORING_BOARD_DESCRIPTION in port.description
            ):
                serial_port = port.device
                break
        assert serial_port and re.match("COM[0-9]+", serial_port), (
            "Cannot find a serial port with serial number "
            f'"{self.MONITORING_BOARD_SERIAL_NUMBER}" and description "{self.MONITORING_BOARD_DESCRIPTION}"!'
        )
        serial_port_index = int(serial_port[len("COM") :])

        # start Tera Term logging. See https://teratermproject.github.io/manual/5/en/commandline/teraterm.html for command line arguments
        teraterm_cmdline = [
            self.TERATERM_PATH,
            f"/BAUD={self.MONITORING_BOARD_BAUD_RATE}",
            f"/C={serial_port_index}",
            f"/L={teraterm_log_path}",
            f"/F={teraterm_ini}",
        ]
        intermittent_power_supply_root = (
            self.PROJECT_ROOT / "tools" / "intermittent-power-supply"
        )
        power_supply_cmdline = [
            sys.executable,
            str(intermittent_power_supply_root / "control-power-supply.py"),
            "--power-trace-log-csv",
            str(power_trace_log_path),
        ]
        if power != "continuous":
            power_supply_cmdline.extend(
                [
                    "--script",
                    str(intermittent_power_supply_root / "script-rf.csv"),
                    "--normalized_average_current",
                    str(int(power[: -len("mW")]) / 1000),
                ]
            )
        else:
            power_supply_cmdline.extend(
                [
                    "--script",
                    str(intermittent_power_supply_root / "script-solar.csv"),
                    "--voltage",
                    "3",
                    "--normalized_average_current",
                    "0.1",
                ]
            )

        print("Running: " + shlex.join(teraterm_cmdline))
        teraterm_proc = subprocess.Popen(teraterm_cmdline)

        print("Running: " + shlex.join(power_supply_cmdline))
        power_supply_proc = subprocess.Popen(power_supply_cmdline)

        try:
            time.sleep(3)

            parse_log_cmdline = [
                sys.executable,
                str(self.PROJECT_ROOT / "exp" / "parse-intermittent-inference-logs.py"),
                "--log-filename",
                str(teraterm_log_path),
                "--num-inferences",
                str(self.NUM_INFERENCES[power]),
            ]
            if power == "continuous":
                parse_log_cmdline.append("--continuous-power")
            self.check_call(parse_log_cmdline)
        finally:
            host = "127.0.0.1"
            port = 65432
            message = "STOP"

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
                s.sendall(message.encode())
            power_supply_proc.wait()
            teraterm_proc.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="all",
        help="Model(s) to run. Format: {static|dynamic}-{cifar10|har|kws}[-nopres]. Use 'all' to run every combination.",
        choices=["all"] + Runner.ALL_MODEL_COMBINATIONS_STR,
    )
    parser.add_argument(
        "--power",
        default="all",
        help="Power-supply profile. '4mW' and '12mW' simulate intermittent energy harvesting; 'continuous' runs without power failures. Use 'all' for every level.",
        choices=Runner.ALL_POWERS,
    )
    parser.add_argument(
        "--platform",
        default="simulator",
        help="Target platform. 'simulator' uses the PC binary; 'real_device' deploys to the MSP432 board (Windows + TeraTerm required).",
        choices=["simulator", "real_device"],
    )
    parser.add_argument(
        "--regenerate-binaries", help=argparse.SUPPRESS, action="store_true"
    )
    args = parser.parse_args()

    if args.model == "all":
        args.model = ",".join(Runner.ALL_MODEL_COMBINATIONS_STR)
    if args.power == "all":
        args.power = ",".join(Runner.ALL_POWERS)

    if args.platform == "simulator":
        runner = SimulatorRunner()
    elif args.platform == "real_device":
        runner = RealDeviceRunner(args.regenerate_binaries)
    runner.run(args.model, args.power)


if __name__ == "__main__":
    main()
