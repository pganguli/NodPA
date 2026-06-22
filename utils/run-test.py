"""
Build and run a single NodPA configuration end-to-end as a CI test step.

Calls transform.py to generate data.cpp/data.h, then cmake+make to build the
PC simulator binary, then runs the simulator and checks the output.  For
intermittent tests, uses exp/run-intermittently.py to inject power failures.

Invoked by generate-buildbot-config.py output / buildbot workers.
Usage: python utils/run-test.py CONFIG_ARGS
"""

import os
import pathlib
from subprocess import check_call, CalledProcessError
import sys
import time

TOPDIR = pathlib.Path(__file__).absolute().parents[1]


def check_call_verbose(*popenargs, **kwargs):
    try:
        args = popenargs[0]
    except IndexError:
        args = kwargs["args"]

    print(f"Starting command {args} at timestamp {time.time()}")

    return check_call(*popenargs, **kwargs)


def build_and_test(config, suffix, intermittent):
    try:
        os.unlink("nvm.bin")
    except FileNotFoundError:
        pass

    my_debug = 1
    config = config.copy()
    # somehow a large samples.bin breaks intermittent
    # execution
    if intermittent:
        try:
            config.remove("--all-samples")
        except ValueError:
            pass
        my_debug = 3
    check_call_verbose(
        [sys.executable, TOPDIR / "dnn-models" / "transform.py", *config]
    )

    check_call_verbose(
        [
            "cmake",
            "-S",
            TOPDIR,
            "-B",
            "build",
            "-DBUILD_MSP432=OFF",
            f"-DMY_DEBUG={my_debug}",
        ]
    )
    check_call_verbose(["make", "-C", "build"])

    rounds = 100
    power_cycle = 0.03
    if "cifar10" in config or "cifar10-dnp" in config:
        rounds = 50
        power_cycle = 0.04

    run_cmd = ["./build/intermittent-cnn"]
    if intermittent:
        run_cmd = [
            sys.executable,
            TOPDIR / "exp" / "run-intermittently.py",
            "--rounds",
            str(rounds),
            "--interval",
            str(power_cycle),
            "--suffix",
            suffix,
            "--compress",
        ] + run_cmd
    check_call_verbose(run_cmd, env={"TMPDIR": "/var/tmp"})


def main():
    # preparation
    suffix = os.environ["LOG_SUFFIX"]
    config = os.environ["CONFIG"].split(" ")

    try:
        build_and_test(config, suffix, intermittent=False)

        # Test intermittent running
        if "--ideal" not in config:
            build_and_test(config, suffix, intermittent=True)
    except CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
