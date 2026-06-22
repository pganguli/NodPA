#!/usr/bin/env python3
"""
Parse TeraTerm serial logs from real-device intermittent-inference experiments.

Reads timestamped log lines produced by the monitoring board (CC1352), accumulates
per-inference latency and recharging time, and prints average latency / active time
over the requested number of inferences.

Usage: python exp/parse-intermittent-inference-logs.py --log-filename FILE --num-inferences N
"""

import argparse
import contextlib
import datetime
import time
import statistics

# If the active time of an inference is very short (ex: < 100ms), it's very likely the inference
# completion signal is a false positive. Specifically, when an inference completion signal is sent
# over GPIO, the inference engine may not record the sent signal on NVM before power fails, and thus
# upon power resumption, the inference completion signal is sent again.
VALID_ACTIVE_TIME_THRESHOLD_MILLISECONDS = 100

# For fixing recharging time
# If power on/off is detected by whether VBAT_OK is high or low (i.e., higher or lower than 1.5V),
# the device remains on until VBAT_OK is roughly 1V. The gap between 1.5V and 1V is roughly 45
# milliseconds.
# If power on/off is detected by whether 3V3 is high or low, set this value to 0.
# EXTRA_POWER_ON_TIME = 45
EXTRA_POWER_ON_TIME = 0


def main():
    parser = argparse.ArgumentParser(
        description="Parse TeraTerm serial logs from real-device intermittent-inference experiments."
    )
    parser.add_argument(
        "--log-filename",
        required=True,
        help="Path to the raw TeraTerm log file (*.log or *-raw.log)",
    )
    parser.add_argument(
        "--num-inferences",
        type=int,
        required=True,
        help="Number of inferences to collect before reporting averages",
    )
    parser.add_argument(
        "--follow",
        action=argparse.BooleanOptionalAction,
        help="Keep reading the log file as it grows (useful for live monitoring)",
    )
    parser.add_argument(
        "--continuous-power",
        action=argparse.BooleanOptionalAction,
        help="Treat the run as continuous-power (skip inferences with power failures)",
    )
    args = parser.parse_args()

    initial_timestamp = None

    with (
        open(args.log_filename, "r") as log_file,
        open(args.log_filename.replace("-raw.log", ".log"), "w")
        if args.log_filename.endswith("-raw.log")
        else contextlib.nullcontext() as filtered_log_file,
    ):
        active_times = []
        inference_latencies = []
        accumulated_recharging_time = 0
        short_recharging_cycles = [0]
        power_failures = []
        first_inference_dropped = False

        current_line = ""

        while True:
            line = log_file.readline()
            # collect incomplete lines, as readline may not read a full line
            # when the last line is not fully written to the log file
            if "\n" not in line:
                current_line += line
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    break
                continue
            line = (current_line + line).strip()
            current_line = ""

            if "]" not in line:
                print(f"WARNING: found corrupted output line {repr(line)}")
                continue

            original_line = line

            timestamp, line = line.split("]")
            timestamp = datetime.datetime.strptime(
                timestamp.removeprefix("["), "%Y-%m-%d %H:%M:%S.%f"
            )

            if not initial_timestamp:
                initial_timestamp = timestamp
            # Ignore lines within the first second, as those may contain data before logging starts
            if (timestamp - initial_timestamp).total_seconds() <= 1.0:
                continue

            line = line.strip()
            if line.startswith("."):
                line = line[1:]
                if " " not in line:
                    print(f"WARNING: found corrupted output line {repr(line)}")
                    continue

                counter, power_failure = line.split(" ")
                cur_inference_latency = int(counter)
                # print(f'Cur inference latency: {cur_inference_latency}')
                power_failure = int(power_failure.removeprefix("PF="))

                cur_active_time = cur_inference_latency - accumulated_recharging_time

                found_new_inference = False
                if not inference_latencies:
                    found_new_inference = True
                else:
                    if cur_active_time < VALID_ACTIVE_TIME_THRESHOLD_MILLISECONDS:
                        found_new_inference = False
                    else:
                        found_new_inference = True

                if not found_new_inference:
                    if inference_latencies:
                        print(f"WARNING: Skip invalid latency {cur_inference_latency}")

                if found_new_inference:
                    inference_latencies.append(cur_inference_latency)
                    active_times.append(cur_active_time)
                    power_failures.append(power_failure)
                else:
                    inference_latencies[-1] += cur_inference_latency
                    active_times[-1] += cur_active_time
                    power_failures[-1] += power_failure

                accumulated_recharging_time = 0
                short_recharging_cycles.append(0)

                if not first_inference_dropped:
                    # Exclude the first inference, whose latency includes experiment setup time
                    inference_latencies = inference_latencies[1:]
                    active_times = active_times[1:]
                    power_failures = power_failures[1:]
                    first_inference_dropped = True
                    continue

                if args.continuous_power and power_failures[-1] != 0:
                    inference_latencies = inference_latencies[:-1]
                    active_times = active_times[:-1]
                    power_failures = power_failures[:-1]
                    continue

                if filtered_log_file:
                    filtered_line = (
                        original_line.replace(" .", " Latency=").replace(
                            " PF=", ", PF="
                        )
                        + f", AT={active_times[-1]}"
                    )
                    filtered_log_file.write(filtered_line + "\n")

                if len(inference_latencies) >= args.num_inferences:
                    average_latency = statistics.mean(
                        inference_latencies[-args.num_inferences :]
                    )
                    print(
                        f"Found {len(inference_latencies)} inferences. Average latency for the last {args.num_inferences} = {(average_latency / 1000):.4f} seconds"
                    )
                    if not args.follow:
                        break
                else:
                    print(
                        f"Found {len(inference_latencies)} inferences. Needs {args.num_inferences - len(inference_latencies)}"
                    )
            elif line.startswith("R"):
                try:
                    cur_recharge_time = int(line[1:])
                except ValueError:
                    print(f"Invalid line for recharging time: {repr(line)}")
                    continue
                # print(cur_recharge_time)

                if cur_recharge_time > EXTRA_POWER_ON_TIME:
                    cur_recharge_time -= EXTRA_POWER_ON_TIME
                else:
                    # cur_recharge_time = 0
                    short_recharging_cycles[-1] += 1

                accumulated_recharging_time += cur_recharge_time

        # print(f'Short recharging cycles: {short_recharging_cycles}')
        print("Inference latencies: " + " ".join(map(str, inference_latencies)))
        print("Active times: " + " ".join(map(str, active_times)))
        print("Power failures: " + " ".join(map(str, power_failures)))
        final_average_latency = (
            statistics.mean(inference_latencies[-args.num_inferences :]) / 1000
        )
        final_average_active_time = (
            statistics.mean(active_times[-args.num_inferences :]) / 1000
        )
        print(f"Average latency={final_average_latency:.4f} seconds")
        print(f"Average active time={final_average_active_time:.4f} seconds")
        print()


if __name__ == "__main__":
    main()
