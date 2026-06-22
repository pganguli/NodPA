"""
Generate Buildbot CI configuration files for NodPA cross-product test matrix.

Produces JSON configs covering all combinations of model × target × approach ×
batch_size × dynamic_dnn_approach.  The output is consumed by the CI system to
schedule one build-and-test job per combination.

Usage: python utils/generate-buildbot-config.py > buildbot_configs.json
"""

import dataclasses
import itertools
import json
import pathlib
from typing import Any


@dataclasses.dataclass
class ModelConfig:
    model: str
    targets: list[str]
    approaches: list[str]
    batch_sizes: list[int]
    dynamic_dnn_approaches: list[str] | None = None


def config_builder(
    model: str,
    batch_size: int,
    approach: str,
    target: str,
    dynamic_dnn_approach: str | None = None,
) -> dict[str, Any]:
    config = f"--target {target} --{approach} --batch-size {batch_size} {model}"
    if approach == "ideal":
        config += " --all-samples"

    if dynamic_dnn_approach:
        config += f" --dynamic-dnn-approach {dynamic_dnn_approach}"

    suffix = f"{model}_{approach}"

    if dynamic_dnn_approach:
        suffix += f"_{dynamic_dnn_approach}"

    suffix += f"_b{batch_size}"

    if target == "msp432":
        suffix += "_cmsis"

    return {
        "builder_name": f"intermittent-cnn-{suffix}",
        "command_env": {
            "CONFIG": config,
            "LOG_SUFFIX": suffix.replace(".", "_"),
        },
    }


def main() -> None:
    all_targets = ["msp430", "msp432"]
    all_approaches = ["hawaii"]
    ideal_config = {"targets": all_targets, "approaches": ["ideal"], "batch_sizes": [1]}
    complete_config = {
        "targets": all_targets,
        "approaches": all_approaches,
        "batch_sizes": [1],
    }
    dynamic_dnn_approaches = [
        "coarse-grained",
        "fine-grained",
        "multiple-indicators-basic",
        "multiple-indicators",
    ]
    model_configs = [
        ModelConfig(model="cifar10-dnp", **ideal_config),
        ModelConfig(
            model="cifar10-dnp",
            targets=all_targets,
            approaches=["hawaii"],
            batch_sizes=[1],
            dynamic_dnn_approaches=dynamic_dnn_approaches,
        ),
        ModelConfig(model="har-dnp", **ideal_config),
        ModelConfig(
            model="har-dnp",
            targets=all_targets,
            approaches=["hawaii"],
            batch_sizes=[1],
            dynamic_dnn_approaches=dynamic_dnn_approaches,
        ),
    ]

    buildbot_configurations = []
    for m in model_configs:
        if m.dynamic_dnn_approaches:
            for target, approach, batch_size, dynamic_dnn_approach in itertools.product(
                m.targets, m.approaches, m.batch_sizes, m.dynamic_dnn_approaches
            ):
                buildbot_configurations.append(
                    config_builder(
                        m.model, batch_size, approach, target, dynamic_dnn_approach
                    )
                )
        else:
            for target, approach, batch_size in itertools.product(
                m.targets, m.approaches, m.batch_sizes
            ):
                buildbot_configurations.append(
                    config_builder(m.model, batch_size, approach, target)
                )

    output_path = pathlib.Path(__file__).resolve().parent / "buildbot-config.json"
    with open(output_path, "w") as f:
        json.dump(buildbot_configurations, f, indent=4)


if __name__ == "__main__":
    main()
