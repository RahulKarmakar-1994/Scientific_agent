import argparse
import json
import math
import random
from pathlib import Path


def run_toy_md(config):
    steps = int(config["steps"])
    temperature_k = float(config["temperature_k"])
    random.seed(7)

    energy = []
    temperature = []

    for step in range(steps):
        relaxation = math.exp(-step / max(steps / 5, 1))
        energy.append(-10.0 + 0.8 * relaxation + random.uniform(-0.02, 0.02))
        temperature.append(temperature_k + random.uniform(-3.0, 3.0))

    return {
        "config": config,
        "status": "completed",
        "engine": "docker_toy_md",
        "summary": {
            "final_energy": energy[-1],
            "mean_temperature_k": sum(temperature) / len(temperature),
            "steps_completed": steps,
        },
        "energy": energy,
        "temperature": temperature,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_toy_md(config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
