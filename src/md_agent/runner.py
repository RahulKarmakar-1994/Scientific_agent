import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path


class MDRunner:
    def __init__(self, output_dir="outputs", input_dir="inputs", external_md_dir="../MD_simulation"):
        self.output_dir = Path(output_dir)
        self.input_dir = Path(input_dir)
        self.external_md_dir = Path(external_md_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def run(self, config, engine="local"):
        if engine == "local":
            return self.run_local_toy_md(config)
        if engine == "docker":
            return self.run_docker_md(config)
        if engine == "external-md":
            return self.run_external_md(config)
        raise ValueError(f"Unsupported MD engine: {engine}")

    def run_local_toy_md(self, config):
        """Run a tiny deterministic toy MD-like calculation.

        This is a placeholder for the Docker/custom-code path. It creates an
        energy and temperature trace so the agent workflow has real outputs.
        """
        steps = int(config["steps"])
        temperature_k = float(config["temperature_k"])
        random.seed(7)

        energy = []
        temperature = []
        base_energy = -10.0

        for step in range(steps):
            relaxation = math.exp(-step / max(steps / 5, 1))
            noise = random.uniform(-0.02, 0.02)
            energy.append(base_energy + 0.8 * relaxation + noise)
            temperature.append(temperature_k + random.uniform(-3.0, 3.0))

        result = {
            "config": config,
            "status": "completed",
            "engine": "local_toy_md",
            "energy": energy,
            "temperature": temperature,
            "summary": {
                "final_energy": energy[-1],
                "mean_temperature_k": sum(temperature) / len(temperature),
                "steps_completed": steps,
            },
        }

        output_path = self.output_dir / "latest_result.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def run_docker_md(self, config, image="md-agent-toy-md"):
        """Run the Dockerized MD tool.

        The image must be built first:
        docker build -t md-agent-toy-md tools/md_docker_tool
        """
        if shutil.which("docker") is None:
            return {
                "config": config,
                "status": "failed",
                "engine": "docker_toy_md",
                "error": "docker command not found. Install Docker or use --engine local.",
                "summary": {
                    "final_energy": None,
                    "mean_temperature_k": None,
                    "steps_completed": 0,
                },
            }

        config_path = self.input_dir / "config.json"
        output_path = self.output_dir / "docker_result.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self.input_dir.resolve()}:/inputs",
            "-v",
            f"{self.output_dir.resolve()}:/outputs",
            image,
            "python",
            "/app/run_md.py",
            "--config",
            "/inputs/config.json",
            "--output",
            "/outputs/docker_result.json",
        ]

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

        if completed.returncode != 0:
            return {
                "config": config,
                "status": "failed",
                "engine": "docker_toy_md",
                "error": completed.stderr.strip() or completed.stdout.strip(),
                "summary": {
                    "final_energy": None,
                    "mean_temperature_k": None,
                    "steps_completed": 0,
                },
            }

        return json.loads(output_path.read_text(encoding="utf-8"))

    def run_external_md(self, config, script_name="run_md_config.py", timeout_seconds=60):
        """Run the user's existing MD_simulation script as an external tool.

        The external wrapper accepts JSON config and writes JSON results.
        """
        md_dir = self.external_md_dir.resolve()
        script_path = md_dir / script_name

        if not script_path.exists():
            return self._external_failure(
                config,
                f"External MD script not found: {script_path}",
            )

        config_path = self.input_dir / "external_md_config.json"
        output_path = self.output_dir / "external_md_result.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    script_name,
                    "--config",
                    str(config_path.resolve()),
                    "--output",
                    str(output_path.resolve()),
                ],
                cwd=md_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return self._external_failure(
                config,
                f"External MD timed out after {timeout_seconds} seconds.",
                stdout=exc.stdout,
                stderr=exc.stderr,
            )

        if completed.returncode != 0:
            result = self._external_failure(
                config,
                completed.stderr.strip() or completed.stdout.strip(),
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        elif output_path.exists():
            result = json.loads(output_path.read_text(encoding="utf-8"))
            result["stdout_tail"] = _tail(completed.stdout)
            result["stderr_tail"] = _tail(completed.stderr)
        else:
            result = {
                "config": config,
                "status": "failed",
                "engine": "external_md",
                "error": "External MD completed but did not write output JSON.",
                "summary": {
                    "final_energy": None,
                    "mean_temperature_k": None,
                    "steps_completed": 0,
                },
                "stdout_tail": _tail(completed.stdout),
                "stderr_tail": _tail(completed.stderr),
            }

        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _external_failure(self, config, error, stdout=None, stderr=None):
        return {
            "config": config,
            "status": "failed",
            "engine": "external_md",
            "error": error,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "summary": {
                "final_energy": None,
                "mean_temperature_k": None,
                "steps_completed": 0,
            },
        }


def _tail(text, limit=2000):
    if not text:
        return ""
    return text[-limit:]
