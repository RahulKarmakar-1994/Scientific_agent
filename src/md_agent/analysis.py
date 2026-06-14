def analyze_md_result(result):
    if result.get("status") != "completed":
        return {
            "stability": "failed",
            "steps_completed": 0,
            "final_energy": None,
            "mean_temperature_k": None,
            "temperature_error_k": None,
            "notes": [
                result.get("error", "MD run did not complete."),
                "Use --engine local for a quick test, or refactor the selected engine to produce structured outputs.",
            ],
        }

    config = result["config"]
    summary = result["summary"]
    if result.get("engine") == "external_md_config":
        return {
            "stability": "completed",
            "steps_completed": summary.get("steps_completed", 0),
            "final_energy": summary.get("final_energy"),
            "mean_temperature_k": summary.get("mean_temperature_k"),
            "temperature_error_k": None,
            "notes": [
                "External MD completed using the config-driven wrapper.",
                "Temperature is reported in the units produced by the external MD code; no Kelvin conversion is applied yet.",
            ],
        }

    if summary.get("mean_temperature_k") is None:
        return {
            "stability": "completed-unparsed",
            "steps_completed": summary.get("steps_completed", 0),
            "final_energy": summary.get("final_energy"),
            "mean_temperature_k": summary.get("mean_temperature_k"),
            "temperature_error_k": None,
            "notes": result.get(
                "notes",
                ["MD completed, but structured outputs are not available yet."],
            ),
        }

    target_temperature = float(config["temperature_k"])
    mean_temperature = float(summary["mean_temperature_k"])
    temperature_error = abs(mean_temperature - target_temperature)

    stability = "stable" if temperature_error < 5.0 else "needs review"

    return {
        "stability": stability,
        "steps_completed": summary["steps_completed"],
        "final_energy": summary["final_energy"],
        "mean_temperature_k": mean_temperature,
        "temperature_error_k": temperature_error,
        "notes": [
            "This is a toy calculation, not a physical MD simulation yet.",
            "Replace the runner with your Dockerized MD code when ready.",
        ],
    }
