import re


def parse_md_request(query):
    """Parse a small MD request into structured parameters."""
    text = query.lower()

    return {
        "raw_query": query,
        "system": _find_system(text),
        "input_file": _find_input_file(query),
        "ensemble": _find_ensemble(text),
        "temperature_k": _find_number(text, r"(\d+(?:\.\d+)?)\s*k", 300.0),
        "pressure": _find_number(text, r"(\d+(?:\.\d+)?)\s*(?:atm|bar)", None),
        "steps": int(_find_number(text, r"(\d+)\s*(?:steps|step)", 1000)),
        "dt": _find_number(text, r"\bdt\s*[=:]?\s*(\d+(?:\.\d+)?)", 0.005),
        "r_cut": _find_number(text, r"(?:r_cut|cutoff)\s*[=:]?\s*(\d+(?:\.\d+)?)", 2.5),
        "timestep_fs": _find_number(text, r"(\d+(?:\.\d+)?)\s*fs", 1.0),
        "thermostat": _find_option(text, ["langevin", "nose-hoover", "berendsen"], "langevin"),
        "force_field": "custom",
    }


def _find_number(text, pattern, default):
    match = re.search(pattern, text)
    if not match:
        return default
    return float(match.group(1))


def _find_ensemble(text):
    for ensemble in ("nve", "nvt", "npt"):
        if ensemble in text:
            return ensemble.upper()
    if "pressure" in text:
        return "NPT"
    return "NVT"


def _find_option(text, options, default):
    for option in options:
        if option in text:
            return option
    return default


def _find_system(text):
    for system in ["water", "argon", "polymer", "methane", "ethanol"]:
        if system in text:
            return system
    return "unknown"


def _find_input_file(query):
    match = re.search(r"([\w./-]+\.data)", query)
    if match:
        return match.group(1)
    return "Particle-256/system.data"
