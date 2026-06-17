import json
import math


SUPPORTED_DEMO_TYPES = {
    "distribution",
    "multi_series_time_evolution",
    "phase_space",
    "random_walk",
    "relation_plot",
    "time_evolution",
}
SUPPORTED_RELATION_FAMILIES = {
    "inverse_square",
    "linear",
    "quadratic",
    "sinusoidal",
    "threshold_linear",
}


def build_demo_code(spec):
    """Build trusted Python code from a reusable simulation spec.

    The spec supplies scientific meaning. This module supplies only generic
    plotting/simulation primitives, so adding a new physics topic should not
    require a new branch here.
    """

    normalized = normalize_simulation_spec(spec)
    if normalized["status"] != "ready":
        return {
            "status": "unavailable",
            "reason": normalized.get("reason") or "Simulation spec is not ready.",
            "spec": normalized,
            "code": None,
        }

    demo_type = normalized.get("demo_type")
    if demo_type == "relation_plot":
        code = _relation_plot_code(normalized)
    elif demo_type == "random_walk":
        code = _random_walk_code(normalized)
    elif demo_type == "distribution":
        code = _distribution_code(normalized)
    elif demo_type == "time_evolution":
        code = _time_evolution_code(normalized)
    elif demo_type == "multi_series_time_evolution":
        code = _multi_series_time_evolution_code(normalized)
    elif demo_type == "phase_space":
        code = _phase_space_code(normalized)
    else:
        code = None

    if not code:
        return {
            "status": "unavailable",
            "reason": f"No trusted primitive is available for demo_type={demo_type!r}.",
            "spec": normalized,
            "code": None,
        }

    return {
        "status": "ready",
        "reason": "Built code from a reusable trusted demo primitive.",
        "spec": normalized,
        "code": code,
    }


def normalize_simulation_spec(spec):
    spec = dict(spec or {})
    status = str(spec.get("status") or "unavailable").lower()
    demo_type = str(spec.get("demo_type") or "").lower()
    relation_family = str(spec.get("relation_family") or "").lower()
    raw_parameters = spec.get("parameters") or {}
    parameters = _numeric_parameters(raw_parameters)

    if status != "ready":
        return {
            **spec,
            "status": "unavailable",
            "reason": spec.get("reason") or "Spec agent did not return a ready demo.",
        }
    if demo_type not in SUPPORTED_DEMO_TYPES:
        inferred_demo_type = _infer_demo_type(spec)
        if inferred_demo_type:
            demo_type = inferred_demo_type

    if demo_type not in SUPPORTED_DEMO_TYPES:
        return {
            **spec,
            "status": "unavailable",
            "reason": f"Unsupported demo_type: {demo_type}",
        }
    if demo_type == "relation_plot" and relation_family not in SUPPORTED_RELATION_FAMILIES:
        return {
            **spec,
            "status": "unavailable",
            "reason": f"Unsupported relation_family: {relation_family}",
        }
    consistency_issue = _relation_consistency_issue(spec, relation_family, raw_parameters)
    if consistency_issue:
        return {
            **spec,
            "status": "unavailable",
            "reason": consistency_issue,
        }

    x_range = _range(spec.get("x_range"), default=[0.0, 10.0])
    if demo_type == "relation_plot" and relation_family == "inverse_square":
        x_range[0] = max(abs(x_range[0]), 0.1)
        x_range[1] = max(abs(x_range[1]), x_range[0] + 1.0)

    return {
        "status": "ready",
        "concept": _short_text(spec.get("concept"), "scientific concept"),
        "demo_type": demo_type,
        "relation_family": relation_family or None,
        "parameters": parameters,
        "x_range": x_range,
        "x_label": _short_text(spec.get("x_label"), "x"),
        "y_label": _short_text(spec.get("y_label"), "y"),
        "title": _short_text(spec.get("title"), _short_text(spec.get("concept"), "Scientific demo")),
        "equation_text": _short_text(spec.get("equation_text"), ""),
        "expected_behavior": _short_text(spec.get("expected_behavior"), ""),
        "reason": _short_text(spec.get("reason"), "Spec produced by model."),
    }


def _relation_plot_code(spec):
    family = spec["relation_family"]
    params = spec["parameters"]
    x_min, x_max = spec["x_range"]
    title = _py_string(spec["title"])
    x_label = _py_string(spec["x_label"])
    y_label = _py_string(spec["y_label"])
    equation_text = _py_string(spec["equation_text"])
    expected_behavior = _py_string(spec["expected_behavior"])

    coefficient = _number(params.get("coefficient"), 1.0)
    slope = _number(params.get("slope"), coefficient)
    intercept = _number(params.get("intercept"), 0.0)
    threshold = _number(params.get("threshold"), 0.0)
    amplitude = _number(params.get("amplitude"), 1.0)
    frequency = _number(params.get("frequency"), 1.0)
    phase = _number(params.get("phase"), 0.0)

    return f"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace({x_min!r}, {x_max!r}, 300)
family = {family!r}
if family == "linear":
    y = {slope!r} * x + {intercept!r}
elif family == "quadratic":
    y = {coefficient!r} * x**2 + {slope!r} * x + {intercept!r}
elif family == "inverse_square":
    y = {coefficient!r} / np.maximum(x**2, 1e-12) + {intercept!r}
elif family == "threshold_linear":
    y = np.maximum(0.0, {slope!r} * (x - {threshold!r}) + {intercept!r})
elif family == "sinusoidal":
    y = {amplitude!r} * np.sin({frequency!r} * x + {phase!r}) + {intercept!r}
else:
    raise ValueError("Unsupported relation family")

print("Simulation-spec demo")
print(f"demo_type = relation_plot")
print(f"relation_family = {{family}}")
if {equation_text}:
    print("equation = " + {equation_text})
if {expected_behavior}:
    print("expected_behavior = " + {expected_behavior})
print(f"x range = {{float(x[0]):.3g}} to {{float(x[-1]):.3g}}")
print(f"y range = {{float(np.min(y)):.3g}} to {{float(np.max(y)):.3g}}")

plt.figure(figsize=(6, 4))
plt.plot(x, y)
plt.xlabel({x_label})
plt.ylabel({y_label})
plt.title({title})
plt.tight_layout()
plt.savefig("demo.png", dpi=150)
""".strip()


def _relation_consistency_issue(spec, relation_family, raw_parameters):
    if relation_family != "threshold_linear":
        return None

    if "threshold" not in (raw_parameters or {}):
        return "threshold_linear relation requires an explicit threshold parameter."

    equation_text = str(spec.get("equation_text") or "").lower()
    x_label = str(spec.get("x_label") or "").lower()
    if ("frequency" in equation_text or "hf" in equation_text or "h f" in equation_text) and "wavelength" in x_label:
        return "x-axis label says wavelength, but the equation/spec describes a frequency relation."

    return None


def _infer_demo_type(spec):
    text = " ".join(
        str(value)
        for value in [
            spec.get("demo_type"),
            spec.get("concept"),
            spec.get("equation_text"),
            spec.get("expected_behavior"),
            spec.get("reason"),
            spec.get("title"),
            spec.get("x_label"),
            spec.get("y_label"),
        ]
        if value is not None
    ).lower()
    if _describes_conserved_exchange(text):
        return "multi_series_time_evolution"
    if "random walk" in text or "mean squared displacement" in text:
        return "random_walk"
    if "phase space" in text or ("position" in text and "velocity" in text):
        return "phase_space"
    if "distribution" in text or "histogram" in text or "probability density" in text:
        return "distribution"
    if "over time" in text or "time evolution" in text:
        return "time_evolution"
    if "plot" in text or "relation" in text or "versus" in text:
        return "relation_plot"
    return None


def _describes_conserved_exchange(text):
    has_conservation_phrase = "conservation of" in text or "conserved" in text
    conserved_quantity = any(
        marker in text
        for marker in [
            "charge",
            "energy",
            "mass",
            "momentum",
            "quantity",
            "total amount",
        ]
    )
    if has_conservation_phrase and conserved_quantity:
        return True

    has_total = any(marker in text for marker in ["total", "sum", "conserved"])
    has_constant = any(marker in text for marker in ["constant", "remains", "conservation"])
    has_exchange = any(
        marker in text
        for marker in [
            "accompanied by",
            "component",
            "converted",
            "decrease",
            "exchange",
            "increase",
            "kinetic",
            "potential",
            "transformed",
        ]
    )
    return has_total and has_constant and has_exchange


def _random_walk_code(spec):
    title = _py_string(spec["title"])
    steps = int(max(20, min(_number(spec["parameters"].get("steps"), 300), 2000)))
    walkers = int(max(50, min(_number(spec["parameters"].get("walkers"), 1000), 5000)))
    return f"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
n_walkers = {walkers}
n_steps = {steps}
steps = rng.choice([-1, 1], size=(n_walkers, n_steps))
positions = np.cumsum(steps, axis=1)
times = np.arange(1, n_steps + 1)
mean_squared_displacement = np.mean(positions**2, axis=0)
slope, intercept = np.polyfit(times[max(5, n_steps // 5):], mean_squared_displacement[max(5, n_steps // 5):], 1)

print("Simulation-spec demo")
print("demo_type = random_walk")
print(f"walkers = {{n_walkers}}")
print(f"steps per walker = {{n_steps}}")
print(f"final mean squared displacement = {{mean_squared_displacement[-1]:.3g}}")
print(f"MSD growth slope = {{slope:.3g}} per step")

plt.figure(figsize=(6, 4))
plt.plot(times, mean_squared_displacement, label="simulation")
plt.plot(times, slope * times + intercept, "--", label="linear fit")
plt.xlabel("step")
plt.ylabel("mean squared displacement")
plt.title({title})
plt.legend()
plt.tight_layout()
plt.savefig("demo.png", dpi=150)
""".strip()


def _distribution_code(spec):
    title = _py_string(spec["title"])
    mean = _number(spec["parameters"].get("mean"), 0.0)
    std = abs(_number(spec["parameters"].get("std"), 1.0)) or 1.0
    samples = int(max(100, min(_number(spec["parameters"].get("samples"), 5000), 20000)))
    return f"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
samples = rng.normal(loc={mean!r}, scale={std!r}, size={samples})

print("Simulation-spec demo")
print("demo_type = distribution")
print(f"sample count = {{len(samples)}}")
print(f"sample mean = {{float(np.mean(samples)):.3g}}")
print(f"sample std = {{float(np.std(samples)):.3g}}")

plt.figure(figsize=(6, 4))
plt.hist(samples, bins=50, density=True, alpha=0.75)
plt.xlabel("value")
plt.ylabel("probability density")
plt.title({title})
plt.tight_layout()
plt.savefig("demo.png", dpi=150)
""".strip()


def _time_evolution_code(spec):
    params = spec["parameters"]
    title = _py_string(spec["title"])
    y_label = _py_string(spec["y_label"])
    amplitude = _number(params.get("amplitude"), 1.0)
    angular_frequency = _number(params.get("angular_frequency"), _number(params.get("frequency"), 1.0))
    decay = max(0.0, _number(params.get("decay"), 0.0))
    phase = _number(params.get("phase"), 0.0)
    t_min, t_max = spec["x_range"]
    return f"""
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace({t_min!r}, {t_max!r}, 400)
y = {amplitude!r} * np.exp(-{decay!r} * t) * np.cos({angular_frequency!r} * t + {phase!r})

print("Simulation-spec demo")
print("demo_type = time_evolution")
print(f"time range = {{float(t[0]):.3g}} to {{float(t[-1]):.3g}}")
print(f"value range = {{float(np.min(y)):.3g}} to {{float(np.max(y)):.3g}}")

plt.figure(figsize=(6, 4))
plt.plot(t, y)
plt.xlabel("time")
plt.ylabel({y_label})
plt.title({title})
plt.tight_layout()
plt.savefig("demo.png", dpi=150)
""".strip()


def _phase_space_code(spec):
    params = spec["parameters"]
    title = _py_string(spec["title"])
    amplitude = _number(params.get("amplitude"), 1.0)
    angular_frequency = _number(params.get("angular_frequency"), _number(params.get("frequency"), 1.0))
    phase = _number(params.get("phase"), 0.0)
    t_min, t_max = spec["x_range"]
    return f"""
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace({t_min!r}, {t_max!r}, 500)
position = {amplitude!r} * np.cos({angular_frequency!r} * t + {phase!r})
velocity = -{amplitude!r} * {angular_frequency!r} * np.sin({angular_frequency!r} * t + {phase!r})

print("Simulation-spec demo")
print("demo_type = phase_space")
print(f"position range = {{float(np.min(position)):.3g}} to {{float(np.max(position)):.3g}}")
print(f"velocity range = {{float(np.min(velocity)):.3g}} to {{float(np.max(velocity)):.3g}}")

plt.figure(figsize=(5, 5))
plt.plot(position, velocity)
plt.xlabel("position")
plt.ylabel("velocity")
plt.title({title})
plt.tight_layout()
plt.savefig("demo.png", dpi=150)
""".strip()


def _multi_series_time_evolution_code(spec):
    params = spec["parameters"]
    title = _py_string(spec["title"])
    y_label = _py_string(spec["y_label"])
    total = _number(params.get("total"), 10.0)
    exchange_amplitude = min(abs(_number(params.get("exchange_amplitude"), total / 2.0)), abs(total) / 2.0)
    angular_frequency = _number(params.get("angular_frequency"), _number(params.get("frequency"), 1.0))
    phase = _number(params.get("phase"), 0.0)
    t_min, t_max = spec["x_range"]
    return f"""
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace({t_min!r}, {t_max!r}, 500)
total = {total!r}
exchange = {exchange_amplitude!r} * np.cos({angular_frequency!r} * t + {phase!r})
series_a = total / 2.0 + exchange
series_b = total - series_a
series_total = series_a + series_b

print("Simulation-spec demo")
print("demo_type = multi_series_time_evolution")
print(f"total target = {{total:.3g}}")
print(f"series_a range = {{float(np.min(series_a)):.3g}} to {{float(np.max(series_a)):.3g}}")
print(f"series_b range = {{float(np.min(series_b)):.3g}} to {{float(np.max(series_b)):.3g}}")
print(f"total range = {{float(np.min(series_total)):.3g}} to {{float(np.max(series_total)):.3g}}")

plt.figure(figsize=(6, 4))
plt.plot(t, series_a, label="component A")
plt.plot(t, series_b, label="component B")
plt.plot(t, series_total, "--", label="total")
plt.xlabel("time")
plt.ylabel({y_label})
plt.title({title})
plt.legend()
plt.tight_layout()
plt.savefig("demo.png", dpi=150)
""".strip()


def _numeric_parameters(parameters):
    normalized = {}
    if not isinstance(parameters, dict):
        return normalized
    for key, value in parameters.items():
        if isinstance(key, str):
            number = _maybe_number(value)
            if number is not None:
                normalized[key.strip().lower()] = number
    return normalized


def _range(value, default):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        start = _maybe_number(value[0])
        end = _maybe_number(value[1])
        if start is not None and end is not None and not math.isclose(start, end):
            return [float(min(start, end)), float(max(start, end))]
    return list(default)


def _number(value, default):
    number = _maybe_number(value)
    return float(default if number is None else number)


def _maybe_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        if math.isfinite(number):
            return number
    return None


def _short_text(value, default):
    text = str(value if value is not None else default).strip()
    if len(text) > 160:
        text = text[:157].rstrip() + "..."
    return text


def _py_string(value):
    return json.dumps(_short_text(value, ""))
