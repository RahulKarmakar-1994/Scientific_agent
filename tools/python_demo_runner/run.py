import ast
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ALLOWED_IMPORT_ROOTS = {
    "collections",
    "itertools",
    "math",
    "matplotlib",
    "numpy",
    "random",
    "statistics",
}
BANNED_CALL_ROOTS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
}
BANNED_IMPORT_ROOTS = {
    "os",
    "pathlib",
    "pickle",
    "shlex",
    "shutil",
    "socket",
    "subprocess",
    "sys",
}


def main():
    payload = json.loads(sys.stdin.read() or "{}")
    code = payload.get("code", "")
    topic = payload.get("topic", "python demo")

    safety_errors = _safety_errors(code)
    if safety_errors:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "topic": topic,
                    "safety_errors": safety_errors,
                },
                indent=2,
            )
        )
        return

    with tempfile.TemporaryDirectory(prefix="scientific_agent_demo_") as tmpdir:
        workdir = Path(tmpdir)
        script_path = workdir / "demo.py"
        script_path.write_text(_with_backend_guard(code), encoding="utf-8")

        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=workdir,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )

        if completed.stdout:
            (workdir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")

        output_dir = Path("outputs/python_demos") / str(int(time.time()))
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []
        for path in sorted(workdir.iterdir()):
            if path.is_file() and path.name != "demo.py":
                target = output_dir / path.name
                target.write_bytes(path.read_bytes())
                generated_files.append(str(target))

        print(
            json.dumps(
                {
                    "status": "completed" if completed.returncode == 0 else "failed",
                    "topic": topic,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-6000:],
                    "stderr": completed.stderr[-6000:],
                    "generated_files": generated_files,
                    "safety_notes": [
                        "Executed in a temporary directory with a timeout.",
                        "Generated files were copied to outputs/python_demos/.",
                        "This local runner is for demos only; use Docker for stronger isolation.",
                    ],
                },
                indent=2,
            )
        )


def _safety_errors(code):
    if not code.strip():
        return ["No code was provided."]

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Syntax error: {exc}"]

    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BANNED_IMPORT_ROOTS or root not in ALLOWED_IMPORT_ROOTS:
                    errors.append(f"Import is not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BANNED_IMPORT_ROOTS or root not in ALLOWED_IMPORT_ROOTS:
                errors.append(f"Import is not allowed: {node.module}")
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            root = name.split(".")[0]
            if root in BANNED_CALL_ROOTS:
                errors.append(f"Call is not allowed: {name}")
    return sorted(set(errors))


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _with_backend_guard(code):
    return "import matplotlib\nmatplotlib.use('Agg')\n" + code


if __name__ == "__main__":
    main()
