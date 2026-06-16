import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class JobStore:
    """Persist agent runs under outputs/jobs/<job_id>/."""

    def __init__(self, root="outputs/jobs"):
        self.root = Path(root)

    def create(self, request):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job_id = f"{timestamp}-{uuid4().hex[:8]}"
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        self.write_json(job_dir, "request.json", {"job_id": job_id, "request": request})
        return job_id, job_dir

    def write_json(self, job_dir, name, payload):
        path = Path(job_dir) / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

    def write_text(self, job_dir, name, text):
        path = Path(job_dir) / name
        path.write_text(text or "", encoding="utf-8")
        return str(path)

    def copy_files(self, job_dir, files, subdir="artifacts"):
        copied = []
        target_dir = Path(job_dir) / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for file_path in files or []:
            source = Path(file_path)
            if not source.exists() or not source.is_file():
                continue
            target = target_dir / source.name
            if target.exists():
                target = target_dir / f"{source.stem}-{uuid4().hex[:6]}{source.suffix}"
            shutil.copy2(source, target)
            copied.append(str(target))
        return copied
