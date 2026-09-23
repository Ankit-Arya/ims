from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_fresh_env_generator_replaces_current_database_placeholders(tmp_path: Path):
    project = tmp_path / "ims"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    (project / ".env.example").write_text(
        (ROOT / ".env.example").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (scripts / "init_env.py").write_text(
        (ROOT / "scripts" / "init_env.py").read_text(encoding="utf-8"), encoding="utf-8"
    )

    subprocess.run([sys.executable, str(scripts / "init_env.py")], cwd=project, check=True)
    generated = (project / ".env").read_text(encoding="utf-8")

    assert "replace-with-strong-database-password" not in generated
    assert "replace-with-at-least-32-random-characters" not in generated
    assert "replace-with-another-long-random-value" not in generated
    assert "BOOTSTRAP_ADMIN_PASSWORD=change-this-before-running" not in generated
    assert "OPENAI_API_KEY=" in generated
