from pathlib import Path
import subprocess
import sys


def test_committed_submission_artifacts_are_complete():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/validate_submission_artifacts.py", "--root", str(root)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Submission artifact validation passed." in result.stdout
