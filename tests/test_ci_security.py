from __future__ import annotations

from pathlib import Path

import yaml


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_dependency_review_workflow_blocks_high_severity_changes():
    workflow = yaml.safe_load(
        (_root() / ".github" / "workflows" / "dependency-review.yml").read_text(
            encoding="utf-8"
        )
    )
    assert "pull_request" in workflow[True]
    steps = workflow["jobs"]["dependency-review"]["steps"]
    action = next(step for step in steps if "dependency-review-action" in step.get("uses", ""))
    assert action["uses"] == (
        "actions/dependency-review-action@"
        "a1d282b36b6f3519aa1f3fc636f609c47dddb294"
    )
    assert action["with"]["fail-on-severity"] == "high"


def test_release_workflow_attests_distributions_before_upload():
    workflow = yaml.safe_load(
        (_root() / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    )
    assert workflow["permissions"]["id-token"] == "write"
    assert workflow["permissions"]["attestations"] == "write"
    steps = workflow["jobs"]["build-attest-upload"]["steps"]
    attest = next(step for step in steps if step.get("uses", "").startswith("actions/attest@"))
    assert attest["uses"] == (
        "actions/attest@59d89421af93a897026c735860bf21b6eb4f7b26"
    )
    subject = attest["with"]["subject-path"]
    assert subject == "dist/*"
    upload_step = next(step for step in steps if "gh release upload" in step.get("run", ""))
    upload_index = steps.index(upload_step)
    attest_index = steps.index(attest)
    assert attest_index < upload_index
    assert "--clobber" not in upload_step["run"]
    assert attest["with"]["subject-path"] == "dist/*"
    assert any("release tag" in step.get("run", "") for step in steps)


def test_dependabot_covers_python_and_github_actions():
    config = yaml.safe_load(
        (_root() / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    )
    ecosystems = {entry["package-ecosystem"] for entry in config["updates"]}
    assert ecosystems == {"pip", "github-actions"}


def test_all_remote_actions_are_pinned_to_full_commit_sha():
    import re

    pattern = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@([0-9a-f]{40})$")
    for workflow_path in (_root() / ".github" / "workflows").glob("*.yml"):
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        for job in workflow.get("jobs", {}).values():
            for step in job.get("steps", []):
                action = step.get("uses")
                if action and not action.startswith("./"):
                    assert pattern.fullmatch(action), (workflow_path, action)


def test_checkout_does_not_persist_credentials():
    for workflow_path in (_root() / ".github" / "workflows").glob("*.yml"):
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        for job in workflow.get("jobs", {}).values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("actions/checkout@"):
                    assert step.get("with", {}).get("persist-credentials") is False


def test_ci_and_release_installs_use_validated_constraints():
    for workflow_name in ("ci.yml", "release.yml"):
        text = (_root() / ".github" / "workflows" / workflow_name).read_text(
            encoding="utf-8"
        )
        for line in text.splitlines():
            if "pip install" in line and '-e ".[dev]"' in line:
                assert "-c constraints/validated.txt" in line


def test_validated_constraints_exclude_unqualified_xgboost_33():
    pyproject = (_root() / "pyproject.toml").read_text(encoding="utf-8")
    constraints = (_root() / "constraints" / "validated.txt").read_text(
        encoding="utf-8"
    )
    assert "xgboost-cpu>=3.2,<3.3" in pyproject
    assert "xgboost-cpu==3.2.0" in constraints
