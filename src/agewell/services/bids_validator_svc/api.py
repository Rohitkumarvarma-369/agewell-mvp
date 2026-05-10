"""FastAPI wrapper for bids-validator."""

from __future__ import annotations

import json
import subprocess

from pydantic import BaseModel, Field

from agewell.services._common.api import build_app

app = build_app("bids-validator-svc")


class ValidateRequest(BaseModel):
    """Request payload for BIDS validation."""

    bids_dir: str


class ValidateResponse(BaseModel):
    """Response payload for BIDS validation."""

    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@app.post("/validate", response_model=ValidateResponse)
def validate(req: ValidateRequest) -> ValidateResponse:
    """Run bids-validator and normalize the result."""
    result = subprocess.run(
        ["bids-validator", req.bids_dir, "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if not result.stdout:
        return ValidateResponse(ok=result.returncode == 0, errors=[result.stderr.strip()])
    payload = json.loads(result.stdout)
    issues = payload.get("issues", {})
    return ValidateResponse(
        ok=result.returncode == 0,
        errors=[issue.get("reason", str(issue)) for issue in issues.get("errors", [])],
        warnings=[issue.get("reason", str(issue)) for issue in issues.get("warnings", [])],
    )
