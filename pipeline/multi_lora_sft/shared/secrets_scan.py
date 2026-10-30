"""Scans a directory for credentials before it is published as a Kaggle dataset or notebook output.

Outputs that become datasets are re-uploaded and shared, so a token that slipped into a log, config dump or notebook
cell would leak with them. Findings carry a redacted preview only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".txt", ".md", ".log", ".py", ".ipynb", ".toml", ".cfg", ".ini", ".env", ".sh", ""}
SKIP_SUFFIXES = {".safetensors", ".parquet", ".bin", ".pt", ".pth", ".npz", ".npy", ".zip", ".gz", ".whl", ".png", ".jpg", ".so"}
SUSPECT_FILENAMES = {".env", "kaggle.json", "id_rsa", "id_ed25519", "credentials", ".netrc", ".pypirc"}
SUSPECT_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
MAX_FILE_BYTES = 4 * 1024 * 1024

PATTERNS: dict[str, re.Pattern[str]] = {
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "wandb_key": re.compile(r"\bwandb_v1_[A-Za-z0-9_]{40,}\b"),
    "kaggle_token": re.compile(r"\bKGAT_[0-9a-fA-F]{20,}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "kaggle_json_key": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    "assigned_secret": re.compile(r"""(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b['"]?\s*[:=]\s*['"][A-Za-z0-9_\-./+]{20,}['"]"""),
}
_HEX40 = re.compile(r"\b[0-9a-f]{40}\b")


@dataclass
class Finding:
    path: str
    line: int
    kind: str
    preview: str


def _redact(text: str) -> str:
    t = text.strip()
    return (t[:4] + "..." + f"({len(t)} chars)") if len(t) > 8 else "***"


def scan_text(text: str, path: str = "<text>") -> list[Finding]:
    out: list[Finding] = []
    for no, line in enumerate(text.splitlines(), 1):
        for kind, pat in PATTERNS.items():
            m = pat.search(line)
            if m:
                out.append(Finding(path, no, kind, _redact(m.group(0))))
        if "wandb" in line.lower():  # a bare 40-hex string is only a secret next to 'wandb' (git SHAs look the same)
            m = _HEX40.search(line)
            if m:
                out.append(Finding(path, no, "wandb_key", _redact(m.group(0))))
    return out


def scan_dir(root: str | Path) -> list[Finding]:
    base = Path(root)
    findings: list[Finding] = []
    if not base.exists():
        return findings
    for f in sorted(base.rglob("*")):
        try:
            if not f.is_file():
                continue
            rel = str(f.relative_to(base)).replace("\\", "/")
            if f.name in SUSPECT_FILENAMES or f.suffix.lower() in SUSPECT_SUFFIXES:
                findings.append(Finding(rel, 0, "credential_file", f.name))
                continue
            suffix = f.suffix.lower()
            if suffix in SKIP_SUFFIXES or suffix not in TEXT_SUFFIXES or f.stat().st_size > MAX_FILE_BYTES:
                continue
            findings.extend(scan_text(f.read_text(encoding="utf-8", errors="ignore"), rel))
        except OSError:
            continue
    return findings


def format_findings(findings: list[Finding], limit: int = 20) -> str:
    lines = [f"  {f.path}:{f.line}  {f.kind}  {f.preview}" for f in findings[:limit]]
    if len(findings) > limit:
        lines.append(f"  ... and {len(findings) - limit} more")
    return "\n".join(lines)
