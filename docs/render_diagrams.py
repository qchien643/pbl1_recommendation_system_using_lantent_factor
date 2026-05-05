"""Render docs/diagrams/*.mmd → docs/figures/*.png qua Kroki API.

Chay:  .venv-report/Scripts/python render_diagrams.py
"""
import base64
import zlib
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
DIAG_DIR = ROOT / "diagrams"
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

KROKI = "https://kroki.io/mermaid/png/{}"

def encode(text: str) -> str:
    """Kroki dung deflate + base64-url-safe theo doc."""
    data = zlib.compress(text.encode("utf-8"), 9)
    return base64.urlsafe_b64encode(data).decode("ascii")

def render(mmd_file: Path, out_file: Path) -> bool:
    text = mmd_file.read_text(encoding="utf-8")
    url = KROKI.format(encode(text))
    print(f"[->] {mmd_file.name} ({len(text)} chars)")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"    FAIL HTTP {r.status_code}: {r.text[:200]}")
            return False
        out_file.write_bytes(r.content)
        print(f"    OK -> {out_file.name} ({len(r.content)} bytes)")
        return True
    except Exception as e:
        print(f"    ERROR {e}")
        return False

if __name__ == "__main__":
    files = sorted(DIAG_DIR.glob("*.mmd"))
    if not files:
        print("Khong co .mmd file nao trong docs/diagrams/")
    ok = 0
    for f in files:
        out = FIG_DIR / (f.stem + ".png")
        if render(f, out):
            ok += 1
    print(f"\n{ok}/{len(files)} diagrams da render.")
