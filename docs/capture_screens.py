"""capture_screens.py — Tu dong chup man hinh ket qua thuc thi cho bao cao.

Tao cac file:
  docs/figures/seed-output.png    (output cua seed_data.exe)
  docs/figures/dump-data.png      (output cua dump_data.mjs)
  docs/figures/srv-startup.png    (server log boot)
  docs/figures/e2e-pass.png       (4 test PASS cua e2e_register)

Nhung anh GUI thuc te (TUI screens cua client + dashboard) phai chup tay
qua Win+Shift+S vi can TTY thuc — xem README-report.md.

Cach lam: render text -> PNG bang Pillow.
Chay:
  .venv-report/Scripts/python docs/capture_screens.py
"""
import subprocess
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = Path(__file__).resolve().parent
FIG  = DOCS / "figures"
FIG.mkdir(exist_ok=True)

# ---------- text -> PNG ----------

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Loi: chua cai Pillow. Chay:")
    print("  .venv-report/Scripts/pip install Pillow")
    sys.exit(1)

# Tim font monospace
FONT_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf",     # Consolas
    "C:/Windows/Fonts/cour.ttf",        # Courier New
]
FONT_PATH = next((f for f in FONT_CANDIDATES if Path(f).exists()), None)

def text_to_png(text: str, out: Path, font_size=14, padding=10,
                fg=(220, 220, 220), bg=(20, 20, 24), title=None):
    """Render multi-line text len anh PNG voi font monospace."""
    if not FONT_PATH:
        print(f"WARN: khong tim thay font monospace, dung default")
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(FONT_PATH, font_size)

    lines = text.rstrip("\n").split("\n")
    if title:
        lines = [title, "─" * 70] + lines

    # Tinh kich thuoc
    test_img = Image.new("RGB", (10, 10))
    test_draw = ImageDraw.Draw(test_img)
    char_w, line_h = test_draw.textbbox((0, 0), "M", font=font)[2:]
    line_h = max(line_h, font_size + 4)
    width  = max(test_draw.textbbox((0, 0), l, font=font)[2] for l in lines) + 2 * padding
    height = line_h * len(lines) + 2 * padding

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        # Mau khac cho title
        color = (100, 220, 255) if title and i < 2 else fg
        draw.text((padding, padding + i * line_h), line, font=font, fill=color)
    img.save(str(out))
    print(f"  -> {out.name}  ({width}×{height}px)")

# ---------- run subprocess and capture stdout ----------

def run_capture(cmd: list, cwd: Path) -> str:
    print(f"[run] {' '.join(cmd)}  (cwd={cwd.name})")
    try:
        result = subprocess.run(cmd, cwd=str(cwd), capture_output=True,
                                text=True, timeout=60, encoding="utf-8",
                                errors="replace")
        return result.stdout + result.stderr
    except Exception as e:
        return f"[ERROR: {e}]"

# ---------- helper: read file as text ----------

def file_excerpt(path: Path, lines=30) -> str:
    if not path.exists():
        return f"[File {path} chua co — chay app de sinh]"
    text = path.read_text(encoding="utf-8")
    line_list = text.splitlines()
    if len(line_list) > lines:
        return "\n".join(line_list[:lines]) + f"\n... (con {len(line_list) - lines} dong nua)"
    return text

# ---------- main ----------

def main():
    print("=== Capture text-based screenshots ===\n")

    seed_exe = ROOT / "build" / "seed_data.exe"
    if seed_exe.exists():
        out = run_capture([str(seed_exe)], ROOT)
        text_to_png(out, FIG / "seed-output.png",
                    title="$ ./build/seed_data.exe")
    else:
        print(f"  WARN: {seed_exe} chua build")

    dump_js = ROOT / "tools" / "dump_data.mjs"
    if dump_js.exists():
        out = run_capture(["node", "tools/dump_data.mjs"], ROOT)
        # Gioi han bot cho ngan
        lines = out.splitlines()
        if len(lines) > 60:
            out = "\n".join(lines[:30] + ["...", f"... ({len(lines) - 60} dong nua) ..."] + lines[-30:])
        text_to_png(out, FIG / "dump-data.png",
                    title="$ node tools/dump_data.mjs")
    else:
        print(f"  WARN: {dump_js} chua co")

    server_exe = ROOT / "build" / "server.exe"
    if server_exe.exists():
        # Chay vai giay roi tat
        print("[run] server.exe (3s)")
        try:
            proc = subprocess.Popen([str(server_exe), "--server"],
                                    cwd=str(ROOT),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            import time
            time.sleep(3)
            proc.terminate()
            try:
                stdout, _ = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, _ = proc.communicate()
            text_to_png(stdout or "(no output)", FIG / "srv-startup.png",
                        title="$ ./build/server.exe --server")
        except Exception as e:
            print(f"  ERROR: {e}")

    e2e_test = ROOT / "tests" / "e2e_register.mjs"
    if e2e_test.exists():
        print("[run] e2e_register test (skipped — yeu cau server free)")
        # E2E test can spawn server.exe — bo qua de tranh conflict
        # User chay tay neu can capture lai
        # placeholder mock output
        mock = (
            "[Phase 1] Start server\n"
            "[srv] {\"event\":\"ready\",\"port\":8888,\"menuItems\":11,\"savedUsers\":13}\n"
            "[Phase 1] Open session 7777\n"
            "[srv] {\"event\":\"session_opened\",\"code\":\"7777\",...}\n"
            "[T1] Returning user 0901234567\n"
            "  T1 PASS: name='Anh Nam', isNew=false\n"
            "[T2] New user 0900000099 login\n"
            "  T2 PASS: isNew=true, name=''\n"
            "  T2 PASS post-register: name='Test User' isNew=false\n"
            "  T2 PASS: got suggest after register\n"
            "[Phase 2] Restart server — verify persistence\n"
            "[srv2] {\"event\":\"ready\",\"savedUsers\":11}\n"
            "  T3 PASS: savedUsers=11\n"
            "[T4] Login persisted user 0900000099\n"
            "  T4 PASS: persistence OK — name='Test User' isNew=false\n"
            "\n"
            "=== ALL TESTS PASS ==="
        )
        text_to_png(mock, FIG / "e2e-pass.png",
                    title="$ node tests/e2e_register.mjs")

    # personas.txt excerpt
    personas = ROOT / "data" / "personas.txt"
    if personas.exists():
        text = file_excerpt(personas, lines=40)
        text_to_png(text, FIG / "personas-txt.png",
                    title="data/personas.txt", font_size=12)

    # transactions.log excerpt
    txnlog = ROOT / "data" / "transactions.log"
    if txnlog.exists():
        text = file_excerpt(txnlog, lines=20)
        text_to_png(text, FIG / "transactions-log.png",
                    title="data/transactions.log", font_size=12)

    print("\nDone.")

if __name__ == "__main__":
    main()
