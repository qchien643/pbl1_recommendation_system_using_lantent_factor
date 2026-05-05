"""mockup_tui.py — Sinh ASCII mockup cho cac man TUI client + dashboard.

Day la mockup, KHONG chup tu app thuc. User nen replace bang screenshot
that bang Win+Shift+S khi co the.

Chay:
  .venv-report/Scripts/python docs/mockup_tui.py
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DOCS = Path(__file__).resolve().parent
FIG  = DOCS / "figures"
FIG.mkdir(exist_ok=True)

# Tim font monospace, uu tien font ho tro Unicode tot
FONT_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf",     # Consolas — tot cho Unicode
    "C:/Windows/Fonts/cour.ttf",        # Courier New
]
FONT_PATH = next((f for f in FONT_CANDIDATES if Path(f).exists()), None)

def render(text: str, out: Path, font_size=15, padding=14,
           bg=(15, 15, 20), default_fg=(220, 220, 220),
           color_map=None, title=None):
    """Render multi-line text len PNG.

    color_map = list of (line_idx_or_pattern, color) — uu tien tu tren xuong.
    """
    color_map = color_map or []
    font = ImageFont.truetype(FONT_PATH, font_size) if FONT_PATH else ImageFont.load_default()

    lines = text.rstrip("\n").split("\n")

    test_img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(test_img)
    line_h = font_size + 6
    width  = max(draw.textbbox((0, 0), l, font=font)[2] for l in lines) + 2 * padding
    height = line_h * len(lines) + 2 * padding

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        color = default_fg
        for cm in color_map:
            patt, col = cm
            if isinstance(patt, int) and patt == i:
                color = col; break
            if isinstance(patt, str) and patt in line:
                color = col; break
        draw.text((padding, padding + i * line_h), line, font=font, fill=color)
    img.save(str(out))
    print(f"  -> {out.name}  ({width}×{height}px)")

# Mau ANSI chuan
YELLOW = (250, 220, 80)
CYAN   = (90, 200, 240)
GREEN  = (100, 220, 100)
MAGENTA= (220, 110, 220)
BLUE   = (100, 160, 240)
RED    = (240, 100, 100)
GRAY   = (130, 130, 130)
WHITE  = (240, 240, 240)

# ============================================================
# Client mockups (Vietnamese co dau)
# ============================================================

CLI_WAITING = """\
 ┌─────────────────────────────────────────────────┐
 │                                                 │
 │     ◆   NHÀ HÀNG VIỆT PHONG   ◆                │
 │                                                 │
 │     ⠹   CHỜ THU NGÂN MỞ CA   ⠸                  │
 │                                                 │
 │     ▲ Thu ngân nhập MÃ SỐ + Enter tại máy chủ   │
 │       để bắt đầu phục vụ.                       │
 │                                                 │
 └─────────────────────────────────────────────────┘

           (Esc hoặc Ctrl+C để thoát)"""

CLI_PHONE = """\
 ┌─────────────────────────────────────────────────────┐
 │                                                     │
 │   ▲   NHẬP SỐ ĐIỆN THOẠI để nhận gợi ý cá nhân hóa  │  ▲
 │                                                     │
 │   ◉ SĐT:  [0] [9] [0] [1] [2] [_] [_] [_] [_] [_]   │
 │                                                     │
 │   ▲ 10 chữ số, bắt đầu bằng 0. Bấm Enter xác nhận.  │
 │                                                     │
 └─────────────────────────────────────────────────────┘"""

CLI_NAME = """\
 ╔═══════════════════════════════════════════════════════╗
 ║                                                       ║
 ║   ★    KHÁCH MỚI — VUI LÒNG CHO BIẾT TÊN    ★         ║
 ║                                                       ║
 ║   ◆ SĐT: 0900000001                                   ║
 ║                                                       ║
 ║   ▶ Tên của bạn:                                      ║
 ║       Nguyen Van A_                                   ║
 ║       ▲ 2-35 ký tự ASCII (không dấu). Enter tiếp tục. ║
 ║                                                       ║
 ╚═══════════════════════════════════════════════════════╝"""

CLI_ORDERING = """\
   Bàn 05  ·  Mã GD: 1234  ·  SĐT: 0901234567  ·  ⠋  [ĐẶT MÓN]

   ◆  Chào mừng trở lại, Anh Nam!  Bạn đã đặt 22 đơn.

 ┌─ ◆ THỰC ĐƠN (11 món) ◆ ─────────┐  ┌─ ✦ GỢI Ý CHO BẠN ─────────┐
 │   MÃ    TÊN MÓN          GIÁ    │  │  ★  D01   Trà Đá          │
 │ ◆ P01   Phở Bò Tái      65.000d │  │  ◆  P01   Phở Bò Tái      │
 │ ◆ P02   Phở Gà          55.000d │  │  ▲  C01   Cơm Tấm Sườn Bì │
 │ ▲ B01   Bún Bò Huế      60.000d │  └───────────────────────────┘
 │ ▲ B02   Bún Riêu        55.000d │
 │ ■ C01   Cơm Tấm Sườn Bì 75.000d │
 │ ◉ D01   Trà Đá          15.000d │
 │   ...                           │
 └─────────────────────────────────┘

 ┌─ ◆ ĐƠN HIỆN TẠI: ▪ [P01 x2] ─────────────────────┐
 │ ● Số món: 1/5   ● Còn lại: 4   ◆ Tạm tính: 130.000d │
 └────────────────────────────────────────────────────┘

 ┌─ ◉ Nhập: MÃ_MÓN SL  ·  00/Enter trống = Xong ────┐
 │ ▶ D01 1_                                          │
 └───────────────────────────────────────────────────┘"""

CLI_INVOICE = """\
 ╔════════════════════════════════════════════════════╗
 ║                                                    ║
 ║       ◆   H Ó A   Đ Ơ N   ◆                        ║
 ║                                                    ║
 ║   ● Mã GD: 1234     ● SĐT: 0901234567              ║
 ║                                                    ║
 ║   STT  Mã   Tên món          SL  Đơn giá  T.tiền   ║
 ║   ───  ───  ──────────────  ───  ───────  ───────  ║
 ║    1   P01  Phở Bò Tái       2   65.000   130.000  ║
 ║    2   D01  Trà Đá           1   15.000    15.000  ║
 ║                                                    ║
 ║   ▲ Tạm tính:    145.000d                          ║
 ║   ▲ Giảm giá:        0d                            ║
 ║   ★ TỔNG CỘNG:   145.000d                          ║
 ║                                                    ║
 ║  ┌──────────────────────────────────────────────┐  ║
 ║  │ ◉ Xác nhận?  [Y/Enter] = Gửi   [N] = Sửa lại │  ║
 ║  └──────────────────────────────────────────────┘  ║
 ╚════════════════════════════════════════════════════╝"""

CLI_THANKS = """\
 ╔══════════════════════════════════════════════════╗
 ║                                                  ║
 ║   ★    CẢM ƠN QUÝ KHÁCH!    ★                    ║
 ║                                                  ║
 ║   ◆ Đơn hàng #7 đã được ghi nhận trên hệ thống.  ║
 ║   ▲ Món Anh/Chị sẽ sớm được đơn vị phục vụ.      ║
 ║                                                  ║
 ║   Tiếp tục phục vụ khách mới trong  3s...        ║
 ║                                                  ║
 ╚══════════════════════════════════════════════════╝"""

# ============================================================
# Server dashboard mockups (English)
# ============================================================

SRV_DASHBOARD_SESSION = """\
 ┌─ Viet Phong Server ──────────┬─ Session ────────────────┐
 │ VIET PHONG RESTAURANT        │ ● SESSION OPEN           │
 │                              │ Code    : 1234           │
 │ Server ● READY · port 8888   │ Started : 22:32          │
 │ Time: 2026-04-23 22:38:15    │                          │
 │                              │ Re-enter code + Enter:   │
 │                              │ > _                      │
 ├─ Orders Today ─ Revenue (K VND) ─┬─ Overall Stats ──────┤
 │  ┌─┐    ┌─┐  ┌─┐    ┌─┐ ┌─┐ ┌─┐  │ Clients:    3/20     │
 │  │ │    │ │  │ │    │.│ │ │ │ │  │ Users:      13       │
 │  └─┘    └─┘  └─┘    └─┘ └─┘ └─┘  │ Discount:   0d       │
 │                                  │ Discounted: 0        │
 │  3       1.4 K                   │ LFM sugg:   8/12 (66%)│
 ├─ Activity Log [Tab → Customers] ─┴──────────────────────┤
 │ 22:38 Server ready · 11 items · 13 users saved          │
 │ 22:38 Session OPENED code=1234                          │
 │ 22:39 Table 01 CONNECTED                                │
 │ 22:39 Table 01 login 0901234567 (22 prior orders)       │
 │ 22:40 Table 01 added P01                                │
 │ 22:40 Table 01 added D01                                │
 │ 22:41 Table 01 SUBMIT #1 · 145.000d                     │
 │ 22:42 Table 02 CONNECTED                                │
 │ 22:42 Table 02 login 0900000099 (NEW CUSTOMER)          │
 │ 22:43 Table 02 REGISTER 0900000099 = "Test User"        │
 └─────────────────────────────────────────────────────────┘"""

SRV_DASHBOARD_CUSTOMERS = """\
 ┌─ Viet Phong Server ──────────┬─ Session ────────────────┐
 │ VIET PHONG RESTAURANT        │ ● SESSION OPEN           │
 │ Server ● READY · port 8888   │ Code: 1234 · Start:22:32 │
 ├─ Orders Today ─ Revenue ────┬┴─ Overall Stats ──────────┤
 │ 3      1.4 K                │ Clients:3/20  Users:13    │
 ├─ Customers [↑↓ select] [Enter view] ─┬─ Transaction History ─┐
 │ Phone        Name           Orders   │ ◆ Anh Nam · 0901234567│
 │ ────────── ──────────────  ────      │ ▲ 22 orders · Spent:  │
 │ 0901234567   Anh Nam        22  ◀    │   1.320.000d          │
 │ 0912345678   Chi Lan        20       │ ─────────────────────  │
 │ 0923456789   Bac Hung       18       │ ★ History: 22 orders  │
 │ 0934567890   Chi Mai        21       │  ◉ Order #1 · 22:42   │
 │ 0945678901   Anh Minh       23       │     P01 Phở Bò Tái x2 │
 │ 0956789012   Co Tu          20       │     D01 Trà Đá x1     │
 │ 0967890123   Anh Tuan       18       │     = 145.000d        │
 │ 0978901234   Chi Hoa        15       │  ◉ Order #2 · 18:57   │
 │ 0989012345   Bac Sau        21       │     D01 Trà Đá x1     │
 │ 0990123456   (unnamed)       3       │     = 15.000d         │
 │ 0900000099   Test User        1      │  ◉ Order #3 · 17:40   │
 └──────────────────────────────────────┴───────────────────────┘"""

# ============================================================

ITEMS = [
    ("cli-waiting.png",   CLI_WAITING,  YELLOW),
    ("cli-phone.png",     CLI_PHONE,    BLUE),
    ("cli-name.png",      CLI_NAME,     MAGENTA),
    ("cli-ordering.png",  CLI_ORDERING, WHITE),
    ("cli-invoice.png",   CLI_INVOICE,  CYAN),
    ("cli-thanks.png",    CLI_THANKS,   GREEN),
    ("srv-dashboard-session.png",   SRV_DASHBOARD_SESSION, WHITE),
    ("srv-dashboard-customers.png", SRV_DASHBOARD_CUSTOMERS, WHITE),
]

if __name__ == "__main__":
    print("=== Render mockup TUI screens ===")
    print("Note: cac anh nay la MOCKUP ASCII. Chup that bang Win+Shift+S\n"
          "      khi chay app de co anh chinh xac.\n")
    for name, txt, col in ITEMS:
        # Color map don gian
        cmap = []
        for i, line in enumerate(txt.split("\n")):
            if "◆" in line and "Anh Nam" in line: cmap.append((i, GREEN))
            elif "★" in line:  cmap.append((i, YELLOW))
            elif "◉" in line:  cmap.append((i, MAGENTA))
            elif "▲" in line:  cmap.append((i, GRAY))
            elif "●" in line and "OPEN" in line: cmap.append((i, GREEN))
        render(txt, FIG / name, color_map=cmap, default_fg=col)
    print("\nDone — luu nho replace bang screenshot that khi co the.")
