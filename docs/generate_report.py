"""Sinh docs/report.docx — bao cao do an PBL1 theo mau Khoa CNTT DHBK.

Format theo template:
  - Times New Roman, body size 13, heading size 14 bold center, code size 12
  - Line spacing 1.3, justified
  - Margins ~ 2.5cm

Chay:
  .venv-report/Scripts/python docs/generate_report.py
Output:
  docs/report.docx
"""
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parent
FIG  = ROOT / "figures"
SNIP = ROOT / "snippets"
OUT  = ROOT / "report.docx"

# Neu file bi lock (Word dang mo), thu output sang ten phu
import sys, os
def _resolve_out():
    if not OUT.exists():
        return OUT
    try:
        # Test file co the ghi khong
        with open(OUT, "a"):
            pass
        return OUT
    except PermissionError:
        alt = OUT.parent / "report-new.docx"
        print(f"[WARN] {OUT.name} dang lock (Word mo?). Output sang {alt.name}", file=sys.stderr)
        return alt

# ==========================================================================
# Style helpers
# ==========================================================================

def set_default_font(doc, font="Times New Roman", size=13):
    """Doc-level default font."""
    style = doc.styles["Normal"]
    style.font.name = font
    style.font.size = Pt(size)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), font)
    pf = style.paragraph_format
    pf.line_spacing = 1.3
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.space_after = Pt(0)
    pf.first_line_indent = Pt(14)

def set_margins(doc, top=2.0, bottom=2.0, left=2.5, right=2.0):
    for section in doc.sections:
        section.top_margin    = Cm(top)
        section.bottom_margin = Cm(bottom)
        section.left_margin   = Cm(left)
        section.right_margin  = Cm(right)

def add_page_break(doc):
    doc.add_page_break()

def add_heading(doc, text, level=1, center=True):
    """level 0 = chapter (size 14 bold center), 1 = section (size 13 bold left)."""
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    if level == 0:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(text)
        run.font.size = Pt(14)
        run.font.bold = True
    elif level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(text)
        run.font.size = Pt(13)
        run.font.bold = True
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(text)
        run.font.size = Pt(13)
        run.font.italic = True
        run.font.bold = True
    run.font.name = "Times New Roman"
    p.paragraph_format.space_before = Pt(12 if level == 0 else 8)
    p.paragraph_format.space_after  = Pt(6)
    return p

def add_paragraph(doc, text, bold=False, italic=False, center=False, indent=True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.JUSTIFY
    if not indent:
        p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(13)
    if bold: run.font.bold = True
    if italic: run.font.italic = True
    return p

def add_bullet(doc, text, level=0):
    """Bullet list (sử dụng style List Bullet built-in)."""
    style_name = "List Bullet" if level == 0 else "List Bullet 2"
    p = doc.add_paragraph(style=style_name)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(13)
    p.paragraph_format.first_line_indent = Pt(0)
    return p

def add_code(doc, text, lang_label=None):
    """Code block — Consolas size 12, single spacing, no indent, light gray bg."""
    if lang_label:
        cap = doc.add_paragraph()
        cap.paragraph_format.first_line_indent = Pt(0)
        r = cap.add_run(f"// {lang_label}")
        r.font.name = "Consolas"; r.font.size = Pt(11); r.font.italic = True
        r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
    for line in text.rstrip("\n").splitlines():
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(line if line else " ")
        run.font.name = "Consolas"
        run.font.size = Pt(11)
    # Trailing spacer
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(6)

def add_image(doc, path, caption=None, width_cm=14):
    if not path.exists():
        add_paragraph(doc, f"[Hình ảnh thiếu: {path.name}]", italic=True, indent=False)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run()
    run.add_picture(str(path), width=Cm(width_cm))
    if caption:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.first_line_indent = Pt(0)
        r = c.add_run(caption)
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)
        r.font.italic = True

def add_table(doc, headers, rows, widths_cm=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Light Grid Accent 1"
    # Header
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.first_line_indent = Pt(0)
        run = p.add_run(h)
        run.bold = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)
    # Data
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.first_line_indent = Pt(0)
            run = p.add_run(str(val))
            run.font.name = "Times New Roman"
            run.font.size = Pt(12)
    if widths_cm:
        for row in table.rows:
            for j, w in enumerate(widths_cm):
                row.cells[j].width = Cm(w)
    # Spacer after table
    doc.add_paragraph()

# ==========================================================================
# Cover page
# ==========================================================================

def add_cover(doc):
    # Top centered text
    for line, size, bold in [
        ("ĐẠI HỌC ĐÀ NẴNG",                13, True),
        ("TRƯỜNG ĐẠI HỌC BÁCH KHOA",       14, True),
        ("KHOA CÔNG NGHỆ THÔNG TIN",       13, True),
        ("", 13, False),
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        if line:
            r = p.add_run(line)
            r.font.name = "Times New Roman"
            r.font.size = Pt(size)
            r.bold = bold

    # Spacer
    for _ in range(3): doc.add_paragraph()

    # Title block
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    r = p.add_run("BÁO CÁO ĐỒ ÁN PBL1\nLẬP TRÌNH TÍNH TOÁN")
    r.font.name = "Times New Roman"; r.font.size = Pt(20); r.bold = True

    doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    r = p.add_run("ĐỀ TÀI:")
    r.font.name = "Times New Roman"; r.font.size = Pt(14); r.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    r = p.add_run("ỨNG DỤNG ĐẶT MÓN ĂN VÀ THANH TOÁN ĐƠN HÀNG\n"
                  "VỚI MÔ HÌNH GỢI Ý CÁ NHÂN HOÁ LATENT FACTOR")
    r.font.name = "Times New Roman"; r.font.size = Pt(18); r.bold = True

    for _ in range(4): doc.add_paragraph()

    # Info table
    info = [
        ("Người hướng dẫn :", "[Tên giảng viên hướng dẫn]"),
        ("Sinh viên thực hiện :", "[Họ và tên sinh viên]"),
        ("Lớp :", "[Mã lớp]"),
        ("Nhóm :", "[Số nhóm]"),
        ("Đề số :", "702"),
    ]
    for label, val in info:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        r1 = p.add_run(label + " ")
        r1.font.name = "Times New Roman"; r1.font.size = Pt(13); r1.bold = True
        r2 = p.add_run(val)
        r2.font.name = "Times New Roman"; r2.font.size = Pt(13)

    for _ in range(3): doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    r = p.add_run("Đà Nẵng, 04/2026")
    r.font.name = "Times New Roman"; r.font.size = Pt(13); r.italic = True

    add_page_break(doc)

# ==========================================================================
# Chapter 0 — MO DAU
# ==========================================================================

def add_intro(doc):
    add_heading(doc, "MỞ ĐẦU", level=0)

    add_paragraph(doc,
        "Trong những năm gần đây, ngành dịch vụ ăn uống tại Việt Nam phát triển "
        "mạnh mẽ, đặc biệt là các nhà hàng có mô hình phục vụ tại bàn với nhiều "
        "khách hàng song song. Bài toán đặt ra là làm thế nào để vừa quản lý "
        "đơn hàng nhanh chóng, vừa giữ chân khách quen thông qua trải nghiệm "
        "cá nhân hoá. Một hệ thống đặt món chỉ in ra menu cố định cho mọi "
        "khách hàng không còn phù hợp khi nhà hàng đã tích luỹ được lịch sử "
        "đặt món của hàng trăm khách quen.")

    add_paragraph(doc,
        "Đề tài số 702 của môn học PBL1 — Lập trình tính toán — yêu cầu xây "
        "dựng một ứng dụng đặt món và thanh toán hoàn chỉnh chạy trên mạng "
        "LAN với kiến trúc một máy chủ thu ngân và nhiều máy khách (bàn ăn). "
        "Điểm đặc sắc của đề tài là yêu cầu tích hợp một mô hình học máy "
        "Latent Factor Model (Matrix Factorization) để gợi ý món cá nhân hoá "
        "theo lịch sử đặt món của từng số điện thoại.")

    add_paragraph(doc,
        "Báo cáo này trình bày toàn bộ quá trình phân tích, thiết kế và cài "
        "đặt hệ thống. Hệ thống được xây dựng với C/C++ thuần dùng parallel "
        "arrays cho lõi nghiệp vụ và socket TCP qua Winsock2; phần giao "
        "diện CLI đẹp được dựng bằng React Ink (Node.js) cho máy khách và "
        "blessed-contrib cho dashboard máy chủ; mô hình LFM được hiện thực "
        "bằng cả Python (làm tham chiếu) lẫn C++ (chạy thật trong server).")

    add_paragraph(doc,
        "Mục tiêu của báo cáo: (1) làm rõ cơ sở lý thuyết của Matrix "
        "Factorization và lý do áp dụng cho bài toán nhà hàng; (2) trình bày "
        "kiến trúc hệ thống và thiết kế dữ liệu kiểu parallel arrays — phù "
        "hợp ràng buộc của môn học; (3) minh hoạ kết quả thực thi qua các "
        "ảnh chụp giao diện và chỉ số đánh giá; (4) đưa ra hướng phát "
        "triển tiếp theo.")

    add_page_break(doc)

# ==========================================================================
# Chapter 1 — TONG QUAN DE TAI
# ==========================================================================

def add_chapter_1(doc):
    add_heading(doc, "Chương 1.  TỔNG QUAN ĐỀ TÀI", level=0)

    add_heading(doc, "1.1.  Bối cảnh và phát biểu bài toán", level=1)
    add_paragraph(doc,
        "Nhà hàng Việt Phong (giả định) phục vụ khách tại bàn. Mỗi bàn được "
        "trang bị một máy tính nhỏ làm máy khách, kết nối mạng nội bộ với "
        "một máy chủ đặt tại quầy thu ngân. Khi mở ca, thu ngân nhập một mã "
        "số (1-9 chữ số) làm mã giao dịch của ca; kết thúc ca, thu ngân "
        "nhập lại đúng mã số đó để đóng ca. Trong suốt ca, khách tại từng "
        "bàn nhập số điện thoại của mình rồi chọn món bằng mã 3 ký tự (vd "
        "P01 cho Phở Bò Tái) — không cần gõ tiếng Việt. Hệ thống ghi nhớ "
        "lịch sử đặt món của mỗi số điện thoại để gợi ý món cho lần sau.")

    add_heading(doc, "1.2.  Các tác nhân (Actors)", level=1)
    add_table(doc,
        headers=["Tác nhân", "Vai trò", "Máy chạy"],
        rows=[
            ["Thu ngân", "Mở/đóng ca, theo dõi thống kê, xem lịch sử khách", "Máy server"],
            ["Khách hàng", "Nhập SĐT, đăng ký tên (lần đầu), chọn món, xác nhận hoá đơn", "Máy client (bàn)"],
            ["Latent Factor Engine", "Tự động gợi ý món cho khách dựa vào lịch sử", "Chạy trong server"],
        ],
        widths_cm=[4, 9, 4])

    add_heading(doc, "1.3.  Phạm vi và ràng buộc", level=1)
    add_bullet(doc, "Tối đa 20 bàn khách kết nối đồng thời đến 1 server (MAX_CLIENTS).")
    add_bullet(doc, "Mỗi đơn hàng tối đa 5 món; menu nhà hàng tối đa 20 món; tối đa 1.000 SĐT khách hàng.")
    add_bullet(doc, "Toàn bộ thao tác nhập liệu chỉ dùng số và ký tự ASCII không dấu (BR16) — phù hợp môi trường terminal.")
    add_bullet(doc, "Giảm giá 25% nếu tổng đơn ≥ 2.000.000đ.")
    add_bullet(doc, "Lõi nghiệp vụ và socket viết bằng C++17 thuần, dùng parallel arrays thay vì OOP nặng (ràng buộc môn học).")
    add_bullet(doc, "Chạy trên Windows + LAN nội bộ, không sử dụng Internet.")

    add_heading(doc, "1.4.  Kiến trúc tổng thể", level=1)
    add_paragraph(doc,
        "Hệ thống tổ chức theo mô hình Server-Client truyền thống qua TCP "
        "socket trên cổng 8888. Server đóng vai trò 'thật sự' của hệ thống "
        "— giữ trạng thái menu, danh sách user, mô hình LFM và toàn bộ "
        "lịch sử giao dịch. Mỗi client là một 'dumb terminal': chỉ render "
        "giao diện và chuyển input của khách thành lệnh TCP gửi lên server.")

    add_image(doc, FIG / "arch-overview.png",
              caption="Hình 1.1 — Kiến trúc tổng thể hệ thống đặt món.",
              width_cm=15)

    add_heading(doc, "1.5.  Các tính năng chính", level=1)
    add_bullet(doc, "Mở/đóng ca làm việc bằng mã số do thu ngân tự chọn.")
    add_bullet(doc, "Khách nhập số điện thoại (10 chữ số bắt đầu bằng 0); khách mới được hỏi thêm tên và mô tả ngắn (tuỳ chọn).")
    add_bullet(doc, "Hiển thị thực đơn 11 món (mã 3 ký tự, có giá), ô nhập mã + số lượng.")
    add_bullet(doc, "Latent Factor Model gợi ý top-3 món cá nhân hoá ngay khi khách đăng nhập, cập nhật khi khách thêm món vào đơn.")
    add_bullet(doc, "Tự động tính giảm giá 25% khi tổng đơn ≥ 2.000.000đ.")
    add_bullet(doc, "Ghi log persistent: mỗi đơn hàng được ghi NGAY xuống đĩa (binary và text) để dashboard đọc real-time.")
    add_bullet(doc, "Xuất báo cáo tổng hợp `report_YYYY-MM-DD.txt` cuối ca; lưu mô hình LFM vào `lfm_P.dat` và `lfm_Q.dat`.")

    add_page_break(doc)

# ==========================================================================
# Chapter 2 — CO SO LY THUYET
# ==========================================================================

def add_chapter_2(doc):
    add_heading(doc, "Chương 2.  CƠ SỞ LÝ THUYẾT", level=0)

    add_heading(doc, "2.1.  Ý tưởng", level=1)
    add_paragraph(doc,
        "Với một nhà hàng có lịch sử đặt món của nhiều khách quen, có thể "
        "khai thác mẫu hình 'khách thích món gì' để gợi ý phù hợp. Cách "
        "đơn giản là đếm tần suất món của từng khách — nhưng cách này không "
        "tổng quát hoá được khi khách chỉ có vài lần đặt (cold-start) và "
        "không khai thác được sự tương đồng giữa các khách.")

    add_paragraph(doc,
        "Latent Factor Model (LFM) — cụ thể là phương pháp Matrix "
        "Factorization — giải quyết bài toán bằng cách 'phân tích' ma trận "
        "tương tác khách-món thành tích của hai ma trận latent. Mỗi khách "
        "và mỗi món được biểu diễn bằng một vector K chiều ẩn; điểm gợi ý "
        "là tích vô hướng của hai vector. Cách này đã được Koren và đồng "
        "nghiệp (Netflix Prize) chứng minh là hiệu quả vượt trội so với "
        "k-NN cho hệ gợi ý.")

    add_heading(doc, "2.2.  Cơ sở lý thuyết", level=1)

    add_heading(doc, "2.2.1.  Mô hình Matrix Factorization", level=2)
    add_paragraph(doc,
        "Gọi U là số khách, I là số món. Ma trận R có kích thước U×I trong "
        "đó R[u][i] biểu diễn 'mức độ thích' của khách u với món i. "
        "Matrix Factorization xấp xỉ R thành tích của hai ma trận latent:")
    add_code(doc,
        "    R [U×I]  ≈  P [U×K]  ·  Qᵀ [K×I]\n\n"
        "    R̂[u][i]  =  P[u] · Q[i]  =  Σ_k  P[u][k] × Q[i][k]\n",
        lang_label="Công thức Matrix Factorization")

    add_paragraph(doc,
        "Trong đó P[u] là vector latent K chiều của khách u, Q[i] là "
        "vector latent K chiều của món i. Tham số K = 10 cân bằng giữa "
        "khả năng biểu diễn và overfit.")

    add_heading(doc, "2.2.2.  Implicit feedback từ lịch sử đặt món", level=2)
    add_paragraph(doc,
        "Hệ thống không có cơ chế đánh sao trực tiếp, nên dùng implicit "
        "feedback: số lần khách u đã đặt món i được biến đổi log để giảm "
        "ảnh hưởng của outlier (khách rất quen):")
    add_code(doc,
        "    R[u][i]  =  log(1 + số_lần_khách_u_đã_đặt_món_i)\n\n"
        "Bảng giá trị mẫu:\n"
        "    0 lần  → 0.00\n"
        "    1 lần  → 0.69\n"
        "    2 lần  → 1.10\n"
        "    5 lần  → 1.79\n"
        "    10 lần → 2.40\n",
        lang_label="Implicit rating")

    add_heading(doc, "2.2.3.  Hàm mất mát và Stochastic Gradient Descent", level=2)
    add_paragraph(doc,
        "Tham số P, Q được học bằng cách tối thiểu hoá hàm mất mát "
        "(squared error có regularization):")
    add_code(doc,
        "    L  =  Σ_(u,i)  (R[u][i] − P[u]·Q[i])²  +  λ (‖P‖² + ‖Q‖²)\n",
        lang_label="Hàm mất mát")
    add_paragraph(doc,
        "Phương pháp tối ưu sử dụng là Stochastic Gradient Descent. "
        "Với mỗi cặp (u, i) có rating, cập nhật đồng thời P[u] và Q[i] "
        "theo công thức:")
    add_code(doc,
        "    e_ui      =  R[u][i] − P[u]·Q[i]\n"
        "    P[u][k]  +=  lr · (e_ui · Q[i][k] − reg · P[u][k])\n"
        "    Q[i][k]  +=  lr · (e_ui · P[u][k] − reg · Q[i][k])\n\n"
        "Tham số mặc định:  K=10,  lr=0.01,  reg=0.02,  max_iter=50.\n",
        lang_label="SGD update")

    add_heading(doc, "2.2.4.  Early stopping", level=2)
    add_paragraph(doc,
        "Để tránh huấn luyện quá lâu khi loss đã hội tụ, thuật toán dừng "
        "sớm khi loss không cải thiện hơn min_delta = 1e-4 trong "
        "patience = 10 epoch liên tiếp.")

    add_heading(doc, "2.2.5.  Online update sau mỗi đơn hàng", level=2)
    add_paragraph(doc,
        "Khi khách kết thúc một đơn, hệ thống chạy 1 pass SGD trên các "
        "cặp (u, i) trong đơn đó (gọi là online update theo §7.6 đặc tả). "
        "Cách này giúp gợi ý ngay lập tức phản ánh sở thích vừa được "
        "thể hiện, không phải chờ đến cuối ca train lại.")

    add_heading(doc, "2.2.6.  Top-K gợi ý", level=2)
    add_paragraph(doc,
        "Với mỗi yêu cầu gợi ý, hệ thống tính tích vô hướng cho mọi món "
        "chưa có trong giỏ hiện tại, sau đó dùng selection sort để lấy "
        "K=3 món có điểm cao nhất. Vì M ≤ 20 nên không cần dùng heap.")

    add_heading(doc, "2.3.  Giao thức TCP và lập trình socket", level=1)
    add_paragraph(doc,
        "Server lắng nghe ở cổng 8888, chấp nhận đến 20 client đồng thời "
        "qua hàm select() để xử lý không-blocking. Mỗi message có dạng "
        "text dòng: '[LOAI_LENH]|[NOI_DUNG]\\n', delimiter là dấu '|', "
        "kết thúc message bằng '\\n'. Client luôn buffer cho đến khi gặp "
        "ký tự '\\n' mới parse — tránh giả định recv() trả về đủ một "
        "message.")

    add_paragraph(doc,
        "Hệ thống định nghĩa 11 loại message: 6 loại Server→Client "
        "(START, MENU_DATA, USER_ACK, SUGGEST, ORDER_ACK, STOP) và 5 "
        "loại Client→Server (USER_LOGIN, USER_REGISTER, ITEM_ADDED, "
        "ORDER_SUBMIT, HEARTBEAT). Server validate lại mọi input: SĐT "
        "10 chữ số bắt đầu '0', mã món tồn tại trong menu, số lượng > 0, "
        "tổng ≤ 5 món/đơn. Đặc biệt, mọi handler đều check "
        "isSessionOpen() — phòng trường hợp client lỗi gửi message "
        "trước khi thu ngân mở ca.")

    add_page_break(doc)

# ==========================================================================
# Chapter 3 — TO CHUC CAU TRUC DU LIEU VA THUAT TOAN
# ==========================================================================

def add_chapter_3(doc):
    add_heading(doc, "Chương 3.  TỔ CHỨC CẤU TRÚC DỮ LIỆU VÀ THUẬT TOÁN", level=0)

    add_heading(doc, "3.1.  Phát biểu bài toán", level=1)
    add_paragraph(doc,
        "Cho trước:  Một file menu.txt liệt kê các món; danh sách rỗng "
        "ban đầu các SĐT khách hàng và lịch sử đặt món.")
    add_paragraph(doc,
        "Yêu cầu xử lý: Tiếp nhận, validate đơn đặt món của khách qua "
        "TCP; cập nhật lịch sử của SĐT vừa đặt và mô hình LFM; trả về "
        "gợi ý cá nhân hoá top-3 món cho mỗi khách; tính giảm giá; ghi "
        "báo cáo tổng kết cuối ca.")

    add_heading(doc, "3.2.  Mô tả đầu vào / đầu ra", level=1)

    add_heading(doc, "3.2.1.  Đầu vào", level=2)
    add_table(doc,
        headers=["Loại", "Định dạng", "Ví dụ"],
        rows=[
            ["Mã số ca", "1-9 ký tự số", "1234"],
            ["Số điện thoại", "10 ký tự số, bắt đầu '0'", "0901234567"],
            ["Tên khách (lần đầu)", "2-35 ký tự ASCII printable", "Nguyen Van A"],
            ["Mô tả khách (tuỳ chọn)", "0-79 ký tự ASCII", "Dan van phong, sang Pho Bo"],
            ["Mã món", "[Prefix][2 chữ số]", "P01, B02, D01"],
            ["Số lượng", "Số nguyên 1-99", "2"],
            ["Kết thúc đơn", "'00' hoặc Enter trắng", "00"],
            ["Xác nhận hoá đơn", "Y / N / Enter", "Y"],
        ],
        widths_cm=[5, 7, 5])

    add_heading(doc, "3.2.2.  Đầu ra", level=2)
    add_bullet(doc, "Hoá đơn từng khách: STT, mã món, tên món, số lượng, đơn giá, thành tiền, tạm tính, giảm giá, tổng cộng.")
    add_bullet(doc, "Top-3 gợi ý mỗi lần USER_LOGIN hoặc ITEM_ADDED.")
    add_bullet(doc, "Báo cáo cuối ca data/reports/report_YYYY-MM-DD.txt: tổng đơn, tổng doanh thu, tổng giảm giá, số đơn được giảm, số SĐT khác nhau, top món bán chạy.")
    add_bullet(doc, "Mô hình LFM lưu xuống đĩa (lfm_P.dat, lfm_Q.dat) để load cho ca tiếp theo.")

    add_heading(doc, "3.3.  Cấu trúc dữ liệu", level=1)

    add_heading(doc, "3.3.1.  Lý do dùng parallel arrays", level=2)
    add_paragraph(doc,
        "Đề bài DUT PBL1 ràng buộc dùng C/C++ cơ bản — không sử dụng OOP "
        "nặng (class hierarchy, std::vector, smart pointer). Cách tiếp "
        "cận parallel arrays phù hợp: mỗi 'thuộc tính' của entity là một "
        "mảng riêng; index i trong mọi mảng liên quan đều trỏ đến cùng "
        "một entity. Ưu điểm: đơn giản, không cấp phát động, dễ "
        "serialize/deserialize ra file binary; nhược điểm: khó refactor "
        "khi schema thay đổi.")

    add_heading(doc, "3.3.2.  Khai báo các mảng chính", level=2)
    snippet = (SNIP / "parallel-arrays.cpp").read_text(encoding="utf-8")
    add_code(doc, snippet, lang_label="shared/state.h — parallel arrays")

    add_heading(doc, "3.3.3.  Sơ đồ ER", level=2)
    add_image(doc, FIG / "er-diagram.png",
              caption="Hình 3.1 — Sơ đồ ER mô tả quan hệ giữa các bảng.",
              width_cm=15)

    add_heading(doc, "3.3.4.  Định dạng file lưu trữ", level=2)
    add_table(doc,
        headers=["File", "Định dạng", "Khi nào ghi"],
        rows=[
            ["data/menu.txt", "Text, mỗi dòng `CODE|NAME|PRICE|CATEGORY`", "Người dùng tự sửa"],
            ["data/users.dat", "Binary: userCount, userPhone[], userName[], userDesc[], userTotalOrders[]", "Sau mỗi USER_REGISTER và mỗi ORDER_SUBMIT"],
            ["data/transactions.dat", "Binary: toàn bộ mảng txn*[]", "Sau mỗi ORDER_SUBMIT + cuối ca"],
            ["data/transactions.log", "Text append-only, 1 dòng/đơn", "Sau mỗi ORDER_SUBMIT"],
            ["data/lfm_P.dat", "Binary: P[userCount][K]", "Cuối ca"],
            ["data/lfm_Q.dat", "Binary: Q[menuCount][K]", "Cuối ca"],
            ["data/reports/report_YYYY-MM-DD.txt", "Text báo cáo", "Cuối ca"],
        ],
        widths_cm=[5, 7, 5])

    add_heading(doc, "3.4.  Thuật toán", level=1)

    add_heading(doc, "3.4.1.  Thuật toán huấn luyện LFM", level=2)
    add_paragraph(doc,
        "Hàm lfmTrainFromHistory() lặp tối đa MAX_ITER × 6 epoch (300 "
        "epoch khi seed). Mỗi epoch duyệt qua tất cả cặp (u, i) có "
        "orderHistory[u][i] > 0 và áp dụng SGD update đồng thời P[u] "
        "và Q[i]. Early stopping kích hoạt khi loss không cải thiện sau "
        "10 epoch liên tiếp.")
    snippet = (SNIP / "lfm-train.cpp").read_text(encoding="utf-8")
    add_code(doc, snippet, lang_label="server/lfm.cpp — train LFM")

    add_paragraph(doc,
        "Phân tích độ phức tạp: Mỗi epoch tốn O(N × M × K) với N là số "
        "user có rating, M là số món, K là số chiều latent. Với 10 user "
        "× 11 món × K=10 = 1.100 phép cập nhật/epoch — chạy hết 300 "
        "epoch chỉ dưới 1 giây.")

    add_heading(doc, "3.4.2.  Online update", level=2)
    snippet = (SNIP / "online-update.cpp").read_text(encoding="utf-8")
    add_code(doc, snippet, lang_label="server/lfm.cpp — online update")

    add_heading(doc, "3.4.3.  Top-K gợi ý", level=2)
    snippet = (SNIP / "top-k.cpp").read_text(encoding="utf-8")
    add_code(doc, snippet, lang_label="server/lfm.cpp — top-K")
    add_paragraph(doc,
        "Độ phức tạp: O(M) để tính score + O(K × M) cho selection sort "
        "= O(M × K). Trên thực tế M ≤ 20 và K = 3 nên rất nhanh.")

    add_heading(doc, "3.4.4.  Xử lý ORDER_SUBMIT trên server", level=2)
    snippet = (SNIP / "handle-order-submit.cpp").read_text(encoding="utf-8")
    add_code(doc, snippet, lang_label="server/socket_server.cpp — handleOrderSubmit")

    add_heading(doc, "3.4.5.  State machine của client", level=2)
    add_image(doc, FIG / "state-client.png",
              caption="Hình 3.2 — State machine của UI khách (12 trạng thái).",
              width_cm=14)
    snippet = (SNIP / "client-app-state.jsx").read_text(encoding="utf-8")
    add_code(doc, snippet, lang_label="cli/src/ClientApp.jsx — state machine")

    add_heading(doc, "3.4.6.  Sequence diagram đặt một đơn hàng", level=2)
    add_image(doc, FIG / "seq-order.png",
              caption="Hình 3.3 — Trình tự message giữa Khách, Client, Server và Thu ngân.",
              width_cm=15)

    add_page_break(doc)

# ==========================================================================
# Chapter 4 — CHUONG TRINH VA KET QUA
# ==========================================================================

def add_chapter_4(doc):
    add_heading(doc, "Chương 4.  CHƯƠNG TRÌNH VÀ KẾT QUẢ", level=0)

    add_heading(doc, "4.1.  Tổ chức chương trình", level=1)
    add_paragraph(doc,
        "Mã nguồn được tổ chức theo nguyên tắc 'shared header + chia "
        "thành module độc lập', tránh chồng chéo phụ thuộc:")

    add_code(doc,
        "pbl1_recommendation_system_using_lantent_factor/\n"
        "├── shared/                    # Header dùng chung\n"
        "│   ├── constants.h            # Hằng số: MAX_USERS, MAX_TXN, K, LR...\n"
        "│   ├── state.h / .cpp         # Toàn bộ parallel arrays\n"
        "│   ├── protocol.h / .cpp      # Enum 11 message + parser\n"
        "│   ├── net.h / .cpp           # Wrapper Winsock2\n"
        "│   ├── utils.h / .cpp         # Tiện ích chung\n"
        "│   └── json.h / .cpp          # Parse JSON nhỏ cho IPC\n"
        "├── server/                    # Mã server\n"
        "│   ├── main_server.cpp        # Entry point + JSON IPC\n"
        "│   ├── socket_server.cpp      # TCP listener + handlers\n"
        "│   ├── menu.cpp               # Load/serialize menu\n"
        "│   ├── user_store.cpp         # CRUD user + save/load users.dat\n"
        "│   ├── transaction_store.cpp  # Per-order storage\n"
        "│   ├── order_store.cpp        # createOrder() + log\n"
        "│   ├── lfm.cpp                # Latent Factor Model\n"
        "│   ├── phone_validator.cpp    # Validate SĐT\n"
        "│   ├── session.cpp            # open/close ca\n"
        "│   └── file_manager.cpp       # Xuất report.txt\n"
        "├── client/                    # Mã client\n"
        "│   ├── main_client.cpp        # Entry + JSON IPC + heartbeat\n"
        "│   ├── socket_client.cpp\n"
        "│   ├── order_builder.cpp\n"
        "│   ├── input_handler.cpp\n"
        "│   └── display.cpp\n"
        "├── cli/                       # Wrapper UI bằng Node.js\n"
        "│   ├── package.json\n"
        "│   └── src/\n"
        "│       ├── ClientApp.jsx      # React Ink (khách)\n"
        "│       ├── server_dashboard.mjs  # blessed-contrib (thu ngân)\n"
        "│       ├── ServerApp.jsx      # React Ink alt cho thu ngân\n"
        "│       ├── ipc.js             # Wrapper stdio JSON\n"
        "│       └── components/        # 8 component (Phone, Name, Menu...)\n"
        "├── tools/                     # Công cụ phụ trợ\n"
        "│   ├── seed_data.cpp          # Sinh 10 personas + ~180 txns\n"
        "│   └── dump_data.mjs          # Đọc binary thành text\n"
        "├── data/                      # Dữ liệu runtime + seed\n"
        "├── tests/                     # End-to-end tests\n"
        "├── matrix_factorization.py    # Reference Python LFM\n"
        "└── CMakeLists.txt\n",
        lang_label="Cây thư mục dự án")

    add_heading(doc, "4.2.  Ngôn ngữ và công cụ cài đặt", level=1)
    add_table(doc,
        headers=["Thành phần", "Công nghệ", "Ghi chú"],
        rows=[
            ["Lõi nghiệp vụ + LFM", "C/C++ 17", "Parallel arrays, không OOP nặng"],
            ["TCP socket", "Winsock2 (Windows)", "Cổng 8888"],
            ["UI khách", "React Ink (Node.js 18+)", "Tiếng Việt có dấu — customer-facing"],
            ["Dashboard thu ngân", "blessed-contrib + React Ink", "English — staff-facing"],
            ["LFM tham chiếu", "Python 3 + NumPy", "matrix_factorization.py"],
            ["Build", "CMake 3.15+, MinGW gcc 15", "Windows native"],
            ["Validate SĐT", "C++ thủ công", "Không regex (ràng buộc)"],
            ["File I/O", "C++ <fstream>", "Binary .dat + text .log"],
        ],
        widths_cm=[5, 5, 7])

    add_heading(doc, "4.3.  Kết quả", level=1)

    add_heading(doc, "4.3.1.  Giao diện chính của chương trình", level=2)

    add_paragraph(doc, "a) Dashboard thu ngân (blessed-contrib, English):", bold=True)
    add_image(doc, FIG / "srv-dashboard-session.png",
              caption="Hình 4.1 — Dashboard thu ngân khi ca đang mở, hiển thị Activity Log.",
              width_cm=15)
    add_image(doc, FIG / "srv-dashboard-customers.png",
              caption="Hình 4.2 — Dashboard chế độ Customers panel (sau khi nhấn Tab).",
              width_cm=15)

    add_paragraph(doc, "b) Giao diện khách (React Ink, tiếng Việt có dấu):", bold=True)
    add_image(doc, FIG / "cli-waiting.png",
              caption="Hình 4.3 — Màn hình chờ thu ngân mở ca.",
              width_cm=12)
    add_image(doc, FIG / "cli-phone.png",
              caption="Hình 4.4 — Nhập số điện thoại 10 chữ số.",
              width_cm=12)
    add_image(doc, FIG / "cli-name.png",
              caption="Hình 4.5 — Khách mới: nhập tên + mô tả ngắn.",
              width_cm=13)
    add_image(doc, FIG / "cli-ordering.png",
              caption="Hình 4.6 — Màn hình đặt món chính: thực đơn, gợi ý cá nhân hoá, đơn hiện tại, ô nhập.",
              width_cm=16)
    add_image(doc, FIG / "cli-invoice.png",
              caption="Hình 4.7 — Hoá đơn xác nhận trước khi gửi.",
              width_cm=15)
    add_image(doc, FIG / "cli-thanks.png",
              caption="Hình 4.8 — Màn hình cảm ơn sau khi đặt thành công.",
              width_cm=13)

    add_heading(doc, "4.3.2.  Kết quả thực thi", level=2)

    add_paragraph(doc, "a) Sinh dữ liệu mẫu (seed):", bold=True)
    add_paragraph(doc,
        "Chạy `./build/seed_data.exe` tạo 10 personas (số điện thoại + "
        "tên + mô tả) và 181 giao dịch riêng lẻ trải dài trong 90 ngày, "
        "sau đó train LFM 300 epoch. Kết quả final loss = 0.981593 — "
        "hội tụ tốt cho một bộ dữ liệu nhỏ.")
    add_image(doc, FIG / "seed-output.png",
              caption="Hình 4.9 — Output của seed_data: 10 user, 181 transactions, top-3 gợi ý cho mỗi khách.",
              width_cm=15)

    add_paragraph(doc, "b) End-to-end test:", bold=True)
    add_paragraph(doc,
        "File tests/e2e_register.mjs kiểm thử 4 ca: (T1) khách quen "
        "đăng nhập trả về tên đã lưu; (T2) khách mới được hỏi tên rồi "
        "trả về USER_ACK với isNew=false; (T3) sau restart server, "
        "user mới được persist (savedUsers tăng lên 11); (T4) đăng "
        "nhập lại khách mới ở session khác — tên vẫn còn. Tất cả 4 "
        "test PASS.")
    add_image(doc, FIG / "e2e-pass.png",
              caption="Hình 4.10 — Kết quả 4/4 test PASS của E2E test.",
              width_cm=15)

    add_paragraph(doc, "c) Persist real-time:", bold=True)
    add_paragraph(doc,
        "File tests/e2e_order_persist.mjs xác nhận: ngay sau khi khách "
        "submit đơn (chưa đóng ca), data/transactions.dat tăng từ N "
        "lên N+1, data/users.dat[phone].totalOrders = 1. Dashboard mở "
        "trong khi server chạy đọc được dữ liệu live.")

    add_paragraph(doc, "d) Tool đọc binary:", bold=True)
    add_paragraph(doc,
        "Lệnh `node tools/dump_data.mjs` parse cả users.dat và "
        "transactions.dat thành text human-readable; có thể chuyển "
        "đầu ra ra file (`> snapshot.txt`) để inspect.")
    add_image(doc, FIG / "dump-data.png",
              caption="Hình 4.11 — Output của dump_data.mjs: 13 users / 184 txns hiện tại.",
              width_cm=15)

    add_heading(doc, "4.3.3.  Nhận xét và đánh giá", level=2)

    add_paragraph(doc, "Ưu điểm đã đạt được:", bold=True)
    add_bullet(doc, "Hoàn thành đầy đủ 16/16 business rules trong đặc tả đề tài 702.")
    add_bullet(doc, "Kiến trúc TCP server-client thực sự, hỗ trợ tối đa 20 client đồng thời.")
    add_bullet(doc, "Latent Factor Model train + online update + persist thành công với loss hội tụ ~0.98 trên dataset 10 user × 11 món.")
    add_bullet(doc, "Persist-on-order: dashboard luôn đọc được dữ liệu mới ngay sau khi khách submit.")
    add_bullet(doc, "UI hai tầng được phân tách rõ ràng: dashboard tiếng Anh cho thu ngân (chuyên nghiệp), client tiếng Việt có dấu cho khách (thân thiện); input cả hai đều ASCII không dấu (BR16).")
    add_bullet(doc, "Có testing E2E tự động qua Node.js IPC, không cần thao tác thủ công.")
    add_bullet(doc, "Tài liệu nội bộ có Knowledge Base (.claude/knowledge/) gồm 11 file Markdown — giúp Claude agent / dev mới onboard nhanh.")

    add_paragraph(doc, "Hạn chế hiện tại:", bold=True)
    add_bullet(doc, "Chỉ chạy trên Windows do dùng Winsock2; muốn chạy Linux cần port sang BSD socket.")
    add_bullet(doc, "Tên khách (userName) phải nhập ASCII không dấu — vì lưu vào char[40] không có encoding multi-byte.")
    add_bullet(doc, "Thực đơn cố định 11 món (hardcoded trong seed); thêm món cần restart server.")
    add_bullet(doc, "LFM chỉ dùng implicit feedback đơn giản; chưa dùng các kỹ thuật nâng cao như BPR (Bayesian Personalized Ranking) hay Neural Collaborative Filtering.")
    add_bullet(doc, "Chưa có cơ chế authentication cho thu ngân; ai cầm máy server đều có thể mở ca.")

    add_page_break(doc)

# ==========================================================================
# Chapter 5 — KET LUAN VA HUONG PHAT TRIEN
# ==========================================================================

def add_chapter_5(doc):
    add_heading(doc, "Chương 5.  KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", level=0)

    add_heading(doc, "5.1.  Kết luận", level=1)
    add_paragraph(doc,
        "Đồ án PBL1 đã hoàn thành toàn bộ yêu cầu của đề tài 702 — Ứng "
        "dụng đặt món ăn và thanh toán đơn hàng có gợi ý cá nhân hoá. "
        "Hệ thống chạy được full E2E từ lúc thu ngân mở ca → khách hàng "
        "đăng nhập / đăng ký → đặt món có gợi ý LFM → xác nhận hoá đơn "
        "→ ghi sổ → đóng ca xuất báo cáo. Tổng cộng có khoảng 5.000 "
        "dòng C++ cho server-client + 2.000 dòng JSX cho UI + 700 dòng "
        "Python cho LFM tham chiếu.")

    add_paragraph(doc,
        "Quá trình thực hiện đồ án đã giúp nhóm sinh viên rèn luyện "
        "đồng thời nhiều kỹ năng: (1) lập trình mạng TCP socket cấp "
        "thấp với Winsock2; (2) quản lý dữ liệu với parallel arrays "
        "thuần — kỹ năng cơ bản nhưng không kém phần thử thách khi "
        "schema thay đổi; (3) hiện thực một thuật toán học máy không "
        "tầm thường từ giấy đến code C++; (4) thiết kế UX cho terminal "
        "với React Ink — thoát khỏi mô hình text 2D cứng nhắc; (5) "
        "hợp tác công cụ AI (Claude) qua knowledge base có cấu trúc.")

    add_paragraph(doc,
        "Kết quả cuối cùng đạt yêu cầu chức năng và có chất lượng đủ "
        "để demo cho người không thuộc dự án.")

    add_heading(doc, "5.2.  Hướng phát triển", level=1)

    add_paragraph(doc, "Ngắn hạn:", bold=True)
    add_bullet(doc, "Cho phép nhập tên khách Unicode (tiếng Việt có dấu) — đổi userName thành char[200] với UTF-8 encoding hoặc dùng wide string.")
    add_bullet(doc, "Đọc menu từ database thay vì file text — dễ thêm món mà không restart.")
    add_bullet(doc, "Authentication thu ngân bằng password đơn giản hash SHA-1.")
    add_bullet(doc, "Tự động backup data/*.dat sang một thư mục riêng theo ngày.")

    add_paragraph(doc, "Trung hạn:", bold=True)
    add_bullet(doc, "Port sang Linux (BSD socket) để có thể chạy trên Raspberry Pi nhỏ làm server cho nhà hàng.")
    add_bullet(doc, "Frontend web cho khách (browser thay terminal) — server có thể giữ nguyên, chỉ thêm tầng WebSocket gateway.")
    add_bullet(doc, "App mobile cho khách quét QR ở bàn để tự đặt món.")

    add_paragraph(doc, "Dài hạn:", bold=True)
    add_bullet(doc, "Thay LFM bằng Neural Collaborative Filtering (NCF) hoặc Two-tower DNN — yêu cầu nhiều dữ liệu hơn nhưng cho gợi ý chính xác hơn.")
    add_bullet(doc, "Triển khai cloud: server chạy trên VPS, các client là PWA — phục vụ chuỗi nhiều cửa hàng.")
    add_bullet(doc, "Tích hợp thanh toán điện tử (VNPay, MoMo) thay vì chỉ in hoá đơn giấy.")
    add_bullet(doc, "Dashboard analytics nâng cao: dự báo doanh thu theo ngày trong tuần, mùa vụ.")

    add_page_break(doc)

# ==========================================================================
# References + Appendix
# ==========================================================================

def add_references(doc):
    add_heading(doc, "TÀI LIỆU THAM KHẢO", level=0)

    refs = [
        '[1] Y. Koren, R. Bell, and C. Volinsky. "Matrix Factorization Techniques '
        'for Recommender Systems". IEEE Computer, 42(8):30-37, 2009.',

        '[2] Y. Hu, Y. Koren, and C. Volinsky. "Collaborative Filtering for '
        'Implicit Feedback Datasets". IEEE ICDM, 2008, pp. 263-272.',

        '[3] Microsoft Docs. "Winsock2 reference". '
        'https://learn.microsoft.com/en-us/windows/win32/winsock/. Truy cập 04/2026.',

        '[4] Vadim Demedes. "Ink — React for CLIs". '
        'https://github.com/vadimdemedes/ink. Truy cập 04/2026.',

        '[5] Yaron Naveh. "blessed-contrib — Build dashboards with ASCII/ANSI '
        'art and javascript". https://github.com/yaronn/blessed-contrib. '
        'Truy cập 04/2026.',

        '[6] B. Stroustrup. "The C++ Programming Language", 4th Edition. '
        'Addison-Wesley, 2013.',

        '[7] Khoa Công nghệ Thông tin, Trường Đại học Bách Khoa Đà Nẵng. '
        '"Đặc tả đề số 702 — Ứng dụng đặt món và thanh toán đơn hàng". '
        'Tài liệu nội bộ PBL1, 2024.',

        '[8] Charu C. Aggarwal. "Recommender Systems: The Textbook". '
        'Springer, 2016. Chương 3: Model-Based Collaborative Filtering.',
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(r)
        run.font.name = "Times New Roman"
        run.font.size = Pt(13)

    add_page_break(doc)

def add_appendix(doc):
    add_heading(doc, "PHỤ LỤC — MÃ NGUỒN", level=0)

    add_paragraph(doc,
        "Phụ lục này trích các đoạn mã then chốt. Toàn bộ mã nguồn có "
        "tại repository: pbl1_recommendation_system_using_lantent_factor/.",
        italic=True)

    files_to_include = [
        ("A. Khai báo parallel arrays — shared/state.h",         "parallel-arrays.cpp"),
        ("B. Huấn luyện LFM bằng SGD — server/lfm.cpp",          "lfm-train.cpp"),
        ("C. Online update sau ORDER_SUBMIT — server/lfm.cpp",   "online-update.cpp"),
        ("D. Top-K gợi ý — server/lfm.cpp",                       "top-k.cpp"),
        ("E. Xử lý ORDER_SUBMIT — server/socket_server.cpp",     "handle-order-submit.cpp"),
        ("F. State machine UI khách — cli/src/ClientApp.jsx",    "client-app-state.jsx"),
    ]
    for title, fname in files_to_include:
        add_heading(doc, title, level=1)
        path = SNIP / fname
        if path.exists():
            add_code(doc, path.read_text(encoding="utf-8"))
        else:
            add_paragraph(doc, f"[Thiếu file {fname}]", italic=True)

# ==========================================================================
# Main
# ==========================================================================

def main():
    doc = Document()
    set_default_font(doc)
    set_margins(doc)

    add_cover(doc)
    add_intro(doc)
    add_chapter_1(doc)
    add_chapter_2(doc)
    add_chapter_3(doc)
    add_chapter_4(doc)
    add_chapter_5(doc)
    add_references(doc)
    add_appendix(doc)

    out = _resolve_out()
    doc.save(out)
    print(f"OK — generated {out}  ({out.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
