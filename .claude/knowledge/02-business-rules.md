# 02 · Business Rules

Nguồn: [phan-tich-du-an-702.md](../../phan-tich-du-an-702.md) §4.2.

## Toàn bộ 16 rules

| ID | Quy tắc |
|---|---|
| **BR01** | Khách chỉ được đặt tối đa **5 món** mỗi đơn hàng |
| **BR02** | Nhấn **Enter trắng** hoặc nhập **`00`** để kết thúc chọn món |
| **BR03** | Đủ 5 món → tự động kết thúc, in hóa đơn ngay |
| **BR04** | Tổng đơn **≥ 2.000.000đ** → giảm **25%** |
| **BR05** | Ca làm việc được định danh bằng **mã số** (vd: `1234`) |
| **BR06** | Kết thúc ca: nhập lại đúng mã số đã mở → đóng ca |
| **BR07** | Toàn bộ đơn hàng ghi ra **file** khi kết thúc ca |
| **BR08** | Server nhận mã giao dịch → broadcast `START` tới tất cả Client |
| **BR09** | Client **chỉ hoạt động** sau khi nhận tín hiệu `START` từ Server |
| **BR10** | Mỗi đơn từ Client gửi về Server **ngay lập tức** qua TCP socket |
| **BR11** | **Khách phải nhập SDT (10 chữ số)** trước khi đặt món |
| **BR12** | SDT là **`user_id`** trong Latent Factor Model — lưu xuyên ca |
| **BR13** | Món có **mã 3 ký tự** (vd `P01`) — khách chỉ nhập mã, không nhập tên |
| **BR14** | Gợi ý món dùng **LFM** — cá nhân hóa theo SDT từng khách |
| **BR15** | Hệ thống tra SDT → chào khách quen và gợi ý thông minh hơn |
| **BR16** | Mọi thao tác nhập liệu chỉ dùng **số** và **mã ASCII không dấu** |

## Nhóm theo chức năng

### Quản lý ca (BR05 – BR07)
- Thu ngân tự chọn mã số khi mở ca.
- Đóng ca yêu cầu nhập lại **đúng** mã số đã mở → chống đóng nhầm.
- Xuất file báo cáo `report_YYYY-MM-DD.txt` + lưu P, Q matrices — xem [08-file-formats.md](08-file-formats.md).

### Xác thực khách (BR11, BR12, BR15)
- Bắt buộc nhập SDT trước khi vào menu.
- Validate: 10 chữ số, bắt đầu bằng `'0'`, toàn chữ số.
- SDT = `user_id` trong LFM → xuyên ca, xuyên session — lưu trong `users.dat`.

### Đặt món (BR01, BR02, BR03, BR13)
- Max 5 món / đơn.
- Mã món 3 ký tự (P01, B02, …) — xem [03-menu-codes.md](03-menu-codes.md).
- Kết thúc: `00` hoặc Enter trắng, hoặc tự động khi đủ 5 món.

### Giảm giá (BR04)
- Chỉ một ngưỡng duy nhất: tổng ≥ **2.000.000đ** → giảm 25% (làm tròn đến đồng).
- Không có rule khác chồng lên.

### Mạng (BR08, BR09, BR10)
- Server chủ động broadcast `START` / `STOP`.
- Client idle cho đến khi nhận `START`.
- Không batch — mỗi đơn gửi ngay. Xem [04-network-protocol.md](04-network-protocol.md).

### LFM (BR14, BR15)
- Gợi ý top-3 cho mỗi khách ngay khi nhập SDT.
- Khách quen → dựa vào P[u] đã học. Khách mới → mặc định top món phổ biến.
- Xem [05-lfm-algorithm.md](05-lfm-algorithm.md).

### UX (BR16)
- Không yêu cầu khách gõ tiếng Việt có dấu ở bất kỳ đâu.
- Output có thể hiển thị Vietnamese (tên món), nhưng input chỉ số + ASCII.
- Xem [07-ux-cli-design.md](07-ux-cli-design.md).
