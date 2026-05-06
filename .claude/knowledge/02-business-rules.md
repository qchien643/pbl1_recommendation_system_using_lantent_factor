# 02 · Business Rules — Quy tắc nghiệp vụ

Toàn bộ 16 quy tắc bắt buộc theo đặc tả [phan-tich-du-an-702.md](../../phan-tich-du-an-702.md) §4.2.

## 2.1 Bảng tổng hợp 16 BR

| ID | Quy tắc | Cài đặt |
|---|---|---|
| **BR01** | Khách chỉ được đặt tối đa **5 món** mỗi đơn | `MAX_ITEMS = 5` trong [shared/constants.h](../../shared/constants.h) |
| **BR02** | Nhấn Enter trắng hoặc nhập `00` để kết thúc chọn món | [client/input_handler.cpp](../../client/input_handler.cpp) |
| **BR03** | Đủ 5 món → tự động kết thúc, in hóa đơn ngay | OrderBuilder client-side |
| **BR04** | Tổng đơn ≥ 2.000.000đ → giảm **25%** | `OrderService::computeDiscount` |
| **BR05** | Ca làm việc định danh bằng **mã số** (1–9 chữ số) | `SessionService::open` |
| **BR06** | Đóng ca: nhập lại đúng mã số đã mở | `SessionService::close` |
| **BR07** | Toàn bộ đơn ghi ra file khi kết thúc ca | `ReportService::writeForCurrentSession` |
| **BR08** | Server nhận mã giao dịch → broadcast `START` tới tất cả Client | `SessionLifecycle::start` |
| **BR09** | Client **chỉ hoạt động** sau khi nhận tín hiệu `START` | Client state machine |
| **BR10** | Mỗi đơn từ Client gửi về Server **ngay lập tức** qua TCP | `MSG_ORDER_SUBMIT` |
| **BR11** | Khách phải nhập **SDT 10 chữ số** trước khi đặt món | `AuthService::isValidPhone` |
| **BR12** | SDT là **`user_id`** trong LFM, lưu xuyên ca | `users` table — HashIndex(phone) UNIQUE |
| **BR13** | Món có **mã 3 ký tự** (P01, B02...) | `MenuService::isValidCode` |
| **BR14** | Gợi ý món dùng **LFM** — cá nhân hóa theo SDT | `LfmService::topK` |
| **BR15** | Tra SDT → chào khách quen + gợi ý thông minh hơn | `AuthController::handleLogin` |
| **BR16** | Mọi input chỉ dùng **số** + **ASCII không dấu** | UI client validate |

## 2.2 Phân nhóm theo chức năng

```mermaid
mindmap
  root((16 Business Rules))
    Quan ly ca
      BR05 Ma so ca
      BR06 Khop ma de dong
      BR07 Xuat report
      BR08 Broadcast START
    Xac thuc khach
      BR11 SDT 10 so
      BR12 SDT la user_id
      BR15 Chao khach quen
    Dat mon
      BR01 Toi da 5 mon
      BR02 Ket thuc bang 00
      BR03 Tu dong khi du 5
      BR13 Ma 3 ky tu
    Giam gia
      BR04 25% khi >= 2tr
    Mang
      BR09 Cho START
      BR10 Gui ngay
    Goi y LFM
      BR14 Top-3 ca nhan hoa
    UX
      BR16 Khong tieng Viet co dau
```

## 2.3 Flowchart: Discount logic (BR04)

```mermaid
flowchart TD
    A[ORDER_SUBMIT arrived] --> B[Resolve items qua MenuRepository]
    B --> C["subtotal = sum price * qty"]
    C --> D{"subtotal >= 2,000,000d ?"}
    D -- "Yes" --> E["discount = subtotal * 0.25"]
    D -- "No" --> F["discount = 0"]
    E --> G["total = subtotal - discount"]
    F --> G
    G --> H[Save TransactionRecord]
    H --> I[Send ORDER_ACK orderId OK]
```

## 2.4 State machine: Mở/đóng ca (BR05–BR08)

```mermaid
stateDiagram-v2
    [*] --> Closed: server start
    Closed --> Open: input ma so (1-9 digits)
    Open --> Open: input ma khac → reject
    Open --> Closed: input ma KHOP
    Closed --> [*]: shutdown

    note right of Open
        BR08 broadcast START + MENU_DATA
        BR11 ready nhan USER_LOGIN
        Persist-on-order: moi ORDER_SUBMIT flush .tbl
    end note

    note right of Closed
        BR07 ghi report YYYY-MM-DD.txt
        Save lfm_p.tbl, lfm_q.tbl
        Broadcast STOP
    end note
```

## 2.5 Flowchart: Đặt món (BR01–BR03, BR13)

```mermaid
flowchart TD
    Start([Bat dau dat mon]) --> NhapMa[Khach nhap MA MON + SO LUONG]
    NhapMa --> Valid{"Ma 3 ky tu hop le BR13?"}
    Valid -- Khong --> Loi[Bao loi, nhap lai]
    Loi --> NhapMa
    Valid -- Co --> Them[Them vao don hien tai]
    Them --> Du{"Du 5 mon BR03?"}
    Du -- "Yes" --> Auto[Tu dong ket thuc]
    Du -- "No" --> Tiep{"Khach nhap 00 hoac Enter trang BR02?"}
    Tiep -- "Yes" --> Done[Ket thuc]
    Tiep -- "No" --> NhapMa
    Auto --> Hoa[In hoa don preview]
    Done --> Hoa
    Hoa --> XN{"Khach xac nhan Y BR10?"}
    XN -- "Yes" --> Submit[MSG_ORDER_SUBMIT gui ngay]
    XN -- "No" --> NhapMa
    Submit --> End([Cho ORDER_ACK])
```

## 2.6 Validation summary (BR11, BR13, BR16)

| Field | Quy tắc | Code reference |
|---|---|---|
| SDT | 10 chữ số, bắt đầu '0', ASCII | `AuthService::isValidPhone` |
| Mã món | 3 ký tự: prefix [PBCGADT] + 2 chữ số + tồn tại trong menu | `MenuService::isValidCode` |
| Số lượng | Số nguyên dương ≤ 99 | `OrderController::handleOrderSubmit` |
| Mã ca | 1–9 ký tự, không trống | `SessionService::open` |
| Tên khách | ASCII no diacritics ≤ 39 ký tự | `AuthController::handleRegister` |

**BR16 — Lý do "ASCII only":** Terminal Windows mặc định không hỗ trợ Unicode đầy đủ; ép input ASCII tránh
được bug rendering, đảm bảo hash của SDT/code chính xác và file binary `.tbl` không cần lo encoding.
UI client vẫn hiển thị tiếng Việt có dấu cho khách (vd "Phở Bò"), nhưng input chỉ là số/ASCII.
