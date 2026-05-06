# 04 · Network Protocol

Giao thức trao đổi giữa Server và Client trên TCP/IP cổng **8888**.

## 4.1 Format thông điệp

```
[LOAI_LENH]|[NOI_DUNG]\n
```

| Đặc trưng | Giá trị |
|---|---|
| Encoding | ASCII (không Vietnamese có dấu trong payload) |
| Field separator (trong nội dung) | `|` |
| Message terminator | `\n` (LF) |
| Max length 1 message | 2048 bytes |
| Port mặc định | 8888 |

### Lý do chọn dạng text-line

- Đơn giản để debug (`netcat`, `telnet` đọc được).
- `\n` framing dễ tách ra khỏi TCP stream (đỉ ốt buffer cho đến khi gặp newline).
- Không cần thư viện serialization (Protocol Buffers, FlatBuffers...) — phù hợp ràng buộc đề bài.

## 4.2 Bảng 11 lệnh đầy đủ

### 4.2.1 Server → Client

| Lệnh | Payload | Mô tả |
|---|---|---|
| `START` | `sessionCode\|dateTime` | Mở ca → client bắt đầu hoạt động |
| `MENU_DATA` | `P01,Pho Bo,65000\|B01,Bun Bo,60000\|...` | Danh sách menu |
| `USER_ACK` | `userId\|isNew\|orderCount\|name` | Xác nhận login |
| `SUGGEST` | `P01,0.92\|D01,0.87\|G01,0.71` | Top-3 gợi ý LFM |
| `ORDER_ACK` | `orderId\|OK` hoặc `0\|FAIL` | Phản hồi submit |
| `STOP` | `dateTime` | Đóng ca |

### 4.2.2 Client → Server

| Lệnh | Payload | Mô tả |
|---|---|---|
| `USER_LOGIN` | `clientId\|phoneNumber` | Khách nhập SDT |
| `USER_REGISTER` | `clientId\|phone\|name\|desc` | Khách mới đăng ký tên |
| `ITEM_ADDED` | `clientId\|userId\|itemCode\|currentCodes` | Trigger gợi ý lại |
| `ORDER_SUBMIT` | `clientId\|userId\|P01,2\|D01,1\|total\|discount` | Đơn hoàn chỉnh |
| `HEARTBEAT` | `clientId\|timestamp` | Kiểm tra kết nối (5s/lần) |

## 4.3 Sequence diagram tiêu biểu

### 4.3.1 Khách quen đặt món

```mermaid
sequenceDiagram
    actor TN as Thu ngan
    participant SRV as Server
    participant C1 as Client Ban 1

    C1->>SRV: TCP Connect (slot=0)
    SRV-->>C1: (chua broadcast START vi session chua mo)

    TN->>SRV: stdin "1234"
    SRV->>SRV: SessionLifecycle.start(1234)
    SRV->>C1: START|1234|2026-05-06 09:00
    SRV->>C1: MENU_DATA|P01,Pho Bo,65000|...

    Note over C1: Khach nhap SDT 0901234567
    C1->>SRV: USER_LOGIN|1|0901234567
    SRV->>SRV: AuthService.getOrCreate
    SRV->>SRV: LfmService.topK(userId, [], 3)
    SRV-->>C1: USER_ACK|5|false|22|Anh Nam
    SRV-->>C1: SUGGEST|P01,2.74|D01,2.70|C01,1.48

    Note over C1: Khach them P01
    C1->>SRV: ITEM_ADDED|1|5|P01|P01
    SRV->>SRV: LfmService.topK(userId, [P01], 3)
    SRV-->>C1: SUGGEST|D01,2.70|C01,1.48|B01,1.20

    Note over C1: Khach xac nhan
    C1->>SRV: ORDER_SUBMIT|1|5|P01,2|D01,1|145000|0
    SRV->>SRV: OrderService.create
    SRV->>SRV: LfmService.onlineUpdate
    SRV->>SRV: Database.saveAll (persist .tbl)
    SRV-->>C1: ORDER_ACK|123|OK

    TN->>SRV: stdin "1234" (dong ca)
    SRV->>C1: STOP|2026-05-06 22:00
    SRV->>SRV: ReportService.write report.txt
```

### 4.3.2 Khách mới đăng ký

```mermaid
sequenceDiagram
    actor KH as Khach moi
    participant CLI
    participant SRV
    participant USERS as users table

    KH->>CLI: SDT 0987654321 (chua co trong db)
    CLI->>SRV: USER_LOGIN|1|0987654321
    SRV->>USERS: findByPhone(0987654321) → null
    SRV->>USERS: insert(userId=10, name=empty)
    SRV->>CLI: USER_ACK|10|true|0|

    Note over CLI: isNew=true → hien NameInput

    KH->>CLI: Ten "Tester", desc "Khach moi"
    CLI->>SRV: USER_REGISTER|1|0987654321|Tester|Khach moi
    SRV->>USERS: updateName(10, Tester, Khach moi)
    SRV->>CLI: USER_ACK|10|false|0|Tester
    SRV->>CLI: SUGGEST|... (cold-start: top global)
```

## 4.4 Parser implementation

[shared/protocol.cpp](../../shared/protocol.cpp) cung cấp 2 hàm chính:

```cpp
bool parseMessage(const char* raw, ParsedMsg* out);
int  buildMessage(MsgType type, const char* payload, char* out, int cap);
```

`ParsedMsg` struct:

```cpp
struct ParsedMsg {
    MsgType type;          // enum MsgType
    char    payload[1024]; // phan sau dau '|' dau tien
};
```

Wrapper OOP cho codec ở [server/network/protocol_codec.h](../../server/network/protocol_codec.h)
(class `ProtocolCodec`).

## 4.5 Routing trên server

```mermaid
graph LR
    A[Raw bytes] --> B[TcpServer.recvLoop]
    B --> C{Tach \\n}
    C --> D[1 line]
    D --> E[ProtocolCodec.parse]
    E --> F{ParsedMsg.type}
    F -->|USER_LOGIN| G1[AuthController.handleLogin]
    F -->|USER_REGISTER| G2[AuthController.handleRegister]
    F -->|ITEM_ADDED| G3[OrderController.handleItemAdded]
    F -->|ORDER_SUBMIT| G4[OrderController.handleOrderSubmit]
    F -->|HEARTBEAT| G5[HeartbeatController.handleHeartbeat]
    G1 --> H[Service layer]
    G2 --> H
    G3 --> H
    G4 --> H
    G5 --> H
```

Cài đặt routing: [server/network/message_router.cpp](../../server/network/message_router.cpp).

## 4.6 Nguyên tắc triển khai

| Quy tắc | Lý do |
|---|---|
| **Receiver phải buffer đến `\n`** | TCP có thể chia 1 message thành nhiều `recv()` |
| **Server không trust Client** | Validate SDT, mã món, số lượng, max items mỗi đơn |
| **Session gate** | Mọi handler ≠ HEARTBEAT phải kiểm tra `sessionService.isOpen()` |
| **Heartbeat timeout 15s** | Không nhận heartbeat sau 15s → đánh dấu disconnect |
| **Reconnect 3 lần** | Client tự retry connect với khoảng cách 2s |
| **Order response order** | `USER_ACK` luôn gửi **trước** `SUGGEST` |
| **Persist-on-order** | Sau mỗi `ORDER_SUBMIT` thành công, server gọi `Database.saveAll` |

## 4.7 Khi thêm lệnh mới

Cập nhật đồng bộ:

1. [shared/protocol.h](../../shared/protocol.h) — thêm enum value
2. [shared/protocol.cpp](../../shared/protocol.cpp) — thêm tên vào `NAMES[]`
3. [server/network/message_router.cpp](../../server/network/message_router.cpp) — thêm case
4. Tạo handler trong controller phù hợp
5. Cập nhật file này (04-network-protocol.md)
