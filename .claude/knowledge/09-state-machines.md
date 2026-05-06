# 09 · State Machines

Tài liệu trình bày state machines của 2 actor chính: **Server** và **Client**.

## 9.1 Server state machine

```mermaid
stateDiagram-v2
    [*] --> BOOT: server.exe start

    BOOT --> READY_CLOSED: load .tbl + open TCP listener
    note right of BOOT
        - netInit (Winsock)
        - initRestaurantSchema
        - openAll("data") load tat ca .tbl
        - menuService.loadFromFile(menu.txt)
        - lfmService.initRandom + loadFromRepository
        - lfmService.rebuildOrderHistory
        - tcpServer.start (port 8888)
    end note

    READY_CLOSED --> READY_CLOSED: client connect (no broadcast)
    READY_CLOSED --> READY_CLOSED: input ma SAI
    READY_CLOSED --> SESSION_OPEN: input ma 1-9 chu so

    SESSION_OPEN --> SESSION_OPEN: USER_LOGIN, USER_REGISTER
    SESSION_OPEN --> SESSION_OPEN: ITEM_ADDED → SUGGEST
    SESSION_OPEN --> SESSION_OPEN: ORDER_SUBMIT → ORDER_ACK + saveAll
    SESSION_OPEN --> SESSION_OPEN: HEARTBEAT
    SESSION_OPEN --> SESSION_OPEN: input ma KHAC → reject
    SESSION_OPEN --> CLOSING: input ma KHOP

    CLOSING --> [*]: write report + saveAll + STOP broadcast
    note right of CLOSING
        - sessionService.close
        - reportService.writeForCurrentSession
        - lfmService.saveToRepository
        - database.saveAll("data")
        - tcpServer.broadcast(STOP)
    end note
```

### 9.1.1 Server actions per state

| State | Hành động cho phép |
|---|---|
| `BOOT` | Init network, schema, load data |
| `READY_CLOSED` | Accept TCP connect; reject mọi message ngoại trừ `HEARTBEAT` (vẫn ghi log nhưng không xử lý logic) |
| `SESSION_OPEN` | Xử lý đầy đủ tất cả 5 lệnh từ client; persist mỗi đơn submit |
| `CLOSING` | Save report + flush .tbl + broadcast STOP |

### 9.1.2 Server-side message gating

Mỗi controller có guard `sessionService_.isOpen()`:

```cpp
void AuthController::handleLogin(int slot, const std::string& payload) {
    if (!sessionService_.isOpen()) return;   // session gate
    ...
}
```

Tránh xử lý nhầm khi session chưa mở (vd client cũ còn cache reconnect lúc server vừa restart).

## 9.2 Client state machine

```mermaid
stateDiagram-v2
    [*] --> CONNECTING: client start

    CONNECTING --> WAITING: TCP connected
    CONNECTING --> ERROR: ECONNREFUSED (3 retry)

    WAITING --> PHONE: nhan START + MENU_DATA
    note right of WAITING
        Hien banner "Cho thu ngan mo ca"
    end note

    PHONE --> LOADING: gui USER_LOGIN
    LOADING --> NAME_INPUT: USER_ACK isNew=true
    LOADING --> ORDERING: USER_ACK isNew=false + SUGGEST

    NAME_INPUT --> REGISTERING: gui USER_REGISTER
    REGISTERING --> ORDERING: USER_ACK isNew=false + SUGGEST

    ORDERING --> ORDERING: them mon → ITEM_ADDED → SUGGEST moi
    ORDERING --> INVOICE: nhap 00 / Enter / du 5 mon

    INVOICE --> SUBMITTING: nhap Y → ORDER_SUBMIT
    INVOICE --> ORDERING: nhap N (sua don)

    SUBMITTING --> THANKS: ORDER_ACK orderId|OK
    SUBMITTING --> INVOICE: ORDER_ACK 0|FAIL

    THANKS --> PHONE: 3s countdown auto

    PHONE --> [*]: nhan STOP
    ORDERING --> [*]: nhan STOP
    INVOICE --> [*]: nhan STOP
    ERROR --> [*]: exit
```

### 9.2.1 Client states và UI component tương ứng

| State | Component | Mô tả |
|---|---|---|
| `CONNECTING` | Spinner banner | Đang mở socket |
| `WAITING` | `WaitingScreen.jsx` | Chờ START từ server |
| `PHONE` | `PhoneInput.jsx` | 10 ô nhập SDT |
| `LOADING` | Spinner | Chờ USER_ACK + SUGGEST |
| `NAME_INPUT` | `NameInput.jsx` | Khách mới nhập tên + desc |
| `REGISTERING` | Spinner | Đang lưu thông tin |
| `ORDERING` | `MenuDisplay` + `SuggestPanel` + `OrderSummary` | Chính: chọn món |
| `INVOICE` | `Invoice.jsx` | Hiển thị hóa đơn để xác nhận |
| `SUBMITTING` | Spinner | Đang gửi ORDER_SUBMIT |
| `THANKS` | `DailySummary.jsx` | "Cảm ơn quý khách" + countdown |

### 9.2.2 Reconnect logic

```mermaid
flowchart TD
    A[Mat ket noi] --> B{So lan retry < 3?}
    B -- "Yes" --> C[Sleep 2s]
    C --> D[Try connect]
    D -- "Success" --> E[Quay ve state truoc do]
    D -- "Fail" --> F[Tang retry counter]
    F --> B
    B -- "No" --> G[Hien thi loi va exit]
```

## 9.3 Order builder state (client-side, in-memory)

```mermaid
stateDiagram-v2
    [*] --> EMPTY
    EMPTY --> HAS_ITEMS: add_item (1-4 mon)
    HAS_ITEMS --> HAS_ITEMS: add_item them
    HAS_ITEMS --> FULL: add_item lan thu 5
    FULL --> FINALIZED: tu dong (BR03)
    HAS_ITEMS --> FINALIZED: input 00 hoac Enter trang (BR02)
    FINALIZED --> EMPTY: cancel
    FINALIZED --> [*]: confirm + ORDER_SUBMIT
```

`ClientOrder` struct trong [client/order_builder.h](../../client/order_builder.h):

```cpp
struct ClientOrder {
    char  codes[MAX_ITEMS][4];      // "P01\0", ...
    int   qtys[MAX_ITEMS];
    float prices[MAX_ITEMS];        // snapshot lúc thêm
    char  names[MAX_ITEMS][50];     // tên hiển thị
    int   count;                    // 0..MAX_ITEMS
    float subtotal, discount, total;
};
```

## 9.4 Session state (server-side)

```mermaid
stateDiagram-v2
    [*] --> Closed
    Closed --> Open: SessionService.open(code)
    Open --> Closed: SessionService.close(code) khop
    Open --> Open: close(code) khong khop → false

    note right of Open
        sessions table:
          code = "1234"
          opened_at = now
          status = "O"
    end note

    note right of Closed
        sessions table row:
          closed_at = now
          status = "C"
    end note
```

Persist trong bảng `sessions` (xem [06-data-structures.md](06-data-structures.md) §6.8).

## 9.5 Heartbeat health check

Mỗi client gửi `HEARTBEAT|clientId|timestamp` mỗi **5 giây**.

```mermaid
sequenceDiagram
    participant Client
    participant Server

    loop moi 5s
        Client->>Server: HEARTBEAT|1|<unix_ts>
        Server->>Server: lastHeartbeat[slot] = ts
    end

    Note over Server: Future: timer kiem tra<br/>now - lastHeartbeat > 15s<br/>→ closeSlot(slot)
```

Hiện tại `HeartbeatController` chỉ log sự kiện. Logic timeout-disconnect chưa enforce
chặt — đang để TCP layer tự phát hiện connection drop qua `recv() <= 0`.

## 9.6 Lý do tách state machines

- **Tách rời bố cục UI ↔ business logic**: state ở client là frontend concern,
  state ở server là transaction concern. Mixing → khó debug.
- **Server không lưu client state**: server chỉ biết "slot X đang active hay không";
  state đặt món của client tự quản trên client (ClientOrder struct). Trừ khi `ORDER_SUBMIT`
  gửi lên, server không biết khách đang chọn món gì.
- **Idempotent**: client có thể restart giữa session, kết nối lại, chọn lại từ đầu —
  server không cần khôi phục state đặt món dở.
