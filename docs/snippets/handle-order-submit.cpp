// server/socket_server.cpp — Xử lý ORDER_SUBMIT (trích lược)
// Lưu đơn vào parallel arrays, gọi LFM online update, persist file ngay.

static void handleOrderSubmit(int slot, const char* payload) {
    if (!isSessionOpen()) {                                      // Session gate
        sendTo(slot, MSG_ORDER_ACK, "0|FAIL");
        return;
    }
    char tokens[16][256];
    int n = splitByPipe(payload, tokens, 16);
    if (n < 4) { sendTo(slot, MSG_ORDER_ACK, "0|FAIL"); return; }

    int clientId = atoi(tokens[0]);
    int userId   = atoi(tokens[1]);
    int itemsEnd = n - 2;        // 2 trường cuối: total, discount

    OrderInput in;
    memset(&in, 0, sizeof(in));
    in.userId   = userId;
    in.clientId = clientId;
    if (userId >= 0 && userId < userCount) {
        strncpy(in.phone, userPhone[userId], 10);
    }

    // Parse từng cặp "P01,2"
    in.itemCount = 0;
    for (int t = 2; t < itemsEnd && in.itemCount < MAX_ITEMS; t++) {
        char code[8] = {0}; int qty = 0;
        if (sscanf(tokens[t], "%7[^,],%d", code, &qty) == 2) {
            int idx = findMenuIndex(code);
            if (idx >= 0) {
                in.itemIdx[in.itemCount] = idx;
                in.qty[in.itemCount]     = qty;
                in.itemCount++;
            }
        }
    }
    if (in.itemCount == 0) { sendTo(slot, MSG_ORDER_ACK, "0|FAIL"); return; }

    // 1) Tạo đơn (push vào order*[] + txn*[] + transactions.log)
    int oid = createOrder(&in);
    if (oid < 0) { sendTo(slot, MSG_ORDER_ACK, "0|FAIL"); return; }

    // 2) LFM online SGD update P, Q + tăng orderHistory
    lfmOnlineUpdate(userId, in.itemIdx, in.qty, in.itemCount);

    // 3) Persist NGAY để dashboard đọc được trong phiên
    saveTransactions("data/transactions.dat");
    saveUsers("data/users.dat");

    char ack[64];
    snprintf(ack, sizeof(ack), "%d|OK", oid);
    sendTo(slot, MSG_ORDER_ACK, ack);
}
