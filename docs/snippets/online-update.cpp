// server/lfm.cpp — Online SGD 1-pass cập nhật P, Q sau mỗi ORDER_SUBMIT
// Dùng để gợi ý của khách trong ca tiếp theo phản ánh đơn vừa đặt.

void lfmOnlineUpdate(int userId, const int* itemIds, const int* qty, int count) {
    if (userId < 0 || userId >= userCount) return;

    for (int idx = 0; idx < count; idx++) {
        int i = itemIds[idx];
        if (i < 0 || i >= menuCount) continue;

        // Rating mới = log(1 + count_cũ + qty_đơn_này)
        int newCount = orderHistory[userId][i] + qty[idx];
        float r_ui  = logf(1.0f + (float)newCount);
        float r_hat = dot(P[userId], Q[i], K);
        float err   = r_ui - r_hat;

        // 1 SGD step trên P[userId] và Q[i]
        for (int k = 0; k < K; k++) {
            float p_old = P[userId][k];
            float q_old = Q[i][k];
            P[userId][k] += LR * (err * q_old - REG * p_old);
            Q[i][k]      += LR * (err * p_old - REG * q_old);
        }

        // Cập nhật cache aggregate
        orderHistory[userId][i] = newCount;
    }
}
