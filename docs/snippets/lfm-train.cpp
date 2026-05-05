// server/lfm.cpp — Hàm train Latent Factor Model bằng SGD + early stopping

// Implicit rating từ lịch sử đặt: R[u][i] = log(1 + count)
static float implicitRating(int userId, int itemIdx) {
    int c = orderHistory[userId][itemIdx];
    return c > 0 ? logf(1.0f + (float)c) : 0.0f;
}

void lfmTrainFromHistory(int maxIter) {
    float prevLoss = 1e30f;
    int   noImprove = 0;

    for (int iter = 0; iter < maxIter; iter++) {
        // 1 epoch = duyệt qua mọi cặp (u, i) có rating > 0
        for (int u = 0; u < userCount; u++) {
            for (int i = 0; i < menuCount; i++) {
                if (orderHistory[u][i] <= 0) continue;

                float r_ui  = implicitRating(u, i);
                float r_hat = dot(P[u], Q[i], K);
                float err   = r_ui - r_hat;

                // SGD update đồng thời P[u] và Q[i]
                // (giữ p_old khi tính Q[i] để khớp với matrix_factorization.py)
                for (int k = 0; k < K; k++) {
                    float p_old = P[u][k];
                    float q_old = Q[i][k];
                    P[u][k] += LR * (err * q_old - REG * p_old);
                    Q[i][k] += LR * (err * p_old - REG * q_old);
                }
            }
        }

        // Early stopping nếu loss không cải thiện sau PATIENCE epoch
        float loss = lfmComputeLoss();
        if (loss < prevLoss - MIN_DELTA) {
            prevLoss = loss;
            noImprove = 0;
        } else if (++noImprove >= PATIENCE) {
            return;
        }
    }
}
