// server/lfm.cpp — Top-K gợi ý dùng selection sort (M ≤ 20 nên không cần heap)

float lfmPredictScore(int userId, int itemIdx) {
    return dot(P[userId], Q[itemIdx], K);
}

int lfmGetTopK(int userId,
               const int* excluded, int exCount,
               int* resultIdx, float* resultScore, int topK)
{
    float scores[MAX_MENU];
    bool  skip[MAX_MENU];
    for (int i = 0; i < MAX_MENU; i++) { scores[i] = 0.0f; skip[i] = false; }

    // Loại các món khách đã thêm vào đơn hiện tại
    for (int e = 0; e < exCount; e++) {
        if (excluded[e] >= 0 && excluded[e] < menuCount) {
            skip[excluded[e]] = true;
        }
    }

    // Tính score cho mỗi món còn lại
    for (int i = 0; i < menuCount; i++) {
        if (!skip[i]) scores[i] = lfmPredictScore(userId, i);
    }

    // Selection sort lấy top-K (K=3, MAX_MENU=20 → O(K·M))
    bool used[MAX_MENU] = {false};
    int  found = 0;
    while (found < topK) {
        int   best = -1;
        float bestScore = -1e30f;
        for (int i = 0; i < menuCount; i++) {
            if (used[i] || skip[i]) continue;
            if (scores[i] > bestScore) {
                bestScore = scores[i];
                best = i;
            }
        }
        if (best == -1) break;
        resultIdx[found]   = best;
        resultScore[found] = bestScore;
        used[best] = true;
        found++;
    }
    return found;
}
