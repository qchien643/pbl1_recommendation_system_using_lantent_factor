#include "lfm.h"
#include "../shared/state.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// Init zero-centered uniform [-0.1, 0.1] — match matrix_factorization.py np.random.normal(scale=0.1).
// Positive-only init nhu §7.7 (rand/RAND_MAX * 0.01) bi degenerate voi implicit feedback:
// moi P[u] cung phia trong k-dim → predict dom inated boi global popularity, khong ca nhan hoa.
static float randSmall() {
    return (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * 0.1f;
}

static float dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

// Implicit rating theo §7.3: r_ui = log(1 + orderHistory[u][i]). Tra ve 0 neu chua dat.
static float implicitRating(int userId, int itemIdx) {
    int c = orderHistory[userId][itemIdx];
    return c > 0 ? logf(1.0f + (float)c) : 0.0f;
}

void lfmInitUserVector(int userId) {
    for (int k = 0; k < K; k++) P[userId][k] = randSmall();
}

void lfmInitItemVector(int itemIdx) {
    for (int k = 0; k < K; k++) Q[itemIdx][k] = randSmall();
}

void lfmInit(unsigned int seed) {
    srand(seed);
    for (int u = 0; u < MAX_USERS; u++) {
        for (int k = 0; k < K; k++) P[u][k] = randSmall();
    }
    for (int i = 0; i < MAX_MENU; i++) {
        for (int k = 0; k < K; k++) Q[i][k] = randSmall();
    }
}

void lfmTrainFromHistory(int maxIter) {
    float prevLoss = 1e30f;
    int   noImprove = 0;

    for (int iter = 0; iter < maxIter; iter++) {
        // 1 epoch = duyet qua moi (u, i) co rating
        for (int u = 0; u < userCount; u++) {
            for (int i = 0; i < menuCount; i++) {
                if (orderHistory[u][i] <= 0) continue;
                float r_ui  = implicitRating(u, i);
                float r_hat = dot(P[u], Q[i], K);
                float err   = r_ui - r_hat;
                // Giu P[u] cu trong p_old de dung khi cap nhat Q[i] (theo matrix_factorization.py:61-66)
                for (int k = 0; k < K; k++) {
                    float p_old = P[u][k];
                    float q_old = Q[i][k];
                    P[u][k] += LR * (err * q_old - REG * p_old);
                    Q[i][k] += LR * (err * p_old - REG * q_old);
                }
            }
        }

        float loss = lfmComputeLoss();
        // Early stopping theo §7.4
        if (loss < prevLoss - MIN_DELTA) {
            prevLoss = loss;
            noImprove = 0;
        } else {
            noImprove++;
            if (noImprove >= PATIENCE) {
                printf("  [LFM early stop] iter %d/%d loss=%.6f\n", iter + 1, maxIter, loss);
                return;
            }
        }
    }
}

// §7.6: online SGD 1 pass tren cac mon trong don vua nhan
void lfmOnlineUpdate(int userId, const int* itemIds, const int* qty, int count) {
    if (userId < 0 || userId >= userCount) return;
    for (int idx = 0; idx < count; idx++) {
        int i = itemIds[idx];
        if (i < 0 || i >= menuCount) continue;
        int newCount = orderHistory[userId][i] + qty[idx];
        float r_ui  = logf(1.0f + (float)newCount);
        float r_hat = dot(P[userId], Q[i], K);
        float err   = r_ui - r_hat;
        for (int k = 0; k < K; k++) {
            float p_old = P[userId][k];
            float q_old = Q[i][k];
            P[userId][k] += LR * (err * q_old - REG * p_old);
            Q[i][k]      += LR * (err * p_old - REG * q_old);
        }
        orderHistory[userId][i] = newCount;
    }
}

float lfmPredictScore(int userId, int itemIdx) {
    return dot(P[userId], Q[itemIdx], K);
}

int lfmGetTopK(int userId, const int* excluded, int exCount,
               int* resultIdx, float* resultScore, int topK) {
    float scores[MAX_MENU];
    bool  skip[MAX_MENU];
    for (int i = 0; i < MAX_MENU; i++) { scores[i] = 0.0f; skip[i] = false; }
    for (int e = 0; e < exCount; e++) {
        if (excluded[e] >= 0 && excluded[e] < menuCount) skip[excluded[e]] = true;
    }
    for (int i = 0; i < menuCount; i++) {
        if (!skip[i]) scores[i] = lfmPredictScore(userId, i);
    }

    bool used[MAX_MENU] = {false};
    int  found = 0;
    while (found < topK) {
        int   best = -1;
        float bestScore = -1e30f;
        for (int i = 0; i < menuCount; i++) {
            if (used[i] || skip[i]) continue;
            if (scores[i] > bestScore) { bestScore = scores[i]; best = i; }
        }
        if (best == -1) break;
        resultIdx[found] = best;
        resultScore[found] = bestScore;
        found++;
        used[best] = true;
    }
    return found;
}

float lfmComputeLoss() {
    float loss = 0.0f;
    for (int u = 0; u < userCount; u++) {
        for (int i = 0; i < menuCount; i++) {
            if (orderHistory[u][i] <= 0) continue;
            float r = implicitRating(u, i);
            float h = dot(P[u], Q[i], K);
            float e = r - h;
            loss += e * e;
        }
    }
    // Regularization
    for (int u = 0; u < userCount; u++)
        for (int k = 0; k < K; k++) loss += REG * P[u][k] * P[u][k];
    for (int i = 0; i < menuCount; i++)
        for (int k = 0; k < K; k++) loss += REG * Q[i][k] * Q[i][k];
    return loss;
}

bool lfmSaveModels(const char* pFile, const char* qFile) {
    FILE* f = fopen(pFile, "wb");
    if (!f) return false;
    int k = K;
    fwrite(&userCount, sizeof(int), 1, f);
    fwrite(&k, sizeof(int), 1, f);
    if (userCount > 0) fwrite(P, sizeof(float), (size_t)userCount * K, f);
    fclose(f);

    f = fopen(qFile, "wb");
    if (!f) return false;
    fwrite(&menuCount, sizeof(int), 1, f);
    fwrite(&k, sizeof(int), 1, f);
    if (menuCount > 0) fwrite(Q, sizeof(float), (size_t)menuCount * K, f);
    fclose(f);
    return true;
}

bool lfmLoadModels(const char* pFile, const char* qFile) {
    FILE* f = fopen(pFile, "rb");
    if (!f) return false;
    int cnt = 0, k = 0;
    if (fread(&cnt, sizeof(int), 1, f) != 1 ||
        fread(&k,   sizeof(int), 1, f) != 1 || k != K) {
        fclose(f); return false;
    }
    int uc = cnt > MAX_USERS ? MAX_USERS : cnt;
    if (uc > 0) fread(P, sizeof(float), (size_t)uc * K, f);
    fclose(f);

    f = fopen(qFile, "rb");
    if (!f) return false;
    if (fread(&cnt, sizeof(int), 1, f) != 1 ||
        fread(&k,   sizeof(int), 1, f) != 1 || k != K) {
        fclose(f); return false;
    }
    int mc = cnt > MAX_MENU ? MAX_MENU : cnt;
    if (mc > 0) fread(Q, sizeof(float), (size_t)mc * K, f);
    fclose(f);
    return true;
}
