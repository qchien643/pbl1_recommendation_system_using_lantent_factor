#include "lfm_service.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>

namespace app {

LfmService::LfmService(ILfmRepository& lfmRepo,
                       IUserRepository& userRepo,
                       IMenuRepository& menuRepo,
                       ITransactionRepository& txnRepo)
    : lfmRepo_(lfmRepo), userRepo_(userRepo), menuRepo_(menuRepo), txnRepo_(txnRepo)
{
    std::memset(orderHistory_, 0, sizeof(orderHistory_));
    std::memset(P_, 0, sizeof(P_));
    std::memset(Q_, 0, sizeof(Q_));
}

float LfmService::randSmall() {
    return (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * 0.1f;
}

float LfmService::dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

float LfmService::implicitRating(int64_t userId, int64_t itemIdx) const {
    if (userId < 0 || userId >= MAX_USERS || itemIdx < 0 || itemIdx >= MAX_MENU) return 0.0f;
    int c = orderHistory_[userId][itemIdx];
    return c > 0 ? logf(1.0f + (float)c) : 0.0f;
}

void LfmService::initRandom(unsigned int seed) {
    srand(seed);
    for (int u = 0; u < MAX_USERS; u++)
        for (int k = 0; k < K; k++) P_[u][k] = randSmall();
    for (int i = 0; i < MAX_MENU; i++)
        for (int k = 0; k < K; k++) Q_[i][k] = randSmall();
}

void LfmService::initUserVector(int64_t userId) {
    if (userId < 0 || userId >= MAX_USERS) return;
    for (int k = 0; k < K; k++) P_[userId][k] = randSmall();
}

void LfmService::initItemVector(int64_t itemIdx) {
    if (itemIdx < 0 || itemIdx >= MAX_MENU) return;
    for (int k = 0; k < K; k++) Q_[itemIdx][k] = randSmall();
}

bool LfmService::loadFromRepository() {
    auto pVecs = lfmRepo_.findAllUserVectors();
    auto qVecs = lfmRepo_.findAllItemVectors();
    if (pVecs.empty() && qVecs.empty()) return false;

    for (const auto& rec : pVecs) {
        if (rec.id < 0 || rec.id >= MAX_USERS) continue;
        for (int k = 0; k < K && k < (int)rec.vec.size(); k++) P_[rec.id][k] = rec.vec[k];
    }
    for (const auto& rec : qVecs) {
        if (rec.id < 0 || rec.id >= MAX_MENU) continue;
        for (int k = 0; k < K && k < (int)rec.vec.size(); k++) Q_[rec.id][k] = rec.vec[k];
    }
    return true;
}

void LfmService::saveToRepository() {
    lfmRepo_.clearUsers();
    lfmRepo_.clearItems();
    int64_t userCount = userRepo_.count();
    int64_t menuCount = menuRepo_.count();
    for (int u = 0; u < userCount && u < MAX_USERS; u++) {
        lfmRepo_.saveUserVector(u, P_[u], K);
    }
    for (int i = 0; i < menuCount && i < MAX_MENU; i++) {
        lfmRepo_.saveItemVector(i, Q_[i], K);
    }
}

void LfmService::rebuildOrderHistory() {
    std::memset(orderHistory_, 0, sizeof(orderHistory_));
    auto txns = txnRepo_.findAll();
    for (const auto& t : txns) {
        if (t.userId < 0 || t.userId >= MAX_USERS) continue;
        for (const auto& it : t.items) {
            int64_t mi = menuRepo_.findIndexByCode(it.itemCode);
            if (mi >= 0 && mi < MAX_MENU)
                orderHistory_[t.userId][mi] += (int)it.qty;
        }
    }
}

void LfmService::trainBatch(int maxIter) {
    int64_t userCount = userRepo_.count();
    int64_t menuCount = menuRepo_.count();

    float prevLoss = 1e30f;
    int   noImprove = 0;

    for (int iter = 0; iter < maxIter; iter++) {
        for (int u = 0; u < userCount; u++) {
            for (int i = 0; i < menuCount; i++) {
                if (orderHistory_[u][i] <= 0) continue;
                float r_ui  = implicitRating(u, i);
                float r_hat = dot(P_[u], Q_[i], K);
                float err   = r_ui - r_hat;
                for (int k = 0; k < K; k++) {
                    float p_old = P_[u][k];
                    float q_old = Q_[i][k];
                    P_[u][k] += LR * (err * q_old - REG * p_old);
                    Q_[i][k] += LR * (err * p_old - REG * q_old);
                }
            }
        }
        float loss = computeLoss();
        if (loss < prevLoss - MIN_DELTA) { prevLoss = loss; noImprove = 0; }
        else {
            noImprove++;
            if (noImprove >= PATIENCE) {
                printf("  [LFM early stop] iter %d/%d loss=%.6f\n", iter + 1, maxIter, loss);
                return;
            }
        }
    }
}

void LfmService::onlineUpdate(int64_t userId,
                               const std::vector<int>& itemIndices,
                               const std::vector<int>& quantities) {
    if (userId < 0 || userId >= MAX_USERS) return;
    size_t n = std::min(itemIndices.size(), quantities.size());
    for (size_t idx = 0; idx < n; idx++) {
        int i = itemIndices[idx];
        if (i < 0 || i >= MAX_MENU) continue;
        int newCount = orderHistory_[userId][i] + quantities[idx];
        float r_ui  = logf(1.0f + (float)newCount);
        float r_hat = dot(P_[userId], Q_[i], K);
        float err   = r_ui - r_hat;
        for (int k = 0; k < K; k++) {
            float p_old = P_[userId][k];
            float q_old = Q_[i][k];
            P_[userId][k] += LR * (err * q_old - REG * p_old);
            Q_[i][k]      += LR * (err * p_old - REG * q_old);
        }
        orderHistory_[userId][i] = newCount;
    }
}

float LfmService::predict(int64_t userId, int64_t itemIdx) const {
    if (userId < 0 || userId >= MAX_USERS || itemIdx < 0 || itemIdx >= MAX_MENU) return 0.0f;
    return dot(P_[userId], Q_[itemIdx], K);
}

std::vector<std::pair<int64_t, float>> LfmService::topK(int64_t userId,
                                                          const std::vector<int64_t>& excludedItems,
                                                          int topK) const {
    std::vector<std::pair<int64_t, float>> out;
    if (userId < 0 || userId >= MAX_USERS) return out;

    int64_t menuCount = menuRepo_.count();
    if (menuCount <= 0) return out;

    bool skip[MAX_MENU] = {false};
    for (int64_t e : excludedItems) {
        if (e >= 0 && e < menuCount) skip[e] = true;
    }
    float scores[MAX_MENU] = {0};
    for (int i = 0; i < menuCount; i++) if (!skip[i]) scores[i] = predict(userId, i);

    bool used[MAX_MENU] = {false};
    for (int r = 0; r < topK; r++) {
        int   best = -1;
        float bestScore = -1e30f;
        for (int i = 0; i < menuCount; i++) {
            if (used[i] || skip[i]) continue;
            if (scores[i] > bestScore) { bestScore = scores[i]; best = i; }
        }
        if (best < 0) break;
        out.emplace_back((int64_t)best, bestScore);
        used[best] = true;
    }
    return out;
}

float LfmService::computeLoss() const {
    int64_t userCount = userRepo_.count();
    int64_t menuCount = menuRepo_.count();

    float loss = 0.0f;
    for (int u = 0; u < userCount; u++) {
        for (int i = 0; i < menuCount; i++) {
            if (orderHistory_[u][i] <= 0) continue;
            float r = implicitRating(u, i);
            float h = dot(P_[u], Q_[i], K);
            float e = r - h;
            loss += e * e;
        }
    }
    for (int u = 0; u < userCount; u++)
        for (int k = 0; k < K; k++) loss += REG * P_[u][k] * P_[u][k];
    for (int i = 0; i < menuCount; i++)
        for (int k = 0; k < K; k++) loss += REG * Q_[i][k] * Q_[i][k];
    return loss;
}

int LfmService::getOrderCount(int64_t userId, int64_t itemIdx) const {
    if (userId < 0 || userId >= MAX_USERS || itemIdx < 0 || itemIdx >= MAX_MENU) return 0;
    return orderHistory_[userId][itemIdx];
}

} // namespace app
