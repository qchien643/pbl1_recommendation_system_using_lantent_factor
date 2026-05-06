#ifndef APP_LFM_SERVICE_H
#define APP_LFM_SERVICE_H

#include "../repositories/i_lfm_repository.h"
#include "../repositories/i_user_repository.h"
#include "../repositories/i_menu_repository.h"
#include "../repositories/i_transaction_repository.h"
#include "../../shared/constants.h"
#include <utility>
#include <vector>

namespace app {

// LFM Service — đóng gói model state (P, Q matrices, orderHistory aggregate)
// + business logic (train, predict, top-K, online update).
//
// Persistence delegated to ILfmRepository (load/save vectors).
// Aggregate orderHistory rebuilt từ ITransactionRepository ở startup.
//
// Hot-path: P, Q, orderHistory dùng flat arrays (dense) cho perf training.
class LfmService {
public:
    LfmService(ILfmRepository& lfmRepo,
               IUserRepository& userRepo,
               IMenuRepository& menuRepo,
               ITransactionRepository& txnRepo);

    // Init model — random uniform [-0.1, 0.1]. Gọi 1 lần lúc server start.
    void                                          initRandom(unsigned int seed);
    void                                          initUserVector(int64_t userId);
    void                                          initItemVector(int64_t itemIdx);

    // Load P, Q từ repository (lfm_p / lfm_q tables) vào flat arrays.
    bool                                          loadFromRepository();
    void                                          saveToRepository();

    // Rebuild orderHistory[u][i] từ tất cả transactions persistent.
    void                                          rebuildOrderHistory();

    // Train batch SGD trên orderHistory[][]; early stopping với patience.
    void                                          trainBatch(int maxIter);

    // Online update: 1-pass SGD trên các (userId, itemIdx, qty) trong đơn vừa submit.
    void                                          onlineUpdate(int64_t userId,
                                                                const std::vector<int>& itemIndices,
                                                                const std::vector<int>& quantities);

    // Predict score = P[u] · Q[i].
    float                                         predict(int64_t userId, int64_t itemIdx) const;

    // Top-K gợi ý cho userId, exclude items đã có trong đơn hiện tại.
    // Trả vector cặp (menuIdx, score) sorted descending.
    std::vector<std::pair<int64_t, float>>        topK(int64_t userId,
                                                        const std::vector<int64_t>& excludedItems,
                                                        int topK) const;

    float                                         computeLoss() const;

    // Truy cập orderHistory cho test/debug
    int                                           getOrderCount(int64_t userId, int64_t itemIdx) const;

private:
    ILfmRepository&         lfmRepo_;
    IUserRepository&        userRepo_;
    IMenuRepository&        menuRepo_;
    ITransactionRepository& txnRepo_;

    float P_[MAX_USERS][K];
    float Q_[MAX_MENU][K];
    int   orderHistory_[MAX_USERS][MAX_MENU];

    static float dot(const float* a, const float* b, int n);
    static float randSmall();
    float        implicitRating(int64_t userId, int64_t itemIdx) const;
};

} // namespace app

#endif
