#ifndef APP_I_LFM_REPOSITORY_H
#define APP_I_LFM_REPOSITORY_H

#include <cstdint>
#include <optional>
#include <vector>

namespace app {

struct LfmVectorRecord {
    int64_t            id;          // user_id hoặc item_idx
    std::vector<float> vec;         // length = K
};

// Persistence-only repo cho LFM matrices. Runtime training vẫn dùng flat float
// arrays (P[][] / Q[][]) trong LfmService cho hot-path performance — repo này
// chỉ load/save lúc startup và cuối ca.
class ILfmRepository {
public:
    virtual ~ILfmRepository() = default;

    virtual void                              saveUserVector(int64_t userId, const float* vec, int k) = 0;
    virtual void                              saveItemVector(int64_t itemIdx, const float* vec, int k) = 0;

    virtual std::optional<std::vector<float>> findUserVector(int64_t userId) = 0;
    virtual std::optional<std::vector<float>> findItemVector(int64_t itemIdx) = 0;

    virtual std::vector<LfmVectorRecord>      findAllUserVectors() = 0;
    virtual std::vector<LfmVectorRecord>      findAllItemVectors() = 0;

    virtual void                              clearUsers() = 0;
    virtual void                              clearItems() = 0;
};

} // namespace app

#endif
