#ifndef APP_LFM_REPOSITORY_H
#define APP_LFM_REPOSITORY_H

#include "i_lfm_repository.h"
#include "../../shared/db/database.h"

namespace app {

class LfmRepository : public ILfmRepository {
public:
    explicit LfmRepository(db::Database& db);

    void                              saveUserVector(int64_t userId, const float* vec, int k) override;
    void                              saveItemVector(int64_t itemIdx, const float* vec, int k) override;

    std::optional<std::vector<float>> findUserVector(int64_t userId) override;
    std::optional<std::vector<float>> findItemVector(int64_t itemIdx) override;

    std::vector<LfmVectorRecord>      findAllUserVectors() override;
    std::vector<LfmVectorRecord>      findAllItemVectors() override;

    void                              clearUsers() override;
    void                              clearItems() override;

private:
    db::Database& db_;
    db::Table&    pTable_;   // lfm_p
    db::Table&    qTable_;   // lfm_q

    static std::string             packVec(const float* vec, int k);
    static std::vector<float>      unpackVec(const std::string& blob, int k);
    void                            saveVector(db::Table& tbl, const std::string& idCol,
                                                int64_t id, const float* vec, int k);
    std::optional<std::vector<float>> findVector(db::Table& tbl, const std::string& idCol,
                                                  int64_t id, int k);
    std::vector<LfmVectorRecord>    findAllVectors(db::Table& tbl, const std::string& idCol, int k);
};

} // namespace app

#endif
