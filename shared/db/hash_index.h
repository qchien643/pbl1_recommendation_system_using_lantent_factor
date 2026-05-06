#ifndef DB_HASH_INDEX_H
#define DB_HASH_INDEX_H

#include "index.h"
#include <vector>
#include <cstdint>

namespace db {

// Hash table với separate-chaining. Manual implementation (DSA showcase).
// Dùng cho equality lookup: phone → user_id, code → menu_idx.
class HashIndex : public Index {
public:
    explicit HashIndex(bool unique = false, size_t initialBuckets = 1024);
    ~HashIndex() override;

    HashIndex(const HashIndex&) = delete;
    HashIndex& operator=(const HashIndex&) = delete;

    IndexKind  kind()    const override { return IndexKind::HASH; }
    bool       unique()  const override { return unique_; }
    size_t     size()    const override { return count_; }

    void       insert(const Value& key, RowId rid) override;
    void       erase (const Value& key, RowId rid) override;
    void       clear() override;

    std::vector<RowId> find (const Value& key) const override;
    std::vector<RowId> range(const Value&, const Value&) const override { return {}; }

    // Diagnostics
    size_t bucketCount() const { return buckets_.size(); }
    double loadFactor()  const { return buckets_.empty() ? 0.0 : (double)count_ / (double)buckets_.size(); }

private:
    struct Node {
        Value  key;
        RowId  rid;
        Node*  next;
        Node(Value k, RowId r) : key(std::move(k)), rid(r), next(nullptr) {}
    };

    std::vector<Node*> buckets_;
    size_t             count_;
    bool               unique_;

    size_t bucketOf(const Value& key) const {
        return (size_t)(key.hash() % (uint64_t)buckets_.size());
    }
    void rehash(size_t newSize);
    void freeAll();
};

} // namespace db

#endif
