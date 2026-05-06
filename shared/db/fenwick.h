#ifndef DB_FENWICK_H
#define DB_FENWICK_H

#include <cstdint>
#include <cstddef>
#include <vector>

namespace db {

// Fenwick (BIT) — prefix-sum động.
// Dùng cho: doanh thu lũy kế (transactions.total), số lần bán (item sales).
// 1-indexed; tree_[0] không dùng. update(i, +x) → prefixSum(i) cộng x.
class FenwickTree {
public:
    FenwickTree() : n_(0) {}
    explicit FenwickTree(size_t n) : n_(n), tree_(n + 1, 0) {}

    void resize(size_t newN);     // bảo toàn data, mở rộng / thu hẹp
    void clear();

    void    update(size_t idx, int64_t delta);   // 1-indexed
    int64_t prefixSum(size_t idx) const;          // sum [1..idx]
    int64_t rangeSum(size_t lo, size_t hi) const; // [lo..hi] inclusive

    size_t  size() const { return n_; }

private:
    size_t               n_;
    std::vector<int64_t> tree_;
};

} // namespace db

#endif
