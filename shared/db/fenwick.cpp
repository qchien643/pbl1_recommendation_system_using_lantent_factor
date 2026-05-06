#include "fenwick.h"

namespace db {

void FenwickTree::resize(size_t newN) {
    tree_.resize(newN + 1, 0);
    n_ = newN;
}

void FenwickTree::clear() {
    for (auto& v : tree_) v = 0;
}

void FenwickTree::update(size_t idx, int64_t delta) {
    if (idx == 0 || idx > n_) return;
    for (size_t i = idx; i <= n_; i += (i & (~i + 1))) tree_[i] += delta;
}

int64_t FenwickTree::prefixSum(size_t idx) const {
    if (idx > n_) idx = n_;
    int64_t s = 0;
    for (size_t i = idx; i > 0; i -= (i & (~i + 1))) s += tree_[i];
    return s;
}

int64_t FenwickTree::rangeSum(size_t lo, size_t hi) const {
    if (lo == 0) lo = 1;
    if (hi < lo) return 0;
    return prefixSum(hi) - prefixSum(lo - 1);
}

} // namespace db
