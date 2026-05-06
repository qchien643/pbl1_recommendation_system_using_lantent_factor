#include "hash_index.h"
#include <stdexcept>

namespace db {

HashIndex::HashIndex(bool unique, size_t initialBuckets)
    : count_(0), unique_(unique)
{
    if (initialBuckets < 4) initialBuckets = 4;
    buckets_.assign(initialBuckets, nullptr);
}

HashIndex::~HashIndex() { freeAll(); }

void HashIndex::freeAll() {
    for (Node* h : buckets_) {
        while (h) { Node* n = h->next; delete h; h = n; }
    }
    buckets_.clear();
    count_ = 0;
}

void HashIndex::clear() {
    freeAll();
    buckets_.assign(1024, nullptr);
}

void HashIndex::rehash(size_t newSize) {
    if (newSize < 4) newSize = 4;
    std::vector<Node*> old = std::move(buckets_);
    buckets_.assign(newSize, nullptr);
    for (Node* h : old) {
        while (h) {
            Node* next = h->next;
            size_t b = (size_t)(h->key.hash() % (uint64_t)buckets_.size());
            h->next = buckets_[b];
            buckets_[b] = h;
            h = next;
        }
    }
}

void HashIndex::insert(const Value& key, RowId rid) {
    if (loadFactor() > 0.75) rehash(buckets_.size() * 2);

    size_t b = bucketOf(key);

    if (unique_) {
        for (Node* h = buckets_[b]; h; h = h->next) {
            if (h->key == key) {
                throw std::runtime_error("HashIndex::insert — UNIQUE violation");
            }
        }
    }
    Node* node = new Node(key, rid);
    node->next = buckets_[b];
    buckets_[b] = node;
    ++count_;
}

void HashIndex::erase(const Value& key, RowId rid) {
    size_t b = bucketOf(key);
    Node** prev = &buckets_[b];
    while (*prev) {
        Node* cur = *prev;
        if (cur->key == key && cur->rid == rid) {
            *prev = cur->next;
            delete cur;
            --count_;
            return;
        }
        prev = &cur->next;
    }
}

std::vector<RowId> HashIndex::find(const Value& key) const {
    std::vector<RowId> out;
    size_t b = bucketOf(key);
    for (Node* h = buckets_[b]; h; h = h->next) {
        if (h->key == key) out.push_back(h->rid);
    }
    return out;
}

} // namespace db
