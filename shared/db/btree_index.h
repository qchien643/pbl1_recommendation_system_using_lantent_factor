#ifndef DB_BTREE_INDEX_H
#define DB_BTREE_INDEX_H

#include "index.h"
#include <vector>

namespace db {

// B-tree manual (order T = 31, mỗi node lên đến 2T-1=61 keys).
// Multi-value: key trùng → entry (key, rid) sắp xếp phụ theo rid.
// Dùng cho range query: transactions theo user_id, theo timestamp.
class BTreeIndex : public Index {
public:
    static constexpr int T = 31;
    static constexpr int MAX_KEYS = 2 * T - 1;
    static constexpr int MIN_KEYS = T - 1;

    BTreeIndex();
    ~BTreeIndex() override;

    BTreeIndex(const BTreeIndex&) = delete;
    BTreeIndex& operator=(const BTreeIndex&) = delete;

    IndexKind kind()    const override { return IndexKind::BTREE; }
    bool      unique()  const override { return false; }
    size_t    size()    const override { return count_; }

    void      insert(const Value& key, RowId rid) override;
    void      erase (const Value& key, RowId rid) override;
    void      clear() override;

    std::vector<RowId> find (const Value& key) const override;
    std::vector<RowId> range(const Value& lo, const Value& hi) const override;

private:
    struct Entry {
        Value key;
        RowId rid;
        bool less(const Entry& o) const {
            if (key < o.key) return true;
            if (o.key < key) return false;
            return rid < o.rid;
        }
        bool equalKey(const Value& k) const { return !(key < k) && !(k < key); }
    };

    struct Node {
        bool   leaf;
        int    n;                  // số entry
        Entry  entries[MAX_KEYS];
        Node*  children[MAX_KEYS + 1];
        Node() : leaf(true), n(0) {
            for (int i = 0; i < MAX_KEYS + 1; i++) children[i] = nullptr;
        }
    };

    Node*  root_;
    size_t count_;

    // Helpers
    void splitChild(Node* parent, int idx);
    void insertNonFull(Node* node, const Entry& e);
    bool eraseFrom(Node* node, const Entry& target);
    void mergeChildren(Node* parent, int idx);
    void fillChild(Node* parent, int idx);
    void borrowFromPrev(Node* parent, int idx);
    void borrowFromNext(Node* parent, int idx);
    Entry getPredecessor(Node* node, int idx);
    Entry getSuccessor(Node* node, int idx);
    void freeNode(Node* node);

    void collectRange(Node* node, const Value& lo, const Value& hi, std::vector<RowId>& out) const;
    void collectKey  (Node* node, const Value& key, std::vector<RowId>& out) const;
};

} // namespace db

#endif
