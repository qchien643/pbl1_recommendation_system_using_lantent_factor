#include "btree_index.h"
#include <stdexcept>

namespace db {

BTreeIndex::BTreeIndex() : root_(new Node()), count_(0) {}

BTreeIndex::~BTreeIndex() { freeNode(root_); root_ = nullptr; }

void BTreeIndex::freeNode(Node* node) {
    if (!node) return;
    if (!node->leaf) {
        for (int i = 0; i <= node->n; i++) freeNode(node->children[i]);
    }
    delete node;
}

void BTreeIndex::clear() {
    freeNode(root_);
    root_ = new Node();
    count_ = 0;
}

// ---- Insert ----

void BTreeIndex::insert(const Value& key, RowId rid) {
    Entry e{ key, rid };
    if (root_->n == MAX_KEYS) {
        Node* s = new Node();
        s->leaf = false;
        s->children[0] = root_;
        root_ = s;
        splitChild(s, 0);
        insertNonFull(s, e);
    } else {
        insertNonFull(root_, e);
    }
    ++count_;
}

void BTreeIndex::splitChild(Node* parent, int idx) {
    Node* y = parent->children[idx];
    Node* z = new Node();
    z->leaf = y->leaf;
    z->n = T - 1;
    for (int j = 0; j < T - 1; j++) z->entries[j] = y->entries[j + T];
    if (!y->leaf) {
        for (int j = 0; j < T; j++) z->children[j] = y->children[j + T];
    }
    y->n = T - 1;
    for (int j = parent->n; j >= idx + 1; j--) parent->children[j + 1] = parent->children[j];
    parent->children[idx + 1] = z;
    for (int j = parent->n - 1; j >= idx; j--) parent->entries[j + 1] = parent->entries[j];
    parent->entries[idx] = y->entries[T - 1];
    parent->n++;
}

void BTreeIndex::insertNonFull(Node* node, const Entry& e) {
    int i = node->n - 1;
    if (node->leaf) {
        while (i >= 0 && e.less(node->entries[i])) {
            node->entries[i + 1] = node->entries[i];
            i--;
        }
        node->entries[i + 1] = e;
        node->n++;
    } else {
        while (i >= 0 && e.less(node->entries[i])) i--;
        i++;
        if (node->children[i]->n == MAX_KEYS) {
            splitChild(node, i);
            if (node->entries[i].less(e)) i++;
        }
        insertNonFull(node->children[i], e);
    }
}

// ---- Find (equality) ----

std::vector<RowId> BTreeIndex::find(const Value& key) const {
    std::vector<RowId> out;
    collectKey(root_, key, out);
    return out;
}

void BTreeIndex::collectKey(Node* node, const Value& key, std::vector<RowId>& out) const {
    if (!node) return;
    int i = 0;
    while (i < node->n && node->entries[i].key < key) i++;
    if (!node->leaf) collectKey(node->children[i], key, out);
    while (i < node->n && node->entries[i].equalKey(key)) {
        out.push_back(node->entries[i].rid);
        if (!node->leaf) collectKey(node->children[i + 1], key, out);
        i++;
    }
    if (!node->leaf && i < node->n + 1 && (i == node->n || !node->entries[i].equalKey(key)))
        collectKey(node->children[i], key, out);
}

// ---- Range [lo..hi] inclusive ----

std::vector<RowId> BTreeIndex::range(const Value& lo, const Value& hi) const {
    std::vector<RowId> out;
    collectRange(root_, lo, hi, out);
    return out;
}

void BTreeIndex::collectRange(Node* node, const Value& lo, const Value& hi, std::vector<RowId>& out) const {
    if (!node) return;
    int i = 0;
    while (i < node->n && node->entries[i].key < lo) i++;
    if (!node->leaf) collectRange(node->children[i], lo, hi, out);
    while (i < node->n && !(hi < node->entries[i].key)) {
        if (!(node->entries[i].key < lo)) out.push_back(node->entries[i].rid);
        if (!node->leaf) collectRange(node->children[i + 1], lo, hi, out);
        i++;
    }
}

// ---- Erase ----

void BTreeIndex::erase(const Value& key, RowId rid) {
    Entry target{ key, rid };
    if (eraseFrom(root_, target)) --count_;
    if (root_->n == 0 && !root_->leaf) {
        Node* old = root_;
        root_ = root_->children[0];
        old->children[0] = nullptr;
        delete old;
    }
}

bool BTreeIndex::eraseFrom(Node* node, const Entry& target) {
    int idx = 0;
    while (idx < node->n && node->entries[idx].less(target)) idx++;

    if (idx < node->n && !node->entries[idx].less(target) && !target.less(node->entries[idx])) {
        // Found exact match in this node
        if (node->leaf) {
            for (int i = idx + 1; i < node->n; i++) node->entries[i - 1] = node->entries[i];
            node->n--;
            return true;
        } else {
            // Internal node — replace with predecessor / successor / merge
            if (node->children[idx]->n >= T) {
                Entry pred = getPredecessor(node, idx);
                node->entries[idx] = pred;
                return eraseFrom(node->children[idx], pred);
            } else if (node->children[idx + 1]->n >= T) {
                Entry succ = getSuccessor(node, idx);
                node->entries[idx] = succ;
                return eraseFrom(node->children[idx + 1], succ);
            } else {
                mergeChildren(node, idx);
                return eraseFrom(node->children[idx], target);
            }
        }
    } else {
        if (node->leaf) return false;
        bool atEnd = (idx == node->n);
        if (node->children[idx]->n < T) fillChild(node, idx);
        if (atEnd && idx > node->n) return eraseFrom(node->children[idx - 1], target);
        return eraseFrom(node->children[idx], target);
    }
}

BTreeIndex::Entry BTreeIndex::getPredecessor(Node* node, int idx) {
    Node* cur = node->children[idx];
    while (!cur->leaf) cur = cur->children[cur->n];
    return cur->entries[cur->n - 1];
}

BTreeIndex::Entry BTreeIndex::getSuccessor(Node* node, int idx) {
    Node* cur = node->children[idx + 1];
    while (!cur->leaf) cur = cur->children[0];
    return cur->entries[0];
}

void BTreeIndex::fillChild(Node* parent, int idx) {
    if (idx > 0 && parent->children[idx - 1]->n >= T)       borrowFromPrev(parent, idx);
    else if (idx < parent->n && parent->children[idx + 1]->n >= T) borrowFromNext(parent, idx);
    else {
        if (idx < parent->n) mergeChildren(parent, idx);
        else                 mergeChildren(parent, idx - 1);
    }
}

void BTreeIndex::borrowFromPrev(Node* parent, int idx) {
    Node* child   = parent->children[idx];
    Node* sibling = parent->children[idx - 1];
    for (int i = child->n - 1; i >= 0; i--) child->entries[i + 1] = child->entries[i];
    if (!child->leaf)
        for (int i = child->n; i >= 0; i--) child->children[i + 1] = child->children[i];
    child->entries[0] = parent->entries[idx - 1];
    if (!child->leaf) child->children[0] = sibling->children[sibling->n];
    parent->entries[idx - 1] = sibling->entries[sibling->n - 1];
    child->n++;
    sibling->n--;
}

void BTreeIndex::borrowFromNext(Node* parent, int idx) {
    Node* child   = parent->children[idx];
    Node* sibling = parent->children[idx + 1];
    child->entries[child->n] = parent->entries[idx];
    if (!child->leaf) child->children[child->n + 1] = sibling->children[0];
    parent->entries[idx] = sibling->entries[0];
    for (int i = 1; i < sibling->n; i++) sibling->entries[i - 1] = sibling->entries[i];
    if (!sibling->leaf)
        for (int i = 1; i <= sibling->n; i++) sibling->children[i - 1] = sibling->children[i];
    child->n++;
    sibling->n--;
}

void BTreeIndex::mergeChildren(Node* parent, int idx) {
    Node* child   = parent->children[idx];
    Node* sibling = parent->children[idx + 1];
    child->entries[T - 1] = parent->entries[idx];
    for (int i = 0; i < sibling->n; i++) child->entries[i + T] = sibling->entries[i];
    if (!child->leaf)
        for (int i = 0; i <= sibling->n; i++) child->children[i + T] = sibling->children[i];
    for (int i = idx + 1; i < parent->n; i++)     parent->entries[i - 1]  = parent->entries[i];
    for (int i = idx + 2; i <= parent->n; i++)    parent->children[i - 1] = parent->children[i];
    child->n = child->n + sibling->n + 1;
    parent->n--;
    sibling->n = 0;
    sibling->leaf = true;
    delete sibling;
    parent->children[parent->n + 1] = nullptr;
}

} // namespace db
