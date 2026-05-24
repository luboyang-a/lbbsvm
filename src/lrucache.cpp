#include "lrucache.h"

LRUCache::LRUCache(): lru_head({-1, 0, nullptr, nullptr, nullptr}), lru_free({-1, 0, nullptr, nullptr, nullptr}),
Map(nullptr), pool(nullptr),data(new double[N]), max_sz(0), sz(0), zsz(0), max_cnt(0), cnt(0), 
indices(nullptr), matrix(nullptr) {}
LRUCache::LRUCache(int max_sz, QMatrix* _matrix): lru_head({-1, 0, nullptr, nullptr, nullptr}),
lru_free({-1, 0, nullptr, nullptr, nullptr}), Map(nullptr), pool(nullptr), data(new double[N]), 
max_sz(max_sz), sz(0), zsz(0), max_cnt(0), cnt(0), indices(nullptr), matrix(_matrix) {
    max_cnt = gmin(max_sz, N / max_sz);
    Map = new lru_node*[max_sz];
    pool = new lru_node[max_cnt];
    lru_head.next = &lru_head;
    lru_head.prev = &lru_head;
    lru_free.next = &lru_free;
    lru_free.prev = &lru_free;
}
void LRUCache::init(int max_sz, QMatrix* _matrix) {
    max_cnt = gmin(max_sz, N / max_sz);
    this->max_sz = max_sz;
    Map = new lru_node*[max_sz];
    pool = new lru_node[max_cnt];
    lru_head.next = &lru_head;
    lru_head.prev = &lru_head;
    lru_free.next = &lru_free;
    lru_free.prev = &lru_free;
    matrix = _matrix;
}
void LRUCache::get_row(lru_node* node) {
    int i = node->idx;
    int l = node->len;
    double* row = node->Qdata;
    int rest = sz - l;
    assert(rest >= 0);
    if(indices == nullptr) {
        #pragma omp parallel for if (rest >= SAMPLE_THRESHOLD)
        for(int j = l; j < sz; j++)
            row[j] = matrix->get(i, j);
    }
    else {
        #pragma omp parallel for if (rest >= SAMPLE_THRESHOLD)
        for(int j = l; j < sz; j++)
            row[j] = matrix->get(indices[i], indices[j]);
    }
}
void LRUCache::free_insert(lru_node* node) {
    node->idx = -1; 
    node->len = 0;
    node->prev = lru_free.prev;
    (lru_free.prev)->next = node;
    node->next = &lru_free;
    lru_free.prev = node;
}
LRUCache::lru_node* LRUCache::free_delete() {
    lru_node* res = lru_free.next;
    lru_node* _next = res->next;
    lru_free.next = _next;
    _next->prev = &lru_free;
    return res;
}
void LRUCache::head_insert(lru_node* node) {
    node->next = lru_head.next;
    (lru_head.next)->prev = node;
    lru_head.next = node;
    node->prev = &lru_head;
}
LRUCache::lru_node* LRUCache::head_delete() {
    lru_node* res = lru_head.prev;
    lru_node* _prev = res->prev;
    _prev->next = &lru_head;
    lru_head.prev = _prev;
    return res;
}
void LRUCache::upd(lru_node* node) {
    if(lru_head.next == node) return;
    node->prev->next = node->next;
    node->next->prev = node->prev;
    node->next = lru_head.next;
    (lru_head.next)->prev = node;
    lru_head.next = node;
    node->prev = &lru_head;
}
const double* LRUCache::query(int i, int* _len) {
    if(Map[i] != nullptr) {
        upd(Map[i]);
        *_len = Map[i]->len;
        return Map[i]->Qdata;
    }
    if(lru_free.next != &lru_free) {
        lru_node* node = free_delete();
        node->idx = i;
        Map[i] = node;
        *_len = node->len = 0;
        head_insert(node);
        return node->Qdata;
    }
    else if(cnt < max_cnt) {
        double* qdata = data + cnt * max_sz;
        lru_node* node = pool + cnt;
        cnt++;
        node->idx = i;
        Map[i] = node;
        *_len = node->len = 0;
        head_insert(node);
        node->Qdata = qdata;
        return qdata;
    }
    lru_node* node = head_delete();
    Map[(node->idx)] = nullptr;
    node->idx = i;
    Map[i] = node;
    *_len = node->len = 0;
    head_insert(node);
    return node->Qdata;
}
int LRUCache::ext(int i) {
    lru_node* node = Map[i];
    assert(node != nullptr);
    get_row(node);
    node->len = sz;
    return sz;
}
void LRUCache::reset(int _zsz) {
    sz = 0;
    zsz = _zsz;
    indices = nullptr;
    lru_node* head = &lru_head;
    lru_node* node = nullptr;
    while(lru_head.next != head) {
        node = head_delete();
        free_insert(node);
    }
}
void LRUCache::reset(int _sz, int* _indices) {
    sz = _sz;
    zsz = 0;
    indices = _indices;
    memset(Map, 0, _sz * sizeof(lru_node*));
}
void LRUCache::sub_reset(int _sz, int* _indices) {
    sz = _sz;
    indices = _indices;
    memset(Map, 0, _sz * sizeof(lru_node*));
    lru_node *node = lru_head.next, *tmp = nullptr, *head = &lru_head;
    while(node != head) {
        if(node->idx < zsz) {
            Map[(node->idx)] = node;
            node->len = zsz;
            node = node->next;
        }
        else {
            tmp = node;
            node = node->next;
            tmp->prev->next = tmp->next;
            tmp->next->prev = tmp->prev;
            free_insert(tmp);
        }
    }
}
double LRUCache::get_qd(int i) {
    int k = i;
    if(indices != nullptr) k = indices[i];
    return matrix->get_qd(k);
}
LRUCache::~LRUCache() {
    delete []Map;
    delete []pool;
    delete []data;
}