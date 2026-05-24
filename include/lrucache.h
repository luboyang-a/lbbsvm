#ifndef LRUCACHE_H
#define LRUCACHE_H

#include "common.h"
#include "qmatrix.h"

class LRUCache {
private:
    struct lru_node {
        int idx;
        int len;
        double* Qdata;
        lru_node *prev, *next;
    };
    lru_node lru_head;
    lru_node lru_free;
    lru_node** Map;
    lru_node* pool;
    double* data;
    int max_sz;
    int sz;
    int zsz;
    int max_cnt, cnt;
    int* indices;
    QMatrix* matrix;
    void head_insert(lru_node* node);
    lru_node* head_delete();
    void free_insert(lru_node* node);
    lru_node* free_delete();
    void get_row(lru_node* node);
    void upd(lru_node* node);
public:
    LRUCache();
    LRUCache(int max_sz, QMatrix* _matrix);
    void init(int max_sz, QMatrix* _matrix);
    void reset(int _zsz);
    void reset(int _sz, int* _indices);
    void sub_reset(int _sz, int* _indices);
    const double* query(int i, int* _len);
    int ext(int i);
    double get_qd(int i);
    ~LRUCache();
};

#endif