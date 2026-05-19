#ifndef LRUCACHE_H
#define LRUCACHE_H

#include "common.h"
#include "qmatrix.h"
#include "list.h"

class LRUCache {
private:
    double *data;
    int sz;
    int max_cnt, cnt;
    int *offsets, *indices;
    List list;
    QMatrix* matrix;
public:
    LRUCache(int max_sz, int _list_pool_size, QMatrix* _matrix);
    LRUCache();
    void init(int max_sz, int _list_pool_size, QMatrix* _matrix);
    void reset(int _sz, int* _indices);
    void get_row(int i, int offset);
    int get_offset(int i);
    double get(int i, int j);
    double get_qd(int i);
    const double* get_ptr(int i);
    ~LRUCache();
};

#endif