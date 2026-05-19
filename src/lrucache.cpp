#include "lrucache.h"

LRUCache::LRUCache(int max_sz, int _list_pool_size, QMatrix* _matrix): sz(0), max_cnt(0), cnt(0),
list(max_sz, _list_pool_size), matrix(_matrix), indices(nullptr) {
    data = new double[N];
    offsets = new int[max_sz];
}
LRUCache::LRUCache(): sz(0), max_cnt(0), cnt(0), matrix(nullptr), offsets(nullptr), 
indices(nullptr), data(nullptr) {}
void LRUCache::init(int max_sz, int _list_pool_size, QMatrix* _matrix) {
    data = new double[N];
    offsets = new int[max_sz];
    list.init(max_sz, _list_pool_size);
    matrix = _matrix;
    sz = max_cnt = cnt = 0;
}
void LRUCache::reset(int _sz, int* _indices) {
    sz = _sz;
    max_cnt = gmin(N / _sz, _sz);
    cnt = 0;
    memset(offsets, -1, _sz * sizeof(int));
    indices = _indices;
    list.reset(max_cnt, _sz);
}
void LRUCache::get_row(int i, int offset) {
    double* row = data + (long long)sz * offset;
    if(indices == nullptr) {
        for(int j = 0; j < sz; j++)
            row[j] = matrix->get(i, j);
    }
    else {
        for(int j = 0; j < sz; j++)
            row[j] = matrix->get(indices[i], indices[j]);
    }
}
int LRUCache::get_offset(int i) {
    if(offsets[i] != -1) {
        list.upd(i);
        return offsets[i];
    }
    if(cnt < max_cnt) {
        offsets[i] = cnt;
        list.insert(i);
        get_row(i, cnt);
        cnt++;
        return offsets[i];
    }
    int j = list.insert(i);
    assert(j != -1);
    int offset = offsets[j];
    assert(offset != -1);
    offsets[j] = -1, offsets[i] = offset;
    get_row(i, offset);
    return offsets[i];
}
double LRUCache::get(int i, int j) {
    int offset = get_offset(i);
    return *(data + (long long)sz * offset + j);
}
const double* LRUCache::get_ptr(int i) {
    int offset = get_offset(i);
    return data + (long long)sz * offset;
}
double LRUCache::get_qd(int i) {
    int k = i;
    if(indices != nullptr) k = indices[i];
    return matrix->get_qd(k);
}
LRUCache::~LRUCache() {
    delete []data;
    delete []offsets;
}