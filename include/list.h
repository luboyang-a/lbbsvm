#ifndef LIST_H
#define LIST_H

#include "common.h"

class List {
private:
    struct Node {
        int idx;
        Node *prev, *next;
    };
    Node *head, *tail;
    Node **Map;
    Node* list_pool;
    int Map_size, list_pool_size;
    int max_cnt, cnt, n;
    int list_pool_pos;
public:
    List(int _Map_size, int _list_pool_size);
    List();
    void init(int _Map_size, int _list_pool_size);
    void reset(int _max_cnt, int _n);
    int insert(int i);
    void upd(int i);
    ~List();
};

#endif