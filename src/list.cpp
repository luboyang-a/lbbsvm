#include "list.h"

List::List(int _Map_size, int _list_pool_size): head(nullptr), tail(nullptr), Map_size(_Map_size),
list_pool_size(_list_pool_size), max_cnt(0), cnt(0), n(0), list_pool_pos(2) {
    Map = new Node*[_Map_size];
    list_pool = new Node[_list_pool_size + 2];
    head = list_pool, tail = list_pool + 1;
    head->next = tail, tail->prev = head;
    head->prev = tail->next = nullptr;
}
List::List(): head(nullptr), tail(nullptr), Map(nullptr), list_pool(nullptr), Map_size(0),
list_pool_size(0), max_cnt(0), cnt(0), n(0), list_pool_pos(0) {}
void List::init(int _Map_size, int _list_pool_size) {
    Map = new Node*[_Map_size];
    list_pool = new Node[_list_pool_size + 2];
    head = list_pool, tail = list_pool + 1;
    head->next = tail, tail->prev = head;
    head->prev = tail->next = nullptr;
    list_pool_pos = 2;
}
void List::reset(int _max_cnt, int _n) {
    head->next = tail, tail->prev = head;
    head->prev = tail->next = nullptr;
    memset(Map, 0, _n * sizeof(Node*));
    max_cnt = _max_cnt;
    n = _n;
    cnt = 0;
    list_pool_pos = 2;
}
int List::insert(int i) {
    if(cnt < max_cnt) {
        Node* node = list_pool + list_pool_pos;
        list_pool_pos++;
        node->idx = i;
        node->next = head->next, head->next->prev = node;
        node->prev = head, head->next = node;
        cnt++;
        Map[i] = node;
        return -1;
    }
    Node* node = tail->prev;
    int j = node->idx;
    Map[j] = nullptr;
    node->prev->next = tail, tail->prev = node->prev;
    node->idx = i;
    node->next = head->next, head->next->prev = node;
    node->prev = head, head->next = node;
    Map[i] = node;
    return j;
}
void List::upd(int i) {
    assert(Map[i] != nullptr);
    Node* node = Map[i];
    if(head->next == node) return;
    node->prev->next = node->next, node->next->prev = node->prev;
    node->next = head->next, head->next->prev = node;
    head->next = node, node->prev = head;
}
List::~List() {
    delete []Map;
    delete []list_pool;
}