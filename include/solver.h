#ifndef SOLVER_H
#define SOLVER_H

#include "common.h"
#include "lrucache.h"

class ResInfo {
public:
    int* indices;
    double* sv_vals;
    double b;
    int sz;
    ResInfo();
    ResInfo(int* _indices, double* _sv_vals, double _b, int _sz);
    ~ResInfo();
};

class Solver {
private:
    enum Status {
            LOWER_BOUND = 0,
            UPPER_BOUND = 1,
            FREE = 2
    };
    int* y;
    int* indices;
    double* alpha;
    int sz;
    double C, eps;
    double* G;
    char* alpha_status;
    int* active_set;
    int active_set_size;
    LRUCache& cache;
    bool unshrink;
    void update_alpha_status(int i);
    bool is_upper_bound(int i);
    bool is_lower_bound(int i);
    bool is_free(int i);
    int select_working_set(int& out_i, int& out_j);
    double calculate_rho();
    void do_shrinking();
    bool be_shrunk(int i, double Gmax1, double Gmax2);
    void un_shrinking();
public:
    Solver(int max_sz, double _C, double _eps, LRUCache& _cache);
    Solver(double _C, double _eps, LRUCache& _cache);
    void init(int max_sz);
    void reset(int* _y, int* _indices, int _sz);
    void solve(int max_iter, ResInfo& model);
    ~Solver();
};

#endif