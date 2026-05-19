#ifndef MODEL_H
#define MODEL_H

#include "common.h"
#include "solver.h"

class SVC {
private:
    double* X;
    int* y;
    int dim, sz;
    ktype ty;
    QMatrix* matrix;
    LRUCache cache;
    Solver solver;
    double C, eps;
    ResInfo model;
    QMatrix* init_matrix(double* _X, int _dim, int _sz, ktype _ty, double sigma);
public:
    SVC(double* _X, int* _y, int _dim, int _sz, ktype _ty, double sigma, double _C, double _eps);
    void fit(int max_iter);
    void predict(double* X_test, int* y_pred, int n_test);
    ~SVC();
};

class OneVsOneSVC {
private:
    double* X;
    int* y;
    int* y_tmp;
    int* indices_tmp;
    int dim, sz;
    int num_class;
    int* indices;
    int* class_idx;
    double C, eps;
    ktype ty;
    QMatrix* matrix;
    LRUCache cache;
    ResInfo* models;
    Solver solver;
    QMatrix* init_matrix(double* _X, int _dim, int _sz, ktype _ty, double sigma);
    struct ResIdx {int t, f;};
    ResIdx* res_idx;
    int pos;
public:
    OneVsOneSVC(double* _X, int* _y, int _dim, int _sz, double _C, double _eps, ktype _ty, double sigma);
    void fit(int max_iter);
    void predict(double* X_test, int* y_pred, int n_test);
    ~OneVsOneSVC();
};

#endif