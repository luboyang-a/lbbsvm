#ifndef QMATRIX_H
#define QMATRIX_H

#include "common.h"

class QMatrix {
protected:
    struct svm_node {
        int idx;
        double value;
    };
public:
    virtual double get(int i, int j) = 0;
    virtual double get(int i, double* b) = 0;
    virtual double get_qd(int i) = 0;
    virtual double get(int i, svm_node* b) = 0;
    virtual ~QMatrix();
};

class VectorLinear : public QMatrix {
private:
    double* X;
    int dim;
    int sz;
    double* QD;
public:
    VectorLinear(double* X, int dim, int sz);
    double get(int i, int j) override;
    double get(int i, double* b) override;
    double get_qd(int i) override;
    double get(int i, svm_node* b) override;
    ~VectorLinear();
};

class VectorRbf : public QMatrix {
private:
    double* X;
    double sigma;
    int dim;
    int sz;
    double* QD;
    double gamma;
public:
    VectorRbf(double* X, double sigma, int dim, int sz);
    double get(int i, int j) override;
    double get(int i, double* b) override;
    double get_qd(int i) override;
    double get(int i, svm_node* b) override;
    ~VectorRbf();
};

class ListLinear : public QMatrix {
private:
    svm_node** X_list;
    int dim;
    int sz;
    double* QD;
    int* sample_cnt;
public:
    ListLinear(double* X, int dim, int sz);
    double get(int i, int j) override;
    double get(int i, double* b) override;
    double get_qd(int i) override;
    double get(int i, svm_node* b) override;
    int get_nxt_ptr(int l, int r, int t, svm_node* h);
    ~ListLinear();
};

#endif