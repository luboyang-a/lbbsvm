#ifndef QMATRIX_H
#define QMATRIX_H

#include "common.h"

class QMatrix {
public:
    virtual double get(int i, int j) = 0;
    virtual double get(int i, double* b) = 0;
    virtual double get_qd(int i) = 0;
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
    ~VectorRbf();
};

#endif