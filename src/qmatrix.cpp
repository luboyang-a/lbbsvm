#include "qmatrix.h"

QMatrix::~QMatrix() = default;

VectorLinear::VectorLinear(double* X, int dim, int sz) : X(X), dim(dim), sz(sz) {
    QD = new double[sz];
    double *a;
    for(int i = 0; i < sz; i++) {
        a = X + i * dim;
        QD[i] = dot_avx2(a, a, dim);
    }
}
double VectorLinear::get(int i, int j) {
    double* a = X + i * dim;
    double* b = X + j * dim;
    double res = dot_avx2(a, b, dim);
    return res;
}
double VectorLinear::get(int i, double* b) {
    double* a = X + i * dim;
    double res = dot_avx2(a, b, dim);
    return res;
}
double VectorLinear::get_qd(int i) {
    return QD[i];
}
VectorLinear::~VectorLinear() {
    delete []QD;
}

VectorRbf::VectorRbf(double* X, double sigma, int dim, int sz): X(X), sigma(sigma), dim(dim),sz(sz), gamma(1.0 / (2.0 * sigma * sigma)) {
    QD = new double[sz];
    double* a;
    for(int i = 0; i < sz; i++) {
        a = X + i * dim;
        QD[i] = dot_avx2(a, a, dim);
    }
}
double VectorRbf::get(int i, int j) {
    double* a = X + i * dim;
    double* b = X + j * dim;
    double sum = QD[i] + QD[j] - 2.0 * dot_avx2(a, b, dim);
    return std::exp(-sum * gamma);
}
double VectorRbf::get(int i, double* b) {
    double* a = X + i * dim;
    double qdb = dot_avx2(b, b, dim);
    double sum = QD[i] + qdb - 2.0 * dot_avx2(a, b, dim);
    return std::exp(-sum * gamma);
}
double VectorRbf::get_qd(int i) {
    return 1.0;
}
VectorRbf::~VectorRbf() {
    delete []QD;
}