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
double VectorLinear::get(int i, svm_node* b) {
    return 0.0;
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
double VectorRbf::get(int i, svm_node* b) {
    return 0.0;
}
VectorRbf::~VectorRbf() {
    delete []QD;
}

ListLinear::ListLinear(double* X, int dim, int sz): X_list(nullptr), dim(dim), sz(sz),
QD(nullptr), sample_cnt(nullptr) {
    X_list = new svm_node*[sz];
    QD = new double[sz];
    sample_cnt = new int[sz];
    int svm_cnt = 0, pos = 0;
    svm_node* head = nullptr;
    for(int i = 0; i < sz; i++) {
        svm_cnt = 0;
        double* xi = X + i * dim;
        for(int j = 0; j < dim; j++)
            if(fabs(xi[j]) > TAU) svm_cnt++;
        head = new svm_node[svm_cnt + 1];
        sample_cnt[i] = svm_cnt;
        pos = 0;
        QD[i] = 0.0;
        for(int j = 0; j < dim; j++) {
            if(fabs(xi[j]) > TAU) {
                QD[i] += xi[j] * xi[j];
                head[pos].idx = j;
                head[pos].value = xi[j];
                pos++;
            }
        }
        head[svm_cnt].idx = -1, head[svm_cnt].value = 0.0;
        X_list[i] = head;
    }
}
int ListLinear::get_nxt_ptr(int l, int r, int t, svm_node* h) {
    int lp = l, rp = r;
    int res = -1, mid = -1;
    while(lp <= rp) {
        mid = (lp + rp) >> 1;
        if(h[mid].idx == t) {
            res = mid; break;
        }
        else if(h[mid].idx < t) lp = mid + 1;
        else rp = mid - 1;
    }
    return (res != -1) ?  res : lp;
}
double ListLinear::get(int i, int j) {
    svm_node *a = X_list[i], *b = X_list[j];
    int ap = 0, bp = 0;
    double res = 0.0;
    while(a[ap].idx != -1 && b[bp].idx != -1) {
        if(a[ap].idx == b[bp].idx) {
            res += a[ap].value * b[bp].value;
            ap++; bp++;
        }
        else if(a[ap].idx < b[bp].idx) {
            if (b[bp].idx - a[ap].idx < SKIP_THRESHOLD) {
                ap++;
            }
            else {
                ap = get_nxt_ptr(ap + 1, sample_cnt[i] - 1, b[bp].idx, a);
            }
        }
        else {
            if (a[ap].idx - b[bp].idx < SKIP_THRESHOLD) {
                bp++;
            }
            else {
                bp = get_nxt_ptr(bp + 1, sample_cnt[j] - 1, a[ap].idx, b);
            }
        }
    }
    return res;
}
double ListLinear::get(int i, double* b) {
    svm_node* a = X_list[i];
    int j = 0;
    double res = 0.0;
    while(a[j].idx != -1) {
        res += a[j].value * b[a[j].idx];
        j++;
    }
    return res;
}
double ListLinear::get(int i, svm_node* b) {
    return 0.0;
}
double ListLinear::get_qd(int i) {
    return QD[i];
}
ListLinear::~ListLinear() {
    delete []QD;
    delete []sample_cnt;
    for(int i = 0; i < sz; i++) {
        delete [](X_list[i]);
    }
    delete []X_list;
}