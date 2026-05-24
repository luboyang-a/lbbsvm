#include "model.h"

QMatrix* SVC::init_matrix(double* _X, int _dim, int _sz, ktype _ty, double sigma) {
    QMatrix* res = nullptr;
    switch(_ty) {
        case LINEAR: res = new VectorLinear(_X, _dim, _sz); break;
        case RBF   : res = new VectorRbf(_X, sigma, _dim, _sz); break;
        case LISTLINEAR: res = new ListLinear(_X, _dim, _sz); break;
        default    : break;
    }
    return res;
}
SVC::SVC(double* _X, int* _y, int _dim, int _sz, ktype _ty, double sigma, double _C, double _eps): 
X(_X), y(_y), dim(_dim), sz(_sz), ty(_ty), matrix(init_matrix(_X, _dim, _sz, _ty, sigma)),
cache(_sz, matrix), solver(_sz, _C, _eps, cache), C(_C), eps(_eps) {}
void SVC::fit(int max_iter) {
    cache.reset(sz, nullptr);
    solver.reset(y, nullptr, sz);
    model.indices = nullptr;
    model.sv_vals = nullptr;
    solver.solve(max_iter, model);
}
void SVC::predict(double* X_test, int* y_pred, int n_test) {
    #pragma omp parallel for if (n_test >= SAMPLE_THRESHOLD)
    for(int i = 0; i < n_test; i++) {
        double* xi = X_test + i * dim;
        double res = 0.0;
        for(int k = 0; k < model.sz; k++) {
            int sv = model.indices[k];
            double kval = matrix->get(sv, xi);
            res += model.sv_vals[k] * kval;
        }
        res -= model.b;
        y_pred[i] = (res >= 0.0) ? 1 : -1;
    }
}
SVC::~SVC() {
    delete matrix;
}

QMatrix* OneVsOneSVC::init_matrix(double* _X, int _dim, int _sz, ktype _ty, double sigma) {
    QMatrix* res = nullptr;
    switch(_ty) {
        case LINEAR: res = new VectorLinear(_X, _dim, _sz); break;
        case RBF   : res = new VectorRbf(_X, sigma, _dim, _sz); break;
        case LISTLINEAR: res = new ListLinear(_X, _dim, _sz); break;
        default    : break;
    }
    return res;
}
OneVsOneSVC::OneVsOneSVC(double* _X, int* _y, int _dim, int _sz, double _C, double _eps, ktype _ty, double sigma):
X(_X), y(_y), dim(_dim), sz(_sz), num_class(0), indices(new int[_sz]), class_idx(nullptr),
C(_C), eps(_eps), ty(_ty), matrix(init_matrix(_X, _dim, _sz, _ty, sigma)), models(nullptr), 
solver(_C, _eps, cache), pos(0) {
    int max_num = 0;
    for(int i = 0; i < _sz; i++) max_num = gmax(max_num, y[i]);
    num_class = max_num + 1;
    assert(num_class >= 2);
    class_idx = new int[num_class];
    int* class_cnts = new int[num_class];
    int* class_pos = new int[num_class];
    memset(class_cnts, 0, num_class * sizeof(int));
    for(int i = 0; i < _sz; i++) class_cnts[y[i]]++;
    int sum = 0;
    for(int i = 0; i < num_class; i++) {
        class_idx[i] = sum;
        sum += class_cnts[i];
    }
    for(int i = 0; i < num_class; i++) class_pos[i] = class_idx[i];
    for(int i = 0; i < _sz; i++) {
        int label = y[i];
        indices[class_pos[label]] = i;
        class_pos[label]++;
    }
    std::sort(class_cnts, class_cnts + num_class, [](const int a, const int b) {return a < b;});
    int max_sz = class_cnts[num_class - 1] + class_cnts[num_class - 2];
    cache.init(max_sz, matrix);
    solver.init(max_sz);
    models = new ResInfo[num_class * (num_class - 1) / 2];
    res_idx = new ResIdx[num_class * (num_class - 1) / 2];
    y_tmp = new int[max_sz];
    indices_tmp = new int[max_sz];
    delete []class_cnts;
    delete []class_pos;
}
void OneVsOneSVC::fit(int max_iter) {
    pos = 0;
    for(int ci = 0; ci < num_class; ci++) {
        int ci_cnt = ((ci == num_class - 1) ? sz : class_idx[ci + 1]) - class_idx[ci];
        memcpy(indices_tmp, indices + class_idx[ci], ci_cnt * sizeof(int));
        cache.reset(ci_cnt);
        for(int k = 0; k < ci_cnt; k++) y_tmp[k] = 1;
        for(int cj = ci + 1; cj < num_class; cj++) {
            int cj_cnt = ((cj == num_class - 1) ? sz : class_idx[cj + 1]) - class_idx[cj];
            int curr_sz = ci_cnt + cj_cnt;
            for(int k = ci_cnt; k < curr_sz; k++) y_tmp[k] = -1;
            memcpy(indices_tmp + ci_cnt, indices + class_idx[cj], cj_cnt * sizeof(int));
            cache.sub_reset(curr_sz, indices_tmp);
            solver.reset(y_tmp, indices_tmp, curr_sz);
            res_idx[pos] = {ci, cj};
            solver.solve(max_iter, models[pos]);
            pos++;
        }
    }
}
void OneVsOneSVC::predict(double* X_test, int* y_pred, int n_test) {
    int pair_cnt = num_class * (num_class - 1) / 2;
    int* votes = new int[n_test * num_class];
    double* scores = new double[n_test * num_class];
    memset(votes, 0, n_test * num_class * sizeof(int));
    memset(scores, 0, n_test * num_class * sizeof(double));
    #pragma omp parallel for if (n_test >= SAMPLE_THRESHOLD)
    for(int i = 0; i < n_test; i++) {
        double* xi = X_test + i * dim;
        int* vote = votes + i * num_class;
        double* score = scores + i * num_class;
        for(int p = 0; p < pair_cnt; p++) {
            int ci = res_idx[p].t, cj = res_idx[p].f;
            double res = 0.0;
            ResInfo& mdl = models[p];
            for(int k = 0; k < mdl.sz; k++) {
                int sv = mdl.indices[k];
                double kval = matrix->get(sv, xi);
                res += mdl.sv_vals[k] * kval;
            }
            res -= mdl.b;
            if(res >= 0.0) {
                vote[ci]++;
                score[ci] += res;
                score[cj] -= res;
            }
            else {
                vote[cj]++;
                score[ci] += res;
                score[cj] -= res;
            }
        }
        int best_class = 0;
        for(int c = 1; c < num_class; c++) {
            if(vote[c] > vote[best_class]) {
                best_class = c;
            } else if(vote[c] == vote[best_class] && score[c] > score[best_class]) {
                best_class = c;
            }
        }
        y_pred[i] = best_class;
    }
    delete []votes;
    delete []scores;
}
OneVsOneSVC::~OneVsOneSVC() {
    delete []y_tmp;
    delete []indices_tmp;
    delete []indices;
    delete []class_idx;
    delete matrix;
    delete []models;
    delete []res_idx;
}