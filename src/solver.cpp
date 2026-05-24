#include "solver.h"

ResInfo::ResInfo(int* _indices, double* _sv_vals, double _b, int _sz): indices(_indices), sv_vals(_sv_vals),
b(_b), sz(_sz) {}
ResInfo::ResInfo(): indices(nullptr), sv_vals(nullptr), b(0.0), sz(0) {}
ResInfo::~ResInfo() {
    if(indices != nullptr) delete []indices;
    if(sv_vals != nullptr) delete []sv_vals;
}

Solver::Solver(int max_sz, double _C, double _eps, LRUCache& _cache): y(nullptr), indices(nullptr),
C(_C), eps(_eps), cache(_cache), unshrink(false) {
    alpha = new double[max_sz];
    G = new double[max_sz];
    alpha_status = new char[max_sz];
    active_set = new int[max_sz];
}
Solver::Solver(double _C, double _eps, LRUCache& _cache): y(nullptr), indices(nullptr), alpha(nullptr), 
sz(0), C(_C), eps(_eps), G(nullptr), alpha_status(nullptr), active_set(nullptr), active_set_size(0),
cache(_cache), unshrink(false) {}
void Solver::reset(int* _y, int* _indices, int _sz) {
    y = _y;
    indices = _indices;
    sz = _sz;
    memset(alpha, 0, _sz * sizeof(double));
    for(int i = 0; i < _sz; i++) {
        active_set[i] = i;
        alpha_status[i] = LOWER_BOUND;
        G[i] = -1.0;
    }
    active_set_size = _sz;
    unshrink = false;
}
void Solver::init(int max_sz) {
    alpha = new double[max_sz];
    G = new double[max_sz];
    alpha_status = new char[max_sz];
    active_set = new int[max_sz];
}
void Solver::update_alpha_status(int i) {
    if(alpha[i] >= C) alpha_status[i] = UPPER_BOUND;
    else if(alpha[i] <= 0.0) alpha_status[i] = LOWER_BOUND;
    else alpha_status[i] = FREE;
}
bool Solver::is_upper_bound(int i) {
    return alpha_status[i] == UPPER_BOUND;
}
bool Solver::is_lower_bound(int i) {
    return alpha_status[i] == LOWER_BOUND;
}
bool Solver::is_free(int i) {
    return alpha_status[i] == FREE;
}
int Solver::select_working_set(int& out_i, int& out_j) {
    out_i = out_j = -1;
    double Gmax = -INF;
    double Gmax2 = -INF;
    int Gmax_idx = -1, Gmin_idx = -1;
    double obj_diff_min = INF;
    for(int k = 0; k < active_set_size; k++) {
        int i = active_set[k];
        if(y[i] == 1) {
            if(!is_upper_bound(i)) {
                if(-G[i] >= Gmax) {
                    Gmax = -G[i];
                    Gmax_idx = i;
                }
            }
        }
        else {
            if(!is_lower_bound(i)) {
                if(G[i] >= Gmax) {
                    Gmax = G[i];
                    Gmax_idx = i;
                }
            }
        }
    }
    out_i = Gmax_idx;
    if(out_i == -1) return 0;
    int _len = 0;
    const double* Qi = cache.query(out_i, &_len);
    for(int k = 0; k < active_set_size; k++) {
        int j = active_set[k];
        if(y[j] == 1) {
            if(!is_lower_bound(j)) {
                double grad_diff = Gmax + G[j];
                if(G[j] >= Gmax2) Gmax2 = G[j];
                if(grad_diff > 0) {
                    if(_len <= j) _len = cache.ext(out_i);
                    double Qij = (double)y[out_i] * (double)y[j] * Qi[j];
                    double quad_coef = cache.get_qd(out_i) + cache.get_qd(j) - 2.0 * (double)y[out_i] * Qij;
                    if(quad_coef <= 0) quad_coef = TAU;
                    double obj_diff = -(grad_diff * grad_diff) / quad_coef;
                    if(obj_diff <= obj_diff_min) {
                        Gmin_idx = j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
        else {
            if(!is_upper_bound(j)) {
                double grad_diff = Gmax - G[j];
                if(-G[j] >= Gmax2) Gmax2 = -G[j];
                if(grad_diff > 0) {
                    if(_len <= j) _len = cache.ext(out_i);
                    double Qij = (double)y[out_i] * (double)y[j] * Qi[j];
                    double quad_coef = cache.get_qd(out_i) + cache.get_qd(j) + 2.0 * (double)y[out_i] * Qij;
                    if(quad_coef <= 0) quad_coef = TAU;
                    double obj_diff = -(grad_diff * grad_diff) / quad_coef;
                    if(obj_diff <= obj_diff_min) {
                        Gmin_idx = j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
    }
    if(Gmax + Gmax2 < eps || Gmin_idx == -1) {
        out_i = out_j = -1;
        return 0;
    }
    out_j = Gmin_idx;
    return 1;
}
double Solver::calculate_rho() {
    double ub = INF, lb = -INF, sum_free = 0.0;
    int nr_free = 0;
    for(int i = 0; i < sz; i++) {
        double yG = (double)y[i] * G[i];
        if(is_upper_bound(i)) {
            if(y[i] == -1) ub = gmin(ub, yG);
            else lb = gmax(lb, yG);
        }
        else if(is_lower_bound(i)) {
            if(y[i] == 1) ub = gmin(ub, yG);
            else lb = gmax(lb, yG);
        }
        else {
            ++nr_free;
            sum_free += yG;
        }
    }
    if(nr_free > 0) return sum_free / nr_free;
    return (ub + lb) / 2.0;
}
bool Solver::be_shrunk(int i, double Gmax1, double Gmax2) {
    if(is_upper_bound(i)) {
        if(y[i] == 1) return -G[i] > Gmax1;
        else return -G[i] > Gmax2;
    }
    else if(is_lower_bound(i)) {
        if(y[i] == 1) return G[i] > Gmax2;
        else return G[i] > Gmax1;
    }
    else return false;
}
void Solver::do_shrinking() {
    int i = 0;
    double Gmax1 = -INF, Gmax2 = -INF;
    for(int k = 0; k < active_set_size; k++) {
        i = active_set[k];
        if(y[i] == 1) {
            if(!is_upper_bound(i)) {
                if(-G[i] >= Gmax1) Gmax1 = -G[i];
            }
            if(!is_lower_bound(i)) {
                if(G[i] >= Gmax2) Gmax2 = G[i];
            }
        }
        else {
            if(!is_upper_bound(i)) {
                if(-G[i] >= Gmax2) Gmax2 = -G[i];
            }
            if(!is_lower_bound(i)) {
                if(G[i] >= Gmax1) Gmax1 = G[i];
            }
        }
    }
    if(Gmax1 + Gmax2 <= eps * 5 && !unshrink) {
        unshrink = true;
        un_shrinking();
    }
    int s = 0;
    while(s < active_set_size) {
        i = active_set[s];
        if(be_shrunk(i, Gmax1, Gmax2)) {
            --active_set_size;
            int tmp = active_set[active_set_size];
            active_set[active_set_size] = active_set[s];
            active_set[s] = tmp;
        }
        else s++;
    }
}
void Solver::un_shrinking() {
    active_set_size = sz;
}
void Solver::solve(int max_iter, ResInfo& model) {
    int iter = 0, shrink_counter = 0;
    int _leni = 0, _lenj = 0;
    while(iter < max_iter) {
        _leni = _lenj = 0;
        if(shrink_counter >= gmin(500, sz)) {
            do_shrinking();
            shrink_counter = 0;
            printf("ACTIVATE_SIZE: %d\n", active_set_size);
        }
        int i = -1, j = -1;
        if(select_working_set(i, j) == 0) {
            if(active_set_size < sz) {
                un_shrinking();
                continue;
            }
            else break;
        }
        double old_alpha_i = alpha[i];
        double old_alpha_j = alpha[j];
        double yi = (double)y[i], yj = (double)y[j];
        const double* Qi = cache.query(i, &_leni);
        if(_leni <= j) _leni = cache.ext(i);
        double Qij = yi * yj * Qi[j];
        if(y[i] != y[j]) {
            double quad_coef = cache.get_qd(i) + cache.get_qd(j) + 2.0 * Qij;
            if(quad_coef <= 0) quad_coef = TAU;
            double delta = (-G[i] - G[j]) / quad_coef;
            double diff = alpha[i] - alpha[j];
            alpha[i] += delta;
            alpha[j] += delta;
            if(diff > 0) {
                if(alpha[j] < 0) {
                    alpha[j] = 0.0;
                    alpha[i] = diff;
                }
            }
            else {
                if(alpha[i] < 0) {
                    alpha[i] = 0.0;
                    alpha[j] = -diff;
                }
            }
            if(diff > 0) {
                if(alpha[i] > C) {
                    alpha[i] = C;
                    alpha[j] = C - diff;
                }
            }
            else {
                if(alpha[j] > C) {
                    alpha[j] = C;
                    alpha[i] = C + diff;
                }
            }
        }
        else {
            double quad_coef = cache.get_qd(i) + cache.get_qd(j) - 2.0 * Qij;
            if(quad_coef <= 0) quad_coef = TAU;
            double delta = (G[i] - G[j]) / quad_coef;
            double sum = alpha[i] + alpha[j];
            alpha[i] -= delta;
            alpha[j] += delta;
            if(sum > C) {
                if(alpha[i] > C) {
                    alpha[i] = C;
                    alpha[j] = sum - C;
                }
            }
            else {
                if(alpha[j] < 0) {
                    alpha[j] = 0.0;
                    alpha[i] = sum;
                }
            }
            if(sum > C) {
                if(alpha[j] > C) {
                    alpha[j] = C;
                    alpha[i] = sum - C;
                }
            }
            else {
                if(alpha[i] < 0) {
                    alpha[i] = 0.0;
                    alpha[j] = sum;
                }
            }
        }
        double delta_alpha_i = alpha[i] - old_alpha_i;
        double delta_alpha_j = alpha[j] - old_alpha_j;
        const double* Qj = cache.query(j, &_lenj);
        if(_lenj < sz) _lenj = cache.ext(j);
        if(_leni < sz) _leni = cache.ext(i);
        __m256d v_yi = _mm256_set1_pd((double)y[i]);
        __m256d v_yj = _mm256_set1_pd((double)y[j]);
        __m256d v_dalpha_i = _mm256_set1_pd(delta_alpha_i);
        __m256d v_dalpha_j = _mm256_set1_pd(delta_alpha_j);
        int k = 0;
        for(; k + 4 <= sz; k += 4) {
            __m128i v_yk_int = _mm_loadu_si128((__m128i*)(y + k));
            __m256d v_yk = _mm256_cvtepi32_pd(v_yk_int);
            __m256d v_Qi = _mm256_loadu_pd(Qi + k);
            __m256d v_Qj = _mm256_loadu_pd(Qj + k);
            __m256d v_G  = _mm256_loadu_pd(G + k);
            __m256d v_term_i = _mm256_mul_pd(_mm256_mul_pd(v_yi, v_yk), _mm256_mul_pd(v_Qi, v_dalpha_i));
            __m256d v_term_j = _mm256_mul_pd(_mm256_mul_pd(v_yj, v_yk), _mm256_mul_pd(v_Qj, v_dalpha_j));
            v_G = _mm256_add_pd(v_G, _mm256_add_pd(v_term_i, v_term_j));
            _mm256_storeu_pd(G + k, v_G);
        }
        for(; k < sz; k++) {
            G[k] += (double)y[i] * (double)y[k] * Qi[k] * delta_alpha_i
                + (double)y[j] * (double)y[k] * Qj[k] * delta_alpha_j;
        }
        update_alpha_status(i);
        update_alpha_status(j);
        iter++;
        shrink_counter++;
    }
    printf("Iter: %d\n", iter);
    if(active_set_size < sz) un_shrinking();
    int sv_cnt = 0;
    for(int t = 0; t < sz; t++)
        if(alpha[t] > 0.0) sv_cnt++;
    if(model.indices != nullptr) delete [](model.indices);
    if(model.sv_vals != nullptr) delete [](model.sv_vals);
    int* _indices = new int[sv_cnt];
    double* _sv_vals = new double[sv_cnt];
    int pos = 0;
    for(int t = 0; t < sz; t++) {
        if(alpha[t] > 0.0) {
            _indices[pos] = (indices != nullptr) ? indices[t] : t;
            _sv_vals[pos] = alpha[t] * (double)y[t];
            pos++;
        }
    }
    model.b = calculate_rho();
    model.indices = _indices;
    model.sv_vals = _sv_vals;
    model.sz = sv_cnt;
}
Solver::~Solver() {
    delete []alpha;
    delete []alpha_status;
    delete []G;
    delete []active_set;
}
