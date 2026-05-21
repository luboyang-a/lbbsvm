#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "model.h"
#include <string>

namespace py = pybind11;

class PyOneVsOneSVC {
private:
    OneVsOneSVC* svm_ptr;
    int dim;
    int sz;
public:
    PyOneVsOneSVC(py::array_t<double> X, py::array_t<int> y, int dim, double C = 1.0, 
    double eps = 1e-3, std::string kernel = "linear", double sigma = 1.0): dim(dim), svm_ptr(nullptr) {
        py::buffer_info buf_X = X.request();
        py::buffer_info buf_y = y.request();
        sz = (int)(buf_y.shape[0]); 
        ktype ty = LINEAR;
        if (kernel == "rbf" || kernel == "RBF") {
            ty = RBF;
        }
        else if(kernel == "LISTLINEAR" || kernel == "listlinear") {
            ty = LISTLINEAR;
        }
        double* X_ptr = (double*)(buf_X.ptr);
        int* y_ptr = (int*)(buf_y.ptr);
        svm_ptr = new OneVsOneSVC(X_ptr, y_ptr, dim, sz, C, eps, ty, sigma);
    }
    void fit(int max_iter = 50000) {
        svm_ptr->fit(max_iter);
    }
    py::array_t<int> predict(py::array_t<double> X_test) {
        py::buffer_info buf_X_test = X_test.request();
        int n_test = (int)(buf_X_test.shape[0]); 
        auto y_pred = py::array_t<int>(n_test);
        py::buffer_info buf_y_pred = y_pred.request();
        svm_ptr->predict((double*)(buf_X_test.ptr), (int*)(buf_y_pred.ptr), n_test);
        return y_pred;
    }
    ~PyOneVsOneSVC() {
        if (svm_ptr) {
            delete svm_ptr;
            svm_ptr = nullptr;
        }
    }
};

class PySVC {
private:
    SVC* svm_ptr;
    int dim;
    int sz;
public:
    PySVC(py::array_t<double> X, py::array_t<int> y, int dim, double C = 1.0, double eps = 1e-3, 
    std::string kernel = "linear", double sigma = 1.0): dim(dim), svm_ptr(nullptr) {
        
        py::buffer_info buf_X = X.request();
        py::buffer_info buf_y = y.request();
        sz = (int)(buf_y.shape[0]);
        ktype ty = LINEAR;
        if (kernel == "rbf" || kernel == "RBF") {
            ty = RBF;
        }
        else if(kernel == "LISTLINEAR" || kernel == "listlinear") {
            ty = LISTLINEAR;
        }
        double* X_ptr = (double*)(buf_X.ptr);
        int* y_ptr = (int*)(buf_y.ptr);
        svm_ptr = new SVC(X_ptr, y_ptr, dim, sz, ty, sigma, C, eps);
    }

    void fit(int max_iter = 50000) {
        svm_ptr->fit(max_iter);
    }
    py::array_t<int> predict(py::array_t<double> X_test) {
        py::buffer_info buf_X_test = X_test.request();
        int n_test = (int)(buf_X_test.shape[0]);
        auto y_pred = py::array_t<int>(n_test);
        py::buffer_info buf_y_pred = y_pred.request();
        svm_ptr->predict((double*)(buf_X_test.ptr), (int*)(buf_y_pred.ptr), n_test);
        return y_pred;
    }
    ~PySVC() {
        if (svm_ptr) {
            delete svm_ptr;
            svm_ptr = nullptr;
        }
    }
};

PYBIND11_MODULE(lbbsvm, m) {
    m.doc() = "lbbsvm";
    py::class_<PySVC>(m, "SVC")
        .def(py::init<py::array_t<double>, py::array_t<int>, int, double, double, std::string, double>(),
             py::arg("X"), py::arg("y"), py::arg("dim"), 
             py::arg("C") = 1.0, py::arg("eps") = 1e-3, 
             py::arg("kernel") = "linear", py::arg("sigma") = 1.0)
        .def("fit", &PySVC::fit, py::arg("max_iter") = 50000)
        .def("predict", &PySVC::predict, py::arg("X_test"));
    py::class_<PyOneVsOneSVC>(m, "OneVsOneSVC")
        .def(py::init<py::array_t<double>, py::array_t<int>, int, double, double, std::string, double>(),
             py::arg("X"), py::arg("y"), py::arg("dim"), 
             py::arg("C") = 1.0, py::arg("eps") = 1e-3, 
             py::arg("kernel") = "linear", py::arg("sigma") = 1.0)
        .def("fit", &PyOneVsOneSVC::fit, py::arg("max_iter") = 50000)
        .def("predict", &PyOneVsOneSVC::predict, py::arg("X_test"));
}