#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>

namespace py = pybind11;
using namespace Eigen;

// Función de entrenamiento para los modelos base (ejemplo: regresión lineal)
MatrixXd train_base_models(const MatrixXd &X, const MatrixXd &y) {
    int n_models = 2;  // Número de modelos base
    int n_samples = X.rows();
    
    MatrixXd base_predictions(n_samples, n_models);

    for (int i = 0; i < n_models; ++i) {
        VectorXd w = (X.transpose() * X).ldlt().solve(X.transpose() * y);  // Regresión lineal simple
        base_predictions.col(i) = X * w;
    }

    return base_predictions;
}

// Entrenamiento del meta-modelo (otra regresión lineal)
VectorXd train_meta_model(const MatrixXd &Z, const MatrixXd &y) {
    return (Z.transpose() * Z).ldlt().solve(Z.transpose() * y);
}

// Predicción con el modelo de stacking
VectorXd stacking_predict(const MatrixXd &X, const MatrixXd &X_meta, const VectorXd &meta_weights) {
    MatrixXd base_predictions = train_base_models(X, X_meta);
    return base_predictions * meta_weights;
}

// Función de entrenamiento principal
std::tuple<py::array_t<double>, py::array_t<double>> stacking_train(
    py::array_t<double> X_np, py::array_t<double> y_np, py::array_t<double> X_test_np) {

    auto X_buf = X_np.request(), y_buf = y_np.request(), X_test_buf = X_test_np.request();
    
    MatrixXd X = Map<MatrixXd>((double*)X_buf.ptr, X_buf.shape[0], X_buf.shape[1]);
    MatrixXd y = Map<MatrixXd>((double*)y_buf.ptr, y_buf.shape[0], y_buf.shape[1]);
    MatrixXd X_test = Map<MatrixXd>((double*)X_test_buf.ptr, X_test_buf.shape[0], X_test_buf.shape[1]);

    // Entrenar modelos base
    MatrixXd Z = train_base_models(X, y);
    
    // Entrenar el meta-modelo
    VectorXd meta_weights = train_meta_model(Z, y);

    // Hacer predicciones
    VectorXd y_pred = stacking_predict(X_test, X, meta_weights);

    // Convertir a numpy
    py::array_t<double> Z_py({Z.rows(), Z.cols()}, Z.data());
    py::array_t<double> y_pred_py({y_pred.size()}, y_pred.data());

    return std::make_tuple(Z_py, y_pred_py);
}

PYBIND11_MODULE(stacking_module, m) {
    m.def("stacking_train", &stacking_train, "Train and predict using Stacking Approach");
}
