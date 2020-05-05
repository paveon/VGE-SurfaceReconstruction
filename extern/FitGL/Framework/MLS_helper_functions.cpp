
#include <cmath>
#include <eigen3/Eigen/Dense>

inline float _f_2d_3(Eigen::MatrixXd coeffs, float x_1, float x_2) {
    return
            coeffs(0, 0) +
            coeffs(1, 0) * x_1 +
            coeffs(2, 0) * x_2 +
            coeffs(3, 0) * x_1 * x_2 +
            coeffs(4, 0) * pow(x_1, 2) +
            coeffs(5, 0) * pow(x_2, 2) +
            coeffs(6, 0) * pow(x_1, 2) * pow(x_2, 2) +
            coeffs(7, 0) * pow(x_1, 3) +
            coeffs(8, 0) * pow(x_2, 3);
};

inline float _f_2d_2(Eigen::MatrixXd coeffs, float x_1, float x_2) {
    return
            coeffs(0, 0) +
            coeffs(1, 0) * x_1 +
            coeffs(2, 0) * x_2 +
            coeffs(3, 0) * x_1 * x_2 +
            coeffs(4, 0) * pow(x_1, 2) +
            coeffs(5, 0) * pow(x_2, 2);
};

inline float _f_2d_1(Eigen::MatrixXd coeffs, float x_1, float x_2) {
    return
            coeffs(0, 0) +
            coeffs(1, 0) * x_1 +
            coeffs(2, 0) * x_2;
};

void mat2d_3(Eigen::MatrixXd &mat, int r, const Eigen::VectorXd &x_1, const Eigen::VectorXd &x_2) {
    mat.row(r) <<
               1,
            x_1[r],
            x_2[r],
            x_1[r] * x_2[r],
            pow(x_1[r], 2),
            pow(x_2[r], 2),
            pow(x_1[r], 2) * pow(x_2[r], 2),
            pow(x_1[r], 3),
            pow(x_2[r], 3);
};

void mat2d_2(Eigen::MatrixXd &mat, int r, const Eigen::VectorXd &x_1, const Eigen::VectorXd &x_2) {
    mat.row(r) <<
               1,
            x_1[r],
            x_2[r],
            x_1[r] * x_2[r],
            pow(x_1[r], 2),
            pow(x_2[r], 2);
};

void mat2d_1(Eigen::MatrixXd &mat, int r, const Eigen::VectorXd &x_1, const Eigen::VectorXd &x_2) {
    mat.row(r) <<
               1,
            x_1[r],
            x_2[r];
};
