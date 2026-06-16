#pragma once

/**
 * @file dense_matrix.hpp
 * @brief 轻量稠密矩阵（row-major），避免 Eigen 依赖。
 */

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace koopman_control::detail {

inline float gelu(float x) {
    return 0.5f * x * (1.f + std::erf(x * 0.70710678118f));  // erf(x/sqrt(2)), matches PyTorch default
}

// d/dx GELU(x) = 0.5(1+erf(x/sqrt2)) + x * exp(-x^2/2)/sqrt(2*pi)
inline float geluGrad(float x) {
    constexpr float kInvSqrt2 = 0.70710678118f;
    constexpr float kInvSqrt2Pi = 0.39894228040f;  // 1/sqrt(2*pi)
    return 0.5f * (1.f + std::erf(x * kInvSqrt2)) + x * std::exp(-0.5f * x * x) * kInvSqrt2Pi;
}

// 角度归一化到 (-pi, pi]
inline float wrapAngle(float a) {
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kTwoPi = 2.f * kPi;
    a = std::fmod(a + kPi, kTwoPi);
    if (a <= 0.f) {
        a += kTwoPi;
    }
    return a - kPi;
}

class Matrix {
public:
    Matrix() = default;
    Matrix(int rows, int cols, float init = 0.f) : rows_(rows), cols_(cols), data_(rows * cols, init) {}

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return static_cast<int>(data_.size()); }
    const float* data() const { return data_.data(); }
    float* data() { return data_.data(); }

    float& operator()(int r, int c) { return data_[r * cols_ + c]; }
    float operator()(int r, int c) const { return data_[r * cols_ + c]; }

    static Matrix identity(int n) {
        Matrix m(n, n, 0.f);
        for (int i = 0; i < n; ++i) {
            m(i, i) = 1.f;
        }
        return m;
    }

    static Matrix matmul(const Matrix& a, const Matrix& b) {
        if (a.cols() != b.rows()) {
            throw std::runtime_error("matmul dimension mismatch");
        }
        Matrix out(a.rows(), b.cols(), 0.f);
        for (int i = 0; i < a.rows(); ++i) {
            for (int k = 0; k < a.cols(); ++k) {
                const float aik = a(i, k);
                for (int j = 0; j < b.cols(); ++j) {
                    out(i, j) += aik * b(k, j);
                }
            }
        }
        return out;
    }

    static std::vector<float> matvec(const Matrix& a, const std::vector<float>& x) {
        if (a.cols() != static_cast<int>(x.size())) {
            throw std::runtime_error("matvec dimension mismatch");
        }
        std::vector<float> y(a.rows(), 0.f);
        for (int i = 0; i < a.rows(); ++i) {
            float s = 0.f;
            for (int j = 0; j < a.cols(); ++j) {
                s += a(i, j) * x[j];
            }
            y[i] = s;
        }
        return y;
    }

    static std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("vector add size mismatch");
        }
        std::vector<float> out(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            out[i] = a[i] + b[i];
        }
        return out;
    }

    static float dot(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("dot size mismatch");
        }
        float s = 0.f;
        for (size_t i = 0; i < a.size(); ++i) {
            s += a[i] * b[i];
        }
        return s;
    }

    static Matrix transpose(const Matrix& a) {
        Matrix t(a.cols(), a.rows());
        for (int i = 0; i < a.rows(); ++i) {
            for (int j = 0; j < a.cols(); ++j) {
                t(j, i) = a(i, j);
            }
        }
        return t;
    }

private:
    int rows_{0};
    int cols_{0};
    std::vector<float> data_;
};

}  // namespace koopman_control::detail
