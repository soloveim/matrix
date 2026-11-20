#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

/**
 * @brief Effiziente, cache-freundliche Matrix-Klasse (row-major, contiguous storage).
 *
 * Design-Entscheidungen:
 *  - Ein einzelner zusammenhängender std::vector<double> statt vector<vector<double>>
 *    -> deutlich bessere Cache-Lokalität, weniger Allokationen, keine Pointer-Indirektion.
 *  - Move-Semantik für billige Rückgaben von temporären Matrizen.
 *  - Multiplikation nutzt i-k-j Schleifenreihenfolge (statt i-j-k) für sequentiellen
 *    Speicherzugriff und damit deutlich weniger Cache-Misses.
 *  - Optionale, in Debug-Builds aktive Bounds-Checks (assert-basiert / at()).
 */
class Matrix {
public:
    using value_type = double;

    // ---------- Konstruktoren ----------

    Matrix() noexcept : rows_(0), cols_(0) {}

    Matrix(std::size_t rows, std::size_t cols, value_type init = value_type{})
        : rows_(rows), cols_(cols), data_(rows * cols, init) {}

    Matrix(std::initializer_list<std::initializer_list<value_type>> init) {
        rows_ = init.size();
        cols_ = rows_ > 0 ? init.begin()->size() : 0;
        data_.reserve(rows_ * cols_);
        for (const auto& row : init) {
            if (row.size() != cols_)
                throw std::invalid_argument("Matrix: inkonsistente Zeilenlaenge in initializer_list");
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }

    // Copy / Move sind trivial dank std::vector -> vom Compiler generiert (Rule of Zero)
    Matrix(const Matrix&)            = default;
    Matrix(Matrix&&) noexcept        = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&) noexcept = default;
    ~Matrix()                        = default;

    // ---------- Fabrikfunktionen ----------

    static Matrix identity(std::size_t n) {
        Matrix m(n, n, 0.0);
        for (std::size_t i = 0; i < n; ++i) m(i, i) = 1.0;
        return m;
    }

    static Matrix random(std::size_t rows, std::size_t cols,
                          value_type lo = 0.0, value_type hi = 1.0) {
        Matrix m(rows, cols);
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<value_type> dist(lo, hi);
        for (auto& v : m.data_) v = dist(gen);
        return m;
    }

    // ---------- Zugriff ----------

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    // Ungeprüfter, schneller Zugriff (Hot Path)
    value_type& operator()(std::size_t r, std::size_t c) noexcept {
        return data_[r * cols_ + c];
    }
    const value_type& operator()(std::size_t r, std::size_t c) const noexcept {
        return data_[r * cols_ + c];
    }

    // Geprüfter Zugriff mit Exception
    value_type& at(std::size_t r, std::size_t c) {
        checkBounds(r, c);
        return data_[r * cols_ + c];
    }
    const value_type& at(std::size_t r, std::size_t c) const {
        checkBounds(r, c);
        return data_[r * cols_ + c];
    }

    // Rohzeiger auf zusammenhängenden Speicher (z.B. für BLAS/SIMD-Interop)
    value_type* data() noexcept { return data_.data(); }
    const value_type* data() const noexcept { return data_.data(); }

    // ---------- Grundoperationen ----------

    Matrix& operator+=(const Matrix& rhs) {
        requireSameShape(rhs);
        for (std::size_t i = 0; i < data_.size(); ++i) data_[i] += rhs.data_[i];
        return *this;
    }

    Matrix& operator-=(const Matrix& rhs) {
        requireSameShape(rhs);
        for (std::size_t i = 0; i < data_.size(); ++i) data_[i] -= rhs.data_[i];
        return *this;
    }

    Matrix& operator*=(value_type scalar) noexcept {
        for (auto& v : data_) v *= scalar;
        return *this;
    }

    friend Matrix operator+(Matrix lhs, const Matrix& rhs) { lhs += rhs; return lhs; }
    friend Matrix operator-(Matrix lhs, const Matrix& rhs) { lhs -= rhs; return lhs; }
    friend Matrix operator*(Matrix lhs, value_type scalar)  { lhs *= scalar; return lhs; }
    friend Matrix operator*(value_type scalar, Matrix rhs)  { rhs *= scalar; return rhs; }

    bool operator==(const Matrix& rhs) const noexcept {
        return rows_ == rhs.rows_ && cols_ == rhs.cols_ && data_ == rhs.data_;
    }
    bool operator!=(const Matrix& rhs) const noexcept { return !(*this == rhs); }

    // ---------- Matrixmultiplikation (cache-optimiert, i-k-j Reihenfolge) ----------
    //
    // Klassische i-j-k Multiplikation läuft in der inneren Schleife spaltenweise durch
    // die rechte Matrix -> springt bei jedem Schritt cols_*sizeof(double) Bytes weiter
    // (schlechte Lokalität, viele Cache-Misses).
    //
    // i-k-j dagegen liest und schreibt in der inneren Schleife jeweils zusammenhängende
    // Speicherbereiche (sequentieller Zugriff auf B und C), was auf realer Hardware
    // typischerweise mehrfach (2-8x) schneller ist.
    friend Matrix operator*(const Matrix& lhs, const Matrix& rhs) {
        if (lhs.cols_ != rhs.rows_)
            throw std::invalid_argument("Matrix*: Dimensionen inkompatibel fuer Multiplikation");

        Matrix result(lhs.rows_, rhs.cols_, 0.0);
        const std::size_t n = lhs.rows_, m = lhs.cols_, p = rhs.cols_;

        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t rowA = i * m;
            const std::size_t rowC = i * p;
            for (std::size_t k = 0; k < m; ++k) {
                const value_type a_ik = lhs.data_[rowA + k];
                if (a_ik == 0.0) continue; // kleine Optimierung bei duennbesetzten Matrizen
                const std::size_t rowB = k * p;
                for (std::size_t j = 0; j < p; ++j) {
                    result.data_[rowC + j] += a_ik * rhs.data_[rowB + j];
                }
            }
        }
        return result;
    }

    Matrix& operator*=(const Matrix& rhs) { *this = *this * rhs; return *this; }

    // ---------- Weitere Operationen ----------

    [[nodiscard]] Matrix transpose() const {
        Matrix result(cols_, rows_);
        // Blockweise transponieren für bessere Cache-Nutzung bei groesseren Matrizen
        constexpr std::size_t BLOCK = 32;
        for (std::size_t ii = 0; ii < rows_; ii += BLOCK) {
            for (std::size_t jj = 0; jj < cols_; jj += BLOCK) {
                const std::size_t iMax = std::min(ii + BLOCK, rows_);
                const std::size_t jMax = std::min(jj + BLOCK, cols_);
                for (std::size_t i = ii; i < iMax; ++i)
                    for (std::size_t j = jj; j < jMax; ++j)
                        result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    [[nodiscard]] value_type trace() const {
        if (rows_ != cols_) throw std::invalid_argument("trace(): nur fuer quadratische Matrizen definiert");
        value_type sum = 0.0;
        for (std::size_t i = 0; i < rows_; ++i) sum += (*this)(i, i);
        return sum;
    }

    [[nodiscard]] value_type frobeniusNorm() const {
        value_type sum = 0.0;
        for (auto v : data_) sum += v * v;
        return std::sqrt(sum);
    }

    // Gauss-Jordan mit Partial Pivoting: liefert Inverse, wirft bei Singularitaet
    [[nodiscard]] Matrix inverse() const {
        if (rows_ != cols_) throw std::invalid_argument("inverse(): nur fuer quadratische Matrizen definiert");
        const std::size_t n = rows_;

        Matrix aug(n, 2 * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) aug(i, j) = (*this)(i, j);
            aug(i, n + i) = 1.0;
        }

        for (std::size_t col = 0; col < n; ++col) {
            // Pivotsuche
            std::size_t pivot = col;
            value_type maxVal = std::abs(aug(col, col));
            for (std::size_t r = col + 1; r < n; ++r) {
                value_type v = std::abs(aug(r, col));
                if (v > maxVal) { maxVal = v; pivot = r; }
            }
            if (maxVal < 1e-12)
                throw std::runtime_error("inverse(): Matrix ist singulaer (oder numerisch instabil)");

            if (pivot != col)
                for (std::size_t j = 0; j < 2 * n; ++j)
                    std::swap(aug(col, j), aug(pivot, j));

            const value_type pivVal = aug(col, col);
            for (std::size_t j = 0; j < 2 * n; ++j) aug(col, j) /= pivVal;

            for (std::size_t r = 0; r < n; ++r) {
                if (r == col) continue;
                const value_type factor = aug(r, col);
                if (factor == 0.0) continue;
                for (std::size_t j = 0; j < 2 * n; ++j)
                    aug(r, j) -= factor * aug(col, j);
            }
        }

        Matrix result(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                result(i, j) = aug(i, n + j);
        return result;
    }

    // ---------- Ausgabe ----------

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        os << std::fixed << std::setprecision(4);
        for (std::size_t i = 0; i < m.rows_; ++i) {
            os << "[ ";
            for (std::size_t j = 0; j < m.cols_; ++j)
                os << std::setw(9) << m(i, j) << " ";
            os << "]\n";
        }
        return os;
    }

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<value_type> data_;

    void checkBounds(std::size_t r, std::size_t c) const {
        if (r >= rows_ || c >= cols_)
            throw std::out_of_range("Matrix: Index ausserhalb des gueltigen Bereichs");
    }

    void requireSameShape(const Matrix& rhs) const {
        if (rows_ != rhs.rows_ || cols_ != rhs.cols_)
            throw std::invalid_argument("Matrix: Dimensionen stimmen nicht ueberein");
    }
};
