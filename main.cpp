#include "Matrix.hpp"
#include <chrono>
#include <iostream>

int main() {
    std::cout << "=== Grundoperationen ===\n";
    Matrix A{{1, 2, 3},
             {4, 5, 6}};
    Matrix B{{7,  8},
             {9,  10},
             {11, 12}};

    std::cout << "A =\n" << A;
    std::cout << "B =\n" << B;

    Matrix C = A * B;
    std::cout << "A * B =\n" << C;

    std::cout << "A^T =\n" << A.transpose();

    Matrix I = Matrix::identity(3);
    std::cout << "Identity(3) =\n" << I;

    Matrix invTest{{4, 7}, {2, 6}};
    Matrix inv = invTest.inverse();
    std::cout << "invTest =\n" << invTest;
    std::cout << "invTest^-1 =\n" << inv;
    std::cout << "invTest * invTest^-1 (sollte Identitaet sein) =\n" << invTest * inv;

    std::cout << "Trace(Identity(3)) = " << I.trace() << "\n";
    std::cout << "Frobenius-Norm(A) = " << A.frobeniusNorm() << "\n\n";

    std::cout << "=== Performance-Benchmark: Multiplikation ===\n";
    const std::size_t n = 400;
    Matrix M1 = Matrix::random(n, n, -1.0, 1.0);
    Matrix M2 = Matrix::random(n, n, -1.0, 1.0);

    auto start = std::chrono::high_resolution_clock::now();
    Matrix M3 = M1 * M2;
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << n << "x" << n << " Multiplikation: " << elapsed.count() << " ms\n";
    std::cout << "Ergebnis-Norm (Sanity-Check): " << M3.frobeniusNorm() << "\n";

    return 0;
}
