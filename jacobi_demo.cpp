#include "Jacobi.hpp"

#include <chrono>
#include <iostream>

// Residuum ||A*V - V*Lambda||_F und Orthogonalitaet ||V^T V - I||_F
static void verify(const Matrix& A, const EigenResult& r) {
    const std::size_t n = A.rows();
    Matrix lambda(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) lambda(i, i) = r.eigenvalues[i];

    const Matrix V = r.eigenvectors;
    const double residual = (A * V - V * lambda).frobeniusNorm();
    const double orthoErr = (V.transpose() * V - Matrix::identity(n)).frobeniusNorm();

    std::cout << "  Sweeps:                 " << r.sweeps << "\n"
              << "  konvergiert:            " << (r.converged ? "ja" : "nein") << "\n"
              << "  ||A*V - V*Lambda||_F:   " << std::scientific << residual << "\n"
              << "  ||V^T*V - I||_F:        " << orthoErr << std::fixed << "\n";
}

int main() {
    std::cout << std::setprecision(6);

    // --- Beispiel 1: Tridiagonalmatrix, Eigenwerte bekannt: 2-sqrt(2), 2, 2+sqrt(2)
    std::cout << "=== Beispiel 1: 3x3 Tridiagonalmatrix ===\n";
    Matrix A{{ 2, -1,  0},
             {-1,  2, -1},
             { 0, -1,  2}};
    EigenResult r = jacobiEigen(A);

    std::cout << "Eigenwerte (erwartet: " << 2 - std::sqrt(2.0) << ", 2, "
              << 2 + std::sqrt(2.0) << "):\n  ";
    for (double ev : r.eigenvalues) std::cout << ev << "  ";
    std::cout << "\nEigenvektoren (Spalten):\n" << r.eigenvectors;
    verify(A, r);

    // --- Beispiel 2: zufaellige symmetrische Matrix
    std::cout << "\n=== Beispiel 2: zufaellige symmetrische 200x200 Matrix ===\n";
    const std::size_t n = 200;
    Matrix R = Matrix::random(n, n, -1.0, 1.0);
    Matrix S = (R + R.transpose()) * 0.5;

    auto t0 = std::chrono::high_resolution_clock::now();
    EigenResult big = jacobiEigen(S);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "  kleinster Eigenwert:    " << big.eigenvalues.front() << "\n"
              << "  groesster Eigenwert:    " << big.eigenvalues.back() << "\n"
              << "  Spur(A) = " << S.trace() << ",  Summe der Eigenwerte = "
              << std::accumulate(big.eigenvalues.begin(), big.eigenvalues.end(), 0.0) << "\n";
    verify(S, big);
    std::cout << "  Laufzeit:               "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    // --- Fehlerfall: nicht symmetrisch
    std::cout << "\n=== Fehlerfall: nicht symmetrische Matrix ===\n";
    try {
        Matrix N{{1, 2}, {3, 4}};
        (void)jacobiEigen(N);
    } catch (const std::exception& e) {
        std::cout << "  Exception: " << e.what() << "\n";
    }

  
    try {
        std::size_t dim = 3;
        Matrix TS( dim,dim);
        for (std::size_t ii = 0; ii  < dim; ++ii)
            for(std::size_t jj = 0; jj < dim; ++jj)
            {
                TS(ii, jj) = (double)( ii + jj );
            }
        double eps = 1e-5;
        
        EigenResult Res = jacobiEigen(TS);
        std::vector<double> vder;
        
        std::size_t J = 1;
        Matrix Tmp = TS;
        Tmp.at(J,J) += eps;
        EigenResult ResEps = jacobiEigen(Tmp);
        for (std::size_t ll = 0; ll < dim; ++ll)
        {   
            double df = (ResEps.eigenvalues[ll] - Res.eigenvalues[ll]) / eps;
            vder.push_back(df);
        }
        double sum = 0;
        for (auto d : vder)
            sum += d; 
        std::cout << " Checksumme: " << sum << "\n";
    }
    catch (const std::exception& e) {
        std::cout << "  Exception: " << e.what() << "\n";
    }
    return 0;
}
