#pragma once

#include "Matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

/**
 * @brief Ergebnis der Eigenwertzerlegung A = V * diag(lambda) * V^T
 */
struct EigenResult {
    std::vector<double> eigenvalues;  ///< aufsteigend sortiert
    Matrix eigenvectors;              ///< Spalte k gehoert zu eigenvalues[k] (orthonormal)
    std::size_t sweeps = 0;           ///< Anzahl durchgefuehrter Sweeps
    double offDiagonalNorm = 0.0;     ///< Rest-Norm der Nebendiagonale am Ende
    bool converged = false;           ///< Konvergenzkriterium erreicht?
};

/**
 * @brief Zyklisches Jacobi-Verfahren fuer symmetrische Matrizen.
 *
 * Idee: Durch eine Folge ebener Rotationen (Givens-Rotationen) J(p,q,theta) wird
 * jeweils ein Nebendiagonalelement a_pq annulliert:   A <- J^T A J.
 * Die Summe der Quadrate der Nebendiagonalelemente nimmt dabei streng monoton ab,
 * das Verfahren konvergiert (asymptotisch quadratisch) gegen eine Diagonalmatrix.
 * Das Produkt aller Rotationen ergibt die Eigenvektormatrix V.
 *
 * Eigenschaften:
 *  - sehr genau (auch kleine Eigenwerte mit hoher relativer Genauigkeit),
 *  - liefert garantiert orthonormale Eigenvektoren,
 *  - Aufwand O(n^3) pro Sweep, typischerweise 5-10 Sweeps.
 *
 * @param A            symmetrische Matrix
 * @param tol          relative Toleranz: Abbruch wenn off(A) <= tol * ||A||_F
 * @param maxSweeps    Sicherheitsgrenze fuer die Anzahl der Sweeps
 * @param symmetryTol  erlaubte relative Asymmetrie von A
 */
[[nodiscard]] inline EigenResult jacobiEigen(const Matrix& A,
                                             double tol = 1e-12,
                                             std::size_t maxSweeps = 100,
                                             double symmetryTol = 1e-10) {
    const std::size_t n = A.rows();
    if (n != A.cols())
        throw std::invalid_argument("jacobiEigen: Matrix muss quadratisch sein");

    const double normA = A.frobeniusNorm();

    // Symmetrie pruefen
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            if (std::abs(A(i, j) - A(j, i)) > symmetryTol * std::max(normA, 1.0))
                throw std::invalid_argument("jacobiEigen: Matrix ist nicht symmetrisch");

    Matrix a = A;                    // Arbeitskopie (wird zur Diagonalmatrix)
    Matrix v = Matrix::identity(n);  // akkumuliert die Rotationen

    // Symmetrie exakt erzwingen (Rundungsfehler aus der Eingabe eliminieren)
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            a(i, j) = a(j, i) = 0.5 * (a(i, j) + a(j, i));

    auto offNorm = [&]() {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j) s += a(i, j) * a(i, j);
        return std::sqrt(2.0 * s);
    };

    EigenResult res;
    const double threshold = tol * (normA > 0.0 ? normA : 1.0);
    double off = offNorm();

    std::size_t sweep = 0;
    while (off > threshold && sweep < maxSweeps) {
        for (std::size_t p = 0; p + 1 < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const double apq = a(p, q);
                if (apq == 0.0) continue;

                // Winkel berechnen (numerisch stabile Variante nach Rutishauser /
                // Golub & Van Loan): t = tan(theta) ist die kleinere Wurzel von
                // t^2 + 2*t*theta - 1 = 0
                const double app = a(p, p), aqq = a(q, q);
                const double tau = (aqq - app) / (2.0 * apq);
                const double t = (tau >= 0.0 ? 1.0 : -1.0) /
                                 (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = t * c;

                // Diagonale und annulliertes Element
                a(p, p) = app - t * apq;
                a(q, q) = aqq + t * apq;
                a(p, q) = a(q, p) = 0.0;

                // Restliche Zeilen/Spalten p und q rotieren (Symmetrie ausnutzen)
                for (std::size_t r = 0; r < n; ++r) {
                    if (r == p || r == q) continue;
                    const double arp = a(r, p), arq = a(r, q);
                    const double nrp = c * arp - s * arq;
                    const double nrq = s * arp + c * arq;
                    a(r, p) = a(p, r) = nrp;
                    a(r, q) = a(q, r) = nrq;
                }

                // Eigenvektormatrix aktualisieren: V <- V * J
                for (std::size_t r = 0; r < n; ++r) {
                    const double vrp = v(r, p), vrq = v(r, q);
                    v(r, p) = c * vrp - s * vrq;
                    v(r, q) = s * vrp + c * vrq;
                }
            }
        }
        ++sweep;
        off = offNorm();
    }

    // Eigenwerte aufsteigend sortieren, Eigenvektoren (Spalten) mitpermutieren
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::sort(idx.begin(), idx.end(),
              [&](std::size_t i, std::size_t j) { return a(i, i) < a(j, j); });

    res.eigenvalues.resize(n);
    res.eigenvectors = Matrix(n, n);
    for (std::size_t k = 0; k < n; ++k) {
        res.eigenvalues[k] = a(idx[k], idx[k]);
        for (std::size_t r = 0; r < n; ++r) res.eigenvectors(r, k) = v(r, idx[k]);
    }
    res.sweeps = sweep;
    res.offDiagonalNorm = off;
    res.converged = (off <= threshold);
    return res;
}
