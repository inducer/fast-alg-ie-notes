#!/usr/bin/env python3
"""Numerical illustrations of the classical Ewald construction in §3.1.

Jiang & Greengard, *A dual-space multilevel kernel-splitting framework
for discrete and continuous convolution*, arXiv:2308.00292, §3.1.

Run
----
    python dmk-expn-test.py
    python dmk-expn-test.py --no-show

The script deliberately works with small direct sums instead of a NUFFT.  Its
purpose is to make the identities and error mechanisms in (17)--(39) visible:

* the exact Ewald split and the finite self-interaction correction;
* exponential localization of the residual kernel R;
* removal of the k=0 singularity by the physical-space window W;
* convergence of a tensor-product Fourier trapezoidal rule for W; and
* the accuracy/cost balance obtained by choosing sigma from the leaf-box size.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import pi, sqrt

import numpy as np
from scipy.special import erf, erfc


# The diameter of [-1/2, 1/2]^3, as in Lemma 8.
BOX_DIAMETER = sqrt(3.0)


def _at_zero(r: np.ndarray, value: float, expression: np.ndarray) -> np.ndarray:
    """Replace the removable r=0 value in a radial-kernel expression."""
    return np.where(np.asarray(r) == 0.0, value, expression)


def laplace(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return np.where(r == 0.0, np.inf, 1.0 / r)


def mollified(r: np.ndarray, sigma: float) -> np.ndarray:
    """M(r) = erf(r/sigma)/r, with M(0) from (21)."""
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = erf(r / sigma) / r
    return _at_zero(r, 2.0 / (sqrt(pi) * sigma), value)


def residual(r: np.ndarray, sigma: float) -> np.ndarray:
    """R(r) = erfc(r/sigma)/r (singular at zero)."""
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return erfc(r / sigma) / r


def windowed(r: np.ndarray, sigma: float, b: float = 6.0) -> np.ndarray:
    """Physical-space W in (26), evaluated stably at r=0."""
    r = np.asarray(r, dtype=float)
    a = BOX_DIAMETER + b * sigma
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (erf(r / sigma) - 0.5 * erf((a + r) / sigma)
                 + 0.5 * erf((a - r) / sigma)) / r
    # Differentiate the numerator at zero.  The two large terms cancel here.
    limit = (2.0 / (sqrt(pi) * sigma)) * (1.0 - np.exp(-(a / sigma) ** 2))
    return _at_zero(r, limit, value)


def window_hat(k: np.ndarray, sigma: float, c_tilde: float) -> np.ndarray:
    """Fourier transform (24), including its removable k=0 value."""
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = 8.0 * pi * (np.sin(c_tilde * k / 2.0) / k) ** 2
        value *= np.exp(-(sigma * k) ** 2 / 4.0)
    return _at_zero(k, 2.0 * pi * c_tilde**2, value)


def difference(r: np.ndarray, sigma: float) -> np.ndarray:
    """D_l with sigma_l=sigma and sigma_{l+1}=sigma/2; cf. (41)."""
    return mollified(r, sigma / 2.0) - mollified(r, sigma)


def pairwise_potential(targets: np.ndarray, sources: np.ndarray,
                       charges: np.ndarray, kernel) -> np.ndarray:
    distances = np.linalg.norm(targets[:, None, :] - sources[None, :, :], axis=2)
    return kernel(distances) @ charges


def fourier_window_potential(targets: np.ndarray, sources: np.ndarray,
                             charges: np.ndarray, sigma: float, b: float,
                             period: float, n: int) -> np.ndarray:
    """Direct Fourier sum for W using a cubic trapezoidal grid.

    The spacing h=2*pi/period makes this the inverse Fourier trapezoidal rule,
    including its h^3/(2*pi)^3 weight.  A large period keeps periodic images
    outside the region used in Lemma 8.
    """
    h = 2.0 * pi / period
    modes = np.arange(-n, n + 1)
    kx, ky, kz = np.meshgrid(h * modes, h * modes, h * modes, indexing="ij")
    k = np.column_stack((kx.ravel(), ky.ravel(), kz.ravel()))
    weights = (h / (2.0 * pi)) ** 3 * window_hat(
        np.linalg.norm(k, axis=1), sigma, BOX_DIAMETER + b * sigma)
    source_phase = np.exp(-1j * k @ sources.T) @ charges
    return np.real(np.exp(1j * targets @ k.T) @ (weights * source_phase))


@dataclass(frozen=True)
class TestResult:
    name: str
    value: float
    bound: float
    explanation: str


def run_numerical_tests(eps: float = 1.0e-8, sigma: float = 0.18,
                        b: float = 6.0) -> list[TestResult]:
    """Run compact numerical checks of the claims in §3.1."""
    rng = np.random.default_rng(230800292)
    results: list[TestResult] = []

    # (17), (19), and (21): the split is exact away from r=0, and M(0) is finite.
    # Below about 1e-6, subtracting two O(1/r) numbers is ill-conditioned
    # in double precision even though the analytic identity remains exact.
    r = np.geomspace(1.0e-6, BOX_DIAMETER, 400)
    split_error = np.max(np.abs(laplace(r) - mollified(r, sigma) - residual(r, sigma)))
    limit_error = abs(mollified(np.array([0.0]), sigma)[0] - 2.0 / (sqrt(pi) * sigma))
    results.append(TestResult("Ewald split (17)", max(split_error, limit_error), 2e-10,
                              "M + R reproduces 1/r and M has the finite limit (21)."))

    # Text below (18): erfc(6) is already below double-precision accuracy.
    tail = erfc(6.0)
    results.append(TestResult("Residual localization", tail, 3e-17,
                              "Relative residual size r R(r) at r = 6 sigma."))

    # Lemma 8 / (29): W and M agree in the unit box to O(erfc(b)).
    r = np.linspace(1.0e-8, BOX_DIAMETER, 2000)
    relative_window_error = np.max(np.abs(windowed(r, sigma, b) - mollified(r, sigma)) * r)
    results.append(TestResult("Window replacement (25), (29)", relative_window_error,
                              max(1.05 * erfc(b), 3e-16),
                              "max_{r <= sqrt(3)} r |W(r)-M(r)| (including roundoff)."))

    # (30)--(35): direct finite Fourier quadrature of W versus its analytic form.
    sources = rng.uniform(-0.5, 0.5, size=(12, 3))
    targets = rng.uniform(-0.5, 0.5, size=(10, 3))
    charges = rng.normal(size=12)
    exact = pairwise_potential(targets, sources, charges,
                               lambda radius: windowed(radius, sigma, b))
    # h=2pi/8: images are distant; Kmax = n*h satisfies (30) with margin.
    spectral = fourier_window_potential(targets, sources, charges, sigma, b,
                                        period=8.0, n=48)
    quadrature_error = np.max(np.abs(spectral - exact)) / np.max(np.abs(exact))
    results.append(TestResult("Fourier quadrature (31)--(35)", quadrature_error, 2e-5,
                              "Relative error of a direct 97^3 Fourier sum for W."))

    # (38)--(39): sigma makes residuals outside a leaf box negligible, whereas
    # the required Fourier cutoff grows inversely with the leaf size.
    leaf_width = 1.0 / 16.0
    sigma_leaf = leaf_width / sqrt(np.log(1.0 / eps))
    local_error = erfc(leaf_width / sigma_leaf)
    kmax = 2.0 * sqrt(np.log(1.0 / eps)) / sigma_leaf
    predicted = 2.0 * np.log(1.0 / eps) / leaf_width
    results.append(TestResult("Leaf-scale balance (38), (39)",
                              max(local_error / eps, abs(kmax / predicted - 1.0)),
                              1.01, "Residual is <= eps and Kmax has the predicted 1/r_L scaling."))

    # (43)--(51): difference kernels are exact rescalings, and their mode count
    # n_f = 3/pi log(1/eps) has no dependence on the refinement level.
    r = np.linspace(0.0, 1.0, 300)
    scale_error = np.max(np.abs(difference(r / 2.0, sigma / 2.0) - 2.0 * difference(r, sigma)))
    nf = int(np.ceil(3.0 * np.log(1.0 / eps) / pi))
    results.append(TestResult("Multilevel scaling (43)--(51)", scale_error, 2e-13,
                              f"D_(l+1)(r/2)=2 D_l(r); n_f={nf} is level-independent."))

    return results


def make_figure(sigma: float, b: float, eps: float) -> None:
    # Import only when plotting: --no-show remains usable without a GUI setup.
    import matplotlib.pyplot as plt

    r = np.linspace(1.0e-4, BOX_DIAMETER, 1600)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    axes[0, 0].plot(r, laplace(r), label=r"$1/r$", lw=2)
    axes[0, 0].plot(r, mollified(r, sigma), label=r"$M(r)$")
    axes[0, 0].plot(r, residual(r, sigma), label=r"$R(r)$")
    axes[0, 0].set(title="Ewald split (17)", xlabel=r"$r$", ylabel="kernel value", ylim=(0, 12))
    axes[0, 0].legend()

    axes[0, 1].semilogy(r / sigma, np.abs(residual(r, sigma)) * r, label=r"$rR(r)=\operatorname{erfc}(r/\sigma)$")
    axes[0, 1].axvline(6, color="k", ls=":", label=r"$r=6\sigma$")
    axes[0, 1].set(title="The residual is numerically local", xlabel=r"$r/\sigma$", ylabel="relative residual")
    axes[0, 1].legend()

    axes[1, 0].semilogy(r, np.abs(windowed(r, sigma, b) - mollified(r, sigma)) * r)
    axes[1, 0].axhline(erfc(b), color="k", ls=":", label=r"$\operatorname{erfc}(b)$")
    axes[1, 0].set(title="Lemma 8: windowing error", xlabel=r"$r$", ylabel=r"$r|W-M|$")
    axes[1, 0].legend()

    levels = np.arange(7)
    r_levels = 2.0 ** (-levels)
    sigma_levels = r_levels / sqrt(np.log(1.0 / eps))
    axes[1, 1].loglog(r_levels, sigma_levels, "o-", label=r"$\sigma_l$ from (43)")
    axes[1, 1].loglog(r_levels, 2.0 * np.sqrt(np.log(1.0 / eps)) / sigma_levels,
                       "s-", label=r"$K_l$ from (30)")
    axes[1, 1].set(title="Refinement trades range for bandwidth", xlabel=r"$r_l$", ylabel="scale")
    axes[1, 1].legend()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=1e-8, help="target tolerance (default: 1e-8)")
    parser.add_argument("--sigma", type=float, default=0.18, help="Ewald smoothing length")
    parser.add_argument("--b", type=float, default=6.0, help="window safety factor in Lemma 8")
    parser.add_argument("--no-show", action="store_true", help="run checks without opening plots")
    args = parser.parse_args()
    if not (0.0 < args.epsilon < 1.0 and args.sigma > 0.0 and args.b > 0.0):
        parser.error("epsilon, sigma, and b must be positive, with epsilon < 1")

    results = run_numerical_tests(args.epsilon, args.sigma, args.b)
    print("Classical Ewald tests from Jiang & Greengard §3.1\n")
    failed = []
    for result in results:
        ok = result.value <= result.bound
        print(f"{'PASS' if ok else 'FAIL'}  {result.name}: {result.value:.3e} (bound {result.bound:.3e})")
        print(f"      {result.explanation}")
        if not ok:
            failed.append(result.name)
    if failed:
        raise SystemExit("Failed: " + ", ".join(failed))
    if not args.no_show:
        make_figure(args.sigma, args.b, args.epsilon)


if __name__ == "__main__":
    main()
