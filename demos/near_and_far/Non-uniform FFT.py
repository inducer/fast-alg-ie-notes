# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.15",
#     "matplotlib>=3.8",
#     "numpy>=2.0",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Non-Uniform FFT

    Copyright (C) 2026 Andreas Kloeckner
    <details>
    <summary>License</summary>
    MIT License Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    </details>

    Based on [Accelerating the Nonuniform Fast Fourier Transform](https://doi.org/10.1137/S003614450343200X) (Greengard, Lee).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Given nonuniform nodes $x_j\in[0,2\pi)$ and complex strengths $c_j$, we want

    $$F_k=\sum_{j=0}^{N-1}c_j e^{-ikx_j},\qquad -M/2\leq k<M/2.$$

    Idea:

    1. Regard the data as point masses $f=\sum_j c_j\delta_{x_j}$.
    2. Smooth them with a narrow periodic Gaussian (the heat kernel).
    3. Sample the smooth result on an oversampled uniform grid and use an FFT.
    4. Undo the known Gaussian attenuation in Fourier space.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    seed_control = mo.ui.slider(1, 100, value=28, step=1, label="Random seed")
    n_control = mo.ui.slider(1, 12, value=2, step=1, label="number of samples $N$")
    m_control = mo.ui.slider(16, 128, value=48, step=8, label="number of modes $M$")
    mo.vstack([seed_control, n_control, m_control])
    return m_control, n_control, seed_control


@app.cell(hide_code=True)
def _(m_control, n_control, np, seed_control):
    N = n_control.value
    M = m_control.value
    rng = np.random.default_rng(seed_control.value)

    x = np.sort(
        np.concatenate(
            [rng.uniform(0.15, 2.35, N // 2), rng.uniform(3.0, 2 * np.pi - 0.1, N - N // 2)]
        )
    )
    #c = (rng.normal(size=N) + 1j * rng.normal(size=N)) / N
    c = rng.normal(size=N)
    return M, c, x


@app.cell(hide_code=True)
def _(c, plt, x):
    plt.figure(figsize=(9, 2.6))
    plt.vlines(x, 0, c, color="#4c78a8", alpha=0.8)
    plt.scatter(x, c, color="#4c78a8", s=22)
    return


@app.cell
def _(M, c, np, x):
    ks = np.arange(-M // 2, M // 2)
    exact = np.exp(-1j * np.outer(ks, x)) @ c
    return exact, ks


@app.cell(hide_code=True)
def _(exact, ks, plt):
    _fig, _ax = plt.subplots(figsize=(9, 3.0))
    _ax.plot(ks, exact.real, ".-", label=r"$\Re F_k$")
    _ax.plot(ks, exact.imag, ".-", label=r"$\Im F_k$")
    _ax.set(xlabel="$k$", ylabel="$F_k$", title="Direct nonuniform DFT")
    _ax.legend(ncols=2)
    _fig
    return


@app.cell
def _(np):
    def gaussian_type1(x_src, strengths, mode_count, oversampling, tau):
        grid_count = oversampling * mode_count
        grid = 2 * np.pi * np.arange(grid_count) / grid_count

        # Signed distance to the nearest periodic image, evaluated on the *whole* grid.
        distance = (grid[:, None] - x_src[None, :] + np.pi) % (2 * np.pi) - np.pi
        bumps = np.exp(-(distance**2) / (4 * tau))
        smooth_grid = bumps @ strengths

        smooth_hat = np.fft.fft(smooth_grid) / grid_count
        modes = np.arange(-mode_count // 2, mode_count // 2)
        selected = smooth_hat[modes % grid_count]

        # Fourier coefficient of exp(-x^2/(4 tau)) is
        # sqrt(tau/pi) exp(-tau k^2), under the normalized 2pi-periodic convention.
        answer = np.sqrt(np.pi / tau) * np.exp(tau * modes**2) * selected
        return answer, grid, smooth_grid, bumps, selected

    return (gaussian_type1,)


@app.cell
def _(mo):
    r_control = mo.ui.slider(1, 4, value=2, step=1, label="oversampling $r$")
    alpha_control = mo.ui.slider(
        2.0, 20.0, value=12.0, step=0.5, label=r"Gaussian scale $\alpha$ in $\tau=\alpha/M^2$"
    )
    mo.hstack([r_control, alpha_control], widths="equal")
    return alpha_control, r_control


@app.cell
def _(M, alpha_control, c, gaussian_type1, r_control, x):
    r = r_control.value
    alpha = alpha_control.value
    tau = alpha / M**2
    approx, grid, smooth_grid, bumps, smooth_hat = gaussian_type1(x, c, M, r, tau)
    return alpha, approx, grid, r, smooth_grid, smooth_hat, tau


@app.cell(hide_code=True)
def _(grid, plt, smooth_grid):
    plt.figure(figsize=(10, 3))
    plt.plot(grid, smooth_grid.real, ".-", ms=3, label="real part")
    plt.plot(grid, smooth_grid.imag, ".-", ms=3, label="imaginary part")
    plt.legend()
    return


@app.cell(hide_code=True)
def _(exact, ks, np, plt, smooth_hat, tau):
    plt.figure(figsize=(10, 3))
    plt.semilogy(ks, np.maximum(np.abs(exact), 1e-16), ".-", label=r"$|F_k|$")
    plt.semilogy(ks, np.maximum(np.abs(smooth_hat), 1e-16), ".-", label=r"$|F_{\tau,k}|$")
    plt.semilogy(
        ks,
        np.maximum(np.sqrt(tau / np.pi) * np.exp(-tau * ks**2) * np.abs(exact), 1e-16),
        "--",
        label="predicted attenuation",
    )
    plt.gca().set(xlabel="$k$", ylabel="magnitude")
    plt.legend(ncols=3)
    return


@app.cell
def _(approx, exact, ks, plt):
    plt.figure(figsize=(10, 3))
    plt.plot(ks, exact.real, "x-", label="direct")
    plt.plot(ks, approx.real, "o--", label="Gaussian NUFFT")
    plt.xlabel("$k$")
    plt.ylabel(r"$\Re F_k$")
    plt.legend(ncols=2)
    return


@app.cell
def _(approx, exact, np):
    coefficient_error = np.abs(approx - exact)
    relative_l2_error = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    return (coefficient_error,)


@app.cell(hide_code=True)
def _(coefficient_error, ks, np, plt):
    plt.figure(figsize=(9, 3.0))
    plt.semilogy(ks, np.maximum(coefficient_error, 1e-17), ".-")
    plt.xlabel("$k$")
    plt.ylabel("absolute error")
    return


@app.cell
def _(M, c, exact, gaussian_type1, np, r, x):
    alpha_scan = np.geomspace(1.5, 30.0, 45)
    tau_errors = np.array(
        [
            np.linalg.norm(gaussian_type1(x, c, M, r, _a / M**2)[0] - exact)
            / np.linalg.norm(exact)
            for _a in alpha_scan
        ]
    )
    return alpha_scan, tau_errors


@app.cell(hide_code=True)
def _(alpha_scan, plt, tau_errors):
    plt.figure(figsize=(10,3))
    plt.loglog(alpha_scan, tau_errors, ".-")
    plt.xlabel(r"$\alpha$ in $\tau=\alpha/M^2$")
    plt.ylabel(r"relative $\ell^2$ error")
    return


@app.cell
def _(M, alpha, c, exact, gaussian_type1, np, x):
    oversampling_scan = np.arange(1, 7)
    oversampling_errors = np.array(
        [
            np.linalg.norm(gaussian_type1(x, c, M, int(_r), alpha / M**2)[0] - exact)
            / np.linalg.norm(exact)
            for _r in oversampling_scan
        ]
    )
    return oversampling_errors, oversampling_scan


@app.cell(hide_code=True)
def _(oversampling_errors, oversampling_scan, plt):
    plt.figure(figsize=(9, 3.0))
    plt.semilogy(oversampling_scan, oversampling_errors, "o-")
    plt.xlabel("oversampling factor $r$")
    plt.ylabel(r"relative $\ell^2$ error")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
