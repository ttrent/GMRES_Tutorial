"""Shared pieces of the GMRES tutorial notebooks (GMRES_<n>_<topic>.ipynb).

Only what every notebook uses lives here: the counting matvec, the example
problems, the version-against-version check and the plot style. The solvers and
the algorithms they are built from stay in the notebooks, where the reader sees
them. A notebook that needs an earlier step's solver loads it from the notebook
that builds it, with `import_from_notebook`, so there is only one copy.

Run `python tutorial_utils.py` to check that every definition appearing in more
than one place (here, or in several notebooks) is identical everywhere.
"""

import ast
import json
import linecache
import pathlib
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.ticker import MaxNLocator


# =============================================================================
# The matvec (Step 1, GMRES_1_krylov.ipynb)
# =============================================================================


class CountingMatvec:
    """The function v -> A v, plus a count of how many times it has been called.

    The solver only ever gets this object, never A itself. The count lets us
    check how many matvecs each version of the solver spends.
    """

    def __init__(self, A):
        self._A = A  # hidden from the solver; only used inside __call__
        self.count = 0

    def __call__(self, v):
        self.count += 1
        return self._A @ v


# =============================================================================
# Example problems. Each builder returns (A, b) as float64 arrays.
# =============================================================================


def geo3():
    """A generic nonsymmetric, well-conditioned 3x3 (determinant 13), b = (1, 1, 1)."""
    A = np.array([
        [3.0, 1.0, 0.0],
        [0.0, 2.0, 1.0],
        [1.0, 0.0, 2.0],
    ])
    return A, np.ones(3)


def diag_narrow(n=100):
    """Diagonal, eigenvalues evenly spaced in [1, 1.5], b = ones: fast convergence."""
    return np.diag(np.linspace(1, 1.5, n)), np.ones(n)


def diag_spread(n=20, decades=4):
    """Diagonal, eigenvalues log-spaced from 1 to 10^decades, b = ones."""
    return np.diag(np.logspace(0, decades, n)), np.ones(n)


def convdiff_n(n=20, *, c, rhs="ones"):
    """Central differences of -u'' + c u' on [0, 1], Dirichlet boundaries, h = 1/(n+1).

    Tridiagonal: 2/h^2 on the diagonal, -1/h^2 - c/(2h) below it and
    -1/h^2 + c/(2h) above it. Symmetric for c = 0, nonsymmetric otherwise.

    With c = 0, two right-hand sides break down early. "ones" is symmetric about
    the middle of the grid, so its Krylov space runs out at k = n/2 (n even).
    "sine" is sin(pi x_i), an exact eigenvector, so it runs out at k = 1.

    Args:
        n: number of interior grid points.
        c: advection velocity. No default: the value that makes the matrix
            visibly nonnormal is chosen in Step 5.
        rhs: "ones", "sine" (sin(pi x_i)) or "ramp" (x_i).
    """
    h = 1.0 / (n + 1)
    grid = h * np.arange(1, n + 1)
    A = (
        np.diag(np.full(n, 2.0 / h**2))
        + np.diag(np.full(n - 1, -1.0 / h**2 - c / (2 * h)), -1)
        + np.diag(np.full(n - 1, -1.0 / h**2 + c / (2 * h)), 1)
    )
    right_hand_sides = {
        "ones": np.ones(n),
        "sine": np.sin(np.pi * grid),
        "ramp": grid,
    }
    return A, right_hand_sides[rhs]


def shift_n(n=8):
    """The cyclic shift, A e_j = e_{j+1} and A e_n = e_1, with b = e_1.

    A permutation matrix: orthogonal, condition number 1. Even so, GMRES makes
    no progress at all until step n, because A K_k is perpendicular to b for k < n.
    """
    return np.roll(np.eye(n), 1, axis=0), np.eye(n)[0]


# Name -> builder, for running every solver on every example (Step 9).
# Builders with a required argument are registered with a fixed choice of it.
EXAMPLES = {
    "geo3": geo3,
    "diag_narrow": diag_narrow,
    "diag_spread": diag_spread,
    "convdiff_n": lambda: convdiff_n(c=20.0),
    "shift_n": shift_n,
}


# =============================================================================
# Checking one version against another
# =============================================================================


def compare_versions(old, new, A, b, *, ks=None, **solver_kwargs):
    """Run two solver versions for k = 1, 2, ... steps each and measure how far apart they are.

    Each run gets a fresh CountingMatvec(A) and rtol=0, so it takes exactly
    max_iterations=k steps unless the Krylov space runs out first.

    Args:
        old, new: solvers with the shared signature (matvec, b, *, rtol, max_iterations).
        A: the matrix, used to build the matvecs and to measure true residuals.
        b: right-hand side, shape (n,).
        ks: the step counts to compare. Defaults to 1, ..., n.
        **solver_kwargs: passed to both solvers.

    Returns:
        dict of arrays, one entry per k:
            "k"                    the step count,
            "x_difference"         ||x_old - x_new|| / ||x_new||,
            "residual_difference"  |last residual of old - last residual of new| / ||b||,
                                   each as the solver itself reported it,
            "true_residual_new"    ||b - A x_new|| / ||b||,
            "matvecs_equal"        whether both spent the same number of matvecs.
    """
    if ks is None:
        ks = range(1, b.shape[0] + 1)
    b_norm = np.linalg.norm(b)
    columns = {name: [] for name in
               ("k", "x_difference", "residual_difference", "true_residual_new", "matvecs_equal")}
    for k in ks:
        x_old, record_old = old(CountingMatvec(A), b, rtol=0.0, max_iterations=k, **solver_kwargs)
        x_new, record_new = new(CountingMatvec(A), b, rtol=0.0, max_iterations=k, **solver_kwargs)
        columns["k"].append(k)
        columns["x_difference"].append(np.linalg.norm(x_old - x_new) / np.linalg.norm(x_new))
        columns["residual_difference"].append(
            abs(record_old["residual_history"][-1] - record_new["residual_history"][-1]) / b_norm
        )
        columns["true_residual_new"].append(np.linalg.norm(b - A @ x_new) / b_norm)
        columns["matvecs_equal"].append(record_old["matvecs"] == record_new["matvecs"])
    return {name: np.array(values) for name, values in columns.items()}


# =============================================================================
# Plot style, matching GMRES_1_krylov.ipynb
# =============================================================================

LINE_COLOR = "#2a78d6"       # the main series
REFERENCE_COLOR = "#52514e"  # reference lines and their labels
GRID_COLOR = "#e4e3df"
# Colors for overlaid series, used in this order. The first three pass the
# colorblind-separation checks as a set; past three, split into more panels.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")

# One-hue blue ramp for heatmaps of |entries|: light = small, dark = large.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "tutorial_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def add_figure_caption(fig, label, caption):
    """Write a bold "Figure <label>." and a short caption under the figure, as part of the image.

    Args:
        fig: matplotlib figure.
        label: the figure number, e.g. "2.1".
        caption: one or two sentences; matplotlib math such as $\\bar H_k$ is allowed.
    """
    fig.text(0.01, -0.02, rf"$\bf{{Figure\ {label}.}}$  {caption}", ha="left", va="top",
             fontsize=10, color="#0b0b0b", wrap=True)


def style_axes(ax, *, xticks=None):
    """Recessive grid, hidden top and right spines, and optional x ticks."""
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_residuals(ax, residual_history, *, b_norm, label=None, color=LINE_COLOR, rtol=None,
                   marker="o", linestyle="-"):
    """Semilog plot of ||r_k|| / ||b|| against the step k.

    Args:
        ax: matplotlib axes.
        residual_history: [||r_0||, ||r_1||, ...], as in a solver's record.
        b_norm: ||b||, to make the residuals relative.
        label: legend label for the line.
        color: line color.
        rtol: if given, also draw the tolerance as a labelled horizontal line.
        marker, linestyle: passed to matplotlib.

    Returns:
        The plotted line.
    """
    steps = np.arange(len(residual_history))
    (line,) = ax.semilogy(steps, np.asarray(residual_history) / b_norm, color=color, linewidth=2,
                          marker=marker, markersize=5, linestyle=linestyle, label=label)
    if rtol is not None:
        ax.axhline(rtol, color=REFERENCE_COLOR, linewidth=1)
        ax.text(0.2, rtol * 2, "rtol: stop once below this line", color=REFERENCE_COLOR,
                fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("step k  (one matvec per step)")
    ax.set_ylabel(r"$\|r_k\|_2 \,/\, \|b\|_2$")
    return line


def plot_matrix(ax, M, *, title=None, log=False, floor=0.0, vmin=None, vmax=None, colorbar=True):
    """Heatmap of |M| with 1-based row and column numbers, as in the notes.

    Entries with |m| <= floor * max|M| are left blank, so exact zeros (and, with
    a floor, round-off) read as empty cells.

    Args:
        ax: matplotlib axes.
        M: the matrix.
        title: axes title.
        log: color by log10 |m| instead of |m|.
        floor: relative threshold below which entries are blank.
        vmin, vmax: color limits, to share one scale across several panels.
            Default to the smallest shown and the largest |entry|.
        colorbar: draw a colorbar beside the axes.

    Returns:
        The image.
    """
    magnitude = np.abs(np.asarray(M, dtype=float))
    largest = magnitude.max()
    shown = np.ma.masked_where(magnitude <= floor * largest, magnitude)
    if log:
        norm = LogNorm(vmin=shown.min() if vmin is None else vmin,
                       vmax=largest if vmax is None else vmax)
    else:
        norm = Normalize(vmin=0.0 if vmin is None else vmin,
                         vmax=largest if vmax is None else vmax)
    rows, columns = magnitude.shape
    image = ax.imshow(shown, cmap=SEQUENTIAL_CMAP, norm=norm, interpolation="nearest",
                      extent=(0.5, columns + 0.5, rows + 0.5, 0.5))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("column j")
    ax.set_ylabel("row i")
    if title is not None:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(image, ax=ax, shrink=0.8, label="|entry|")
    return image


# =============================================================================
# Reading notebooks: using an earlier step's definitions, and the drift check
# =============================================================================


def _code_cells(notebook):
    """Yield the source of each code cell, with IPython magics and shell escapes blanked out."""
    for cell in json.loads(pathlib.Path(notebook).read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        yield "\n".join("" if line.lstrip().startswith(("%", "!")) else line for line in lines)


def _is_notebook_import(node):
    """Whether `node` is a top-level `name, ... = import_from_notebook(...)`."""
    return (
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "import_from_notebook"
    )


def import_from_notebook(notebook, *names):
    """Load functions or classes from another tutorial notebook, without running the rest of it.

    Only the notebook's top-level import statements, defs and classes are run,
    plus its own `import_from_notebook` loads ("From Step n" cells), so its
    plots, prints and example data are skipped. A loaded definition must
    therefore not rely on any other top-level variable of that notebook.
    The source is registered with linecache, so `name??` and tracebacks show it.

    Args:
        notebook: file name of the notebook, e.g. "GMRES_1_krylov.ipynb", in this
            module's directory.
        *names: the functions or classes to return.

    Returns:
        The named definition, or a tuple of them in the order asked for.

    Raises:
        ValueError: If the notebook doesn't define one of the names.
    """
    path = pathlib.Path(__file__).resolve().parent / notebook
    kept = []
    for source in _code_cells(path):
        lines = source.splitlines()
        for node in ast.parse(source).body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)) or _is_notebook_import(node):
                first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
                kept.append("\n".join(lines[first - 1 : node.end_lineno]))
    program = "\n\n\n".join(kept) + "\n"

    filename = f"<{notebook}>"
    linecache.cache[filename] = (len(program), None, program.splitlines(keepends=True), filename)
    namespace = {}
    exec(compile(program, filename, "exec"), namespace)

    missing = [name for name in names if name not in namespace]
    if missing:
        raise ValueError(f"{notebook} defines no {', '.join(missing)}")
    found = tuple(namespace[name] for name in names)
    return found[0] if len(found) == 1 else found


def _top_level_definitions(source):
    """Map each top-level def and class name in `source` to its source text."""
    tree = ast.parse(source)
    return {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def _notebook_definitions(path):
    """Top-level defs and classes in a notebook's code cells, name -> source text."""
    definitions = {}
    for source in _code_cells(path):
        definitions.update(_top_level_definitions(source))
    return definitions


def find_mismatched_copies(directory=None):
    """Find definitions that appear in several places with different source text.

    Looks at this module and every GMRES_*.ipynb. A def or class whose name
    appears in only one place is skipped.

    Args:
        directory: where the notebooks are. Defaults to this file's directory.

    Returns:
        A list of problems, empty if every copy matches.
    """
    here = pathlib.Path(__file__).resolve()
    directory = here.parent if directory is None else pathlib.Path(directory)

    copies = {}  # name -> {place: source text}
    for name, source in _top_level_definitions(here.read_text()).items():
        copies.setdefault(name, {})[here.name] = source
    for notebook in sorted(directory.glob("GMRES_*.ipynb")):
        for name, source in _notebook_definitions(notebook).items():
            copies.setdefault(name, {})[notebook.name] = source

    problems = []
    for name, by_place in copies.items():
        if len(set(by_place.values())) > 1:
            problems.append(f"{name}: the copies in {', '.join(by_place)} differ")
    return problems


if __name__ == "__main__":
    found = find_mismatched_copies()
    for problem in found:
        print(problem)
    if found:
        sys.exit(1)
    print("Every repeated definition matches.")
