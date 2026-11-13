# GMRES Tutorial — Outline

Planning document for a series of notebooks (`GMRES_<n>_<topic>.ipynb`) that builds GMRES up
from the bare idea to the solver in
`WhistlerRegulatedHeatDispersion/src/time_integration/implicit/`. Every step follows the same
pattern: run the previous version on a small matrix, show what goes wrong, then add the fix.

Steps 0–6 are written. For those, this outline records what each notebook contains and the
numbers it found. For Steps 7–9 it lists what each notebook should contain.

---

## The series at a glance

The ten steps fall into three parts. Each part ends with a solver that the next part takes as
its starting point.

| Part | Notebooks | Question | Kind of step | Ends with |
|---|---|---|---|---|
| **I. A correct GMRES** | 1–5 | How does GMRES compute its iterate, accurately and safely? | Same math, computed better (2–4) or safely (5) | `gmres_v4` |
| **II. A fast GMRES** | 6–8 | Why does it need many steps on some matrices, and what can be done about it? | Analysis (6), then changes the math (7, 8) | `gmres_v6` |
| **III. The production code** | 9 | How does the same algorithm look in JAX, and why? | Translation | production `gmres` |

- **Part I** keeps the iterates fixed. Every version from v1 to v4 produces the same x_k as
  `gmres_v0` would in exact arithmetic. What changes is how much round-off the computation
  picks up, how much work each step costs, and whether the solver can be trusted when it
  reports success. Each step is checked against the previous version with `compare_versions`.
- **Part II** asks how many steps that solver takes. Step 6 explains the answer with
  polynomials and eigenvalues. Steps 7 and 8 then change the iterates on purpose: restarting
  limits the cost per step, and preconditioning changes the problem so that fewer steps are
  needed. Each new version is checked to reduce to the previous one when its new option is
  switched off.
- **Part III** translates `gmres_v6` into the production JAX code and runs both on every
  example.

A reader can stop after Part I with a complete, correct (if unrestarted and unpreconditioned)
GMRES. Part II needs only `gmres_v4` from Part I, and Part III only `gmres_v6` and the
examples.

| Notebook | Step | Part | Kind | Builds | Loads from earlier notebooks | Status |
|---|---|---|---|---|---|---|
| `GMRES_1_krylov.ipynb` | 0–1 | I | first version | `gmres_v0` | — | done |
| `GMRES_2_arnoldi_orthogonalise.ipynb` | 2 | I | same math, computed better | `gmres_v1` | 1 | done |
| `GMRES_3_arnoldi_least_squares.ipynb` | 3 | I | same math, computed better | `gmres_v2` | 1, 2 | done |
| `GMRES_4_givens.ipynb` | 4 | I | same math, computed better | `gmres_v3` | 2, 3 | done |
| `GMRES_5_breakdown.ipynb` | 5 | I | same math, computed safely | `gmres_v4` | 2, 4 | done |
| `GMRES_6_convergence.ipynb` | 6 | II | analysis, no new version | (uses v4) | 5 | done |
| `GMRES_7_restart.ipynb` | 7 | II | changes the math | `gmres_v5` | 5 | planned |
| `GMRES_8_preconditioning.ipynb` | 8 | II | changes the math | `gmres_v6` | 7 | planned |
| `GMRES_8a_probing.ipynb` | appendix | II | helpers, no new version | probing helpers | 8 | proposed |
| `GMRES_9_jax.ipynb` | 9 | III | translation | (production `gmres`) | 8 | planned |

## How each notebook leads to the next

Each notebook ends on a problem, and the next one exists to fix it.

**Part I: a correct GMRES**

1. **Step 1 → 2.** GMRES searches the Krylov space `K_k = span{b, Ab, …, A^{k−1}b}`. The space
   is fine but the literal basis `b, Ab, A²b, …` is not. Its columns turn toward the dominant
   eigenvector, and on `diag_spread` cond(K_5) reaches about 3e15, after which the residual
   *rises*. Step 2 keeps the space and builds an orthonormal basis for it (Arnoldi).
2. **Step 2 → 3.** `gmres_v1` still solves an n×k least-squares problem from scratch at every
   step, stores all of `A V_k`, and throws away the numbers `h_ij` that Arnoldi computed.
   Step 3 shows those numbers record each `A v_j` in the new basis, which shrinks the problem
   to (k+1)×k, with no n in it.
3. **Step 3 → 4.** `gmres_v2` re-solves the whole small problem every step only to learn the
   residual norm, although `H̄_k` is just `H̄_{k−1}` with one column added. Step 4 updates a QR
   factorisation one column at a time with Givens rotations, and the residual norm comes out
   of the update for free.
4. **Step 4 → 5.** Step 4 ends with "the residual is zero exactly when s_k = 0", which holds
   only if the triangular factor R_k is nonsingular. On a singular A it isn't: `gmres_v3`
   reports convergence on `singular3` and `nilpotent2` while returning 1.5e16 or NaN.
   Step 5 makes the solver trustworthy at a breakdown.

**Part I → Part II**

5. **Step 5 → 6.** The solver is now correct, so the remaining question is speed. Step 4
   already showed that the factor by which each step shrinks the residual, |s_k|, is about 0.1
   on `diag_narrow`, about 0.96 on `convdiff_n` and exactly 1 on `shift_n` until the last step.
   Step 6 explains what decides it.

**Part II: a fast GMRES**

6. **Step 6 → 7.** The residual is a curve q_k through (0, 1), and GMRES picks the lowest one
   on the eigenvalues. Few steps when the eigenvalues sit in a few bunches whose width is small
   compared with their distance from 0. Many when they reach toward 0 (`diag_spread`),
   surround it (`shift_n`), or belong to a stencil matrix that clears about one cell per step
   (`convdiff_n`, I + Δt·L at large Δt). Many steps are expensive: step j orthogonalises
   against j vectors and the solver stores all of them. Step 7 caps the steps at m and starts
   again.
7. **Step 7 → 8.** Restarting caps the cost but can stall: `shift_n` never converges with
   m < n, and on `embree3` a longer restart length stalls where a shorter one converges.
   Step 8 removes the cause instead: it changes the problem so its eigenvalues cluster.

**Part II → Part III**

8. **Step 8 → 9.** `gmres_v6` is the full algorithm in NumPy. Step 9 rewrites it under JAX's
   constraints (immutable arrays, fixed shapes, no Python `if` or `break` on values) and
   shows that each constraint explains a design choice in production.

---

## Scope

**In:** GMRES as a linear solver: `gmres.py`, `arnoldi.py`, `givens.py`, `preconditioner.py`,
`vector_ops.py`, with `jacobian_blocks.py` as an appendix on building a preconditioner.

**Out:** Newton iteration, `jax.linearize` inside Newton, Eisenstat–Walker, `residual.py`, the
per-field Newton stopping tests, and differentiating through GMRES. The Newton context gets one
paragraph (in Step 0): in practice the matrix is the Jacobian of an implicit time step, and is
only available as a function `v -> Av`.

## Conventions

**Code layout**

- **Algorithms stay in the notebooks.** Every solver version and the algorithms it is built
  from (`krylov_matrix`, the Arnoldi step functions, `arnoldi`, `givens_rotation`,
  `back_substitution`, …) is defined in the notebook that introduces it, so the reader sees
  it. Anything used in only one notebook stays in that notebook.
- **Loading from an earlier notebook.** A notebook that needs an earlier step's code loads it
  in a "From Step n" cell with `import_from_notebook("GMRES_<n>_<topic>.ipynb", "name", ...)`.
  That runs only the earlier notebook's imports, defs and classes, plus its own "From Step n"
  loads, so each definition has a single copy. (This is how notebook 4 loads `gmres_v2` from
  notebook 3, which itself loads `arnoldi_step_classical_gram_schmidt_twice` from notebook 2.)
  A loaded function must therefore not depend on its notebook's other top-level variables.
- **`tutorial_utils.py`** holds only what every notebook uses: `CountingMatvec`, the example
  builders (each returns `(A, b)`) and `EXAMPLES`, `compare_versions`, `import_from_notebook`,
  and the plot style (`style_axes`, `plot_residuals`, `plot_residual_polynomial`,
  `plot_matrix`, `add_figure_caption`, colours). Notebooks import it with `%autoreload 2` and an explicit
  `from tutorial_utils import (...)`.
- `python tutorial_utils.py` runs `find_mismatched_copies`, which checks that any definition
  appearing in more than one place (today only `CountingMatvec`, in notebook 1 and the module)
  is identical everywhere.
- NumPy in float64 for Steps 0–8, JAX (float64) only in Step 9.

**Solver interface**

- Every version takes a `matvec` function, never a matrix, starting from v0.
- Shared signature `(matvec, b, *, rtol, max_iterations, verbose)`, returning the solution and
  a record with `iterations`, `matvecs`, `residual_history` and `converged`.
- Later versions only add keywords and record keys, never rename or remove them:

  | Version | Step | Adds |
  |---|---|---|
  | `gmres_v3` | 4 | keyword `atol`; tolerance `max(rtol·‖b‖, atol)` |
  | `gmres_v4` | 5 | record keys `breakdown` and `residual_norm` (the residual of the returned x: the Givens estimate, or the true residual after a breakdown) |
  | `gmres_v5` | 7 | keyword `restart=None`; record key `cycles` |
  | `gmres_v6` | 8 | optional third positional argument `preconditioner=None`, in production's order |

**Checking each version**

- **Same math** steps (2–5) check the new version against the previous one with
  `compare_versions(old, new, A, b)`. It runs both with `rtol=0, max_iterations=k` for each k
  and reports the relative difference of the iterates. "Up to round-off" means within about
  cond(basis)·ε, so the check asserts only where the previous version is well conditioned
  (see Step 2: v0 against v1).
- **Changes the math** steps (7, 8) check that the new version equals the previous one with
  its new option off: v5 with `restart ≥ max_iterations` equals v4, and v6 with no
  preconditioner equals v5.
- Step 9 runs production `gmres` against `gmres_v6` on every entry of `EXAMPLES`.

**Notation and writing**

- x_0 = 0 throughout. `K_k` and `c` for the literal Krylov basis and its coefficients
  (Step 1); `v_1 … v_{k+1}`, `V_k`, `h_ij`, `H̄_k` ((k+1)×k), `H_k` (its top k×k block), `y`,
  `β = ‖b‖₂` and `e_1` from Step 2; `G_j`, `c_j`, `s_j`, `Ω_k`, `R_k` and `g` from Step 4.
- The grid spacing is written Δs, with points s_i = iΔs (Steps 3–6), because h_ij is the
  Hessenberg entry. Step 6 adds the time step Δt and the cell Péclet number cΔs/2.
- Every symbol is defined in the text where it first appears, and every example matrix states
  its A and b the first time a notebook uses it.
- Function and variable names spell terms out (`modified_gram_schmidt`, not `mgs`).
- Every figure has a numbered caption inside the image, under the plot
  (`add_figure_caption(fig, "2.3", "...")`, numbered `<step>.<n>`), and the text refers to
  figures by that number.
- Each notebook ends with a "Summary of Step n" (bullets), a "Ties to code" paragraph linking
  the production files, and a "Next: Step n+1" paragraph naming the problem the next step
  fixes.

## Still open

- Math depth overall: full derivations, or short derivations plus a numerical check each?
- Sliders (`ipywidgets`) for "step k" plots, or static figure grids? (Step 3 uses static
  grids.)
- The probing appendix as its own notebook (`GMRES_8a_probing.ipynb`) or inside notebook 8?
- Production's pivot test in `gmres.py: _arnoldi_steps` is absolute (`pivot > 0`). On
  `singular3` the round-off pivot 1.7e-16 passes it, and the true-residual check then falls
  back to the start of the cycle (x = 0, residual 1.73) instead of the best iterate x₂
  (residual 1.0). A relative test, `pivot > eps * norm(column)`, returns x₂ and changes
  nothing on the other examples (Step 5.9). Change production? The tutorial's v4–v6 keep the
  absolute test so Step 9 compares exactly.
- `gmres.py`'s module docstring counts a solve as `iterations + cycles − 1` operator calls,
  plus one per breakdown cycle. It leaves out the call spent on an unusable step:
  `nilpotent2` takes 0 iterations and 2 calls (Step 5.7).

**Still to check numerically**

- ~~The velocity c at which `convdiff_n` becomes visibly nonnormal.~~ Resolved in Step 6.6:
  default c = 20 (cΔs/2 = 0.48 at n = 20). Its diagonal twin (same eigenvalues, b = ones) is
  11× lower after 5 steps (0.076 against 0.836), and as c goes from 10 to 40 the eigenvalue
  spread λ_max/λ_min falls from 49.5 to 1.86 with almost no change in convergence
  (‖r₁₀‖/‖b‖ 0.604 → 0.705).
- That v5, v6 and production `gmres` give the same iterates to round-off (Step 9).

---

## Example matrices

Builders live in `tutorial_utils.py`, each returning `(A, b)`, and are added by the first
notebook that uses them. Each example is a matrix together with a right-hand side, because
GMRES's behaviour depends on both.

### How big

- **2×2 and 3×3:** small enough to draw and to check by hand. In 3D every Krylov space is a
  point, a line, a plane or all of space, so the geometry of the residual can be seen
  directly. GMRES on a 3×3 finishes in at most 3 steps, though, so these can't show
  convergence *behaviour*. They suit the breakdown cases of Step 5, where one step is the
  whole story.
- **n ≈ 8–20:** enough steps for a residual curve to have a shape, while `H̄` and `VᵀV`
  heatmaps still have readable cells.
- **n ≈ 20–50:** for effects that need many steps (loss of orthogonality, restart stagnation,
  the effect of a preconditioner). Heatmaps get crowded past about 50. Timing plots, which
  only plot time per step, can go to n in the hundreds (Step 4 uses n = 400).

| Name | Size | Shows | Used in Step | In `tutorial_utils.py` |
|---|---|---|---|---|
| `geo3` | 3×3 | Krylov geometry; Arnoldi by hand; a flagged breakdown at k = 3 | 1, 2, 3, 4, 5 | yes |
| `diag_narrow` | 100×100 | Fast convergence, eigenvalues in [1, 1.5] | 1, 2, 3, 4, 5, 6 | yes |
| `diag_spread` | n×n, n = 20 | Slow convergence; v0's failure; at 12 decades, one-pass Gram–Schmidt's failure | 1–4, 6 (12 decades: 2, 3) | yes |
| `convdiff_n` | n×n, n ≈ 20–50 | Hessenberg shape, nonnormality, preconditioning | 3, 4, 5, 6, 7, 8 | yes |
| `shift_n` | n×n, n = 8 | No progress at all until step n | 4, 6, 7 | yes |
| `eig_rhs` | 3×3 | Lucky breakdown at step 1 | 5 | yes |
| `singular3` | 3×3 | Breakdown with a round-off zero pivot; v3 reports false convergence. With `rhs="in_range"`, a lucky breakdown on the same singular A | 5 | yes |
| `nilpotent2` | 2×2 | Exact zero pivot on the very first step | 5 | yes |
| `diag_clustered` | n×n, n = 30 (also 3000 in Step 6) | Three bunches ±1e-3 around 1, 2, 3: a drop of ~1e-4 every 3 steps, 9 steps to 1e-10 at any n | 6 | yes |
| `embree3` | 3×3 | A longer restart can be slower | 7 | to add |
| `two_scale` | 2n×2n, n ≈ 10 | Why scaling matters for a 2-norm method | 8 | to add |

`EXAMPLES` (the name → builder map Step 9 runs over) holds `geo3`, `diag_narrow`,
`diag_spread`, `diag_clustered`, `convdiff_n` (through its default, c = 20), `shift_n`,
`eig_rhs`, `singular3`, `singular3_in_range` and `nilpotent2` today. Later builders join it
when added.

### Each matrix

**`geo3`: a generic 3×3.** `A = [[3, 1, 0], [0, 2, 1], [1, 0, 2]]` (nonsymmetric, determinant
13), `b = (1, 1, 1)`. Nothing special about it, which is the point: GMRES needs all 3 steps,
going through a line (k = 1), a plane (k = 2) and all of space (k = 3, exact solution
`x = (3, 4, 5)/13`). The running example for "what is a Krylov space" and "what is GMRES
minimising". Arnoldi by hand (Step 2): β = √3, v₁ = (1,1,1)/√3, h₁₁ = 10/3, h₂₁ = √2/3,
v₂ = (2,−1,−1)/√6, h₁₂ = √2/3, h₂₂ = 13/6, h₃₂ = √3/2, v₃ = (0,−1,1)/√2. Its h₁₃ is 0 by
accident (Aᵀb = (4,3,3) ⟂ v₃; with b = (1,2,3) it is 0.63), so it can't show the Hessenberg
shape. Step 3 is a flagged breakdown, since K₃ is all of R³.

**`diag_narrow`: the easy case.** `diag(linspace(1, 1.5, 100))`, `b = ones(100)`. Eigenvalues
bunched in [1, 1.5]: GMRES shrinks the residual about 10× per step and reaches 1e-10 after 11
steps, far fewer than n.

**`diag_spread` and `diag_clustered`: eigenvalues you choose.** Diagonal matrices
`np.diag(lam)` with `b = ones(n)`. Their eigenvalues are the diagonal entries and their
eigenvectors the coordinate axes, so they are *normal*, and the residual is exactly the curve
at the dots: with b = ones, ‖r_k‖/‖b‖ is the RMS of q_k(λ_i) (Step 6.1). That isolates the one variable Step 6 is about: where the eigenvalues sit. Being symmetric,
they give a tridiagonal `H̄`, so they can't show the Hessenberg shape.
- *spread:* `diag_spread(n=20, decades=4)`, eigenvalues `np.logspace(0, decades, n)`
  (condition number 1e4). Convergence is slow and steady. With `decades=12` (condition number
  1e12) one-pass Gram–Schmidt visibly fails (Steps 2, 3). `convdiff_n` can't show that: for
  n = 20–50 and c up to 100, one pass of either kind keeps the loss of orthogonality below
  about 4e-13.
- *clustered:* `diag_clustered(n=30, centers=(1, 2, 3), spread=1e-3)`, the n eigenvalues
  shared evenly between three bunches, each evenly spaced over center ± spread. One degree-3
  polynomial is small on all three bunches, so the residual drops by about 1e-4 every 3 steps
  (1.8e-4, 3.3e-8, 5.9e-12 at k = 3, 6, 9) and reaches 1e-10 in 9 steps at n = 30, 300 or
  3000. Shrink each bunch to a point and it becomes Step 5.3's diag(1,1,2,2,3,3).

**`convdiff_n`: the PDE-like example.** `convdiff_n(n=20, *, c=20.0, rhs="ones")`: central
differences of `−u'' + c·u'` on [0, 1], Dirichlet boundaries, spacing `Δs = 1/(n+1)`. A
tridiagonal matrix with diagonal `2/Δs²`, sub-diagonal `−1/Δs² − c/(2Δs)` and super-diagonal
`−1/Δs² + c/(2Δs)`. `rhs` is `"ones"`, `"sine"` (sin(πs_i)) or `"ramp"` (s_i). The velocity c
(equivalently the cell Péclet number cΔs/2) sets its character:
- `c = 0`: symmetric, the 1D heat-equation matrix, with a tridiagonal `H̄` (Step 3.4). Two
  right-hand sides break down early: `ones` is symmetric about the grid's middle, so its
  Krylov space stops at k = n/2 (k = 10 for n = 20); `sine` is an eigenvector in exact
  arithmetic (k = 1). In floating point the step-1 remainder of `sine` is 61.8 ε‖Av₁‖, so the
  breakdown is not flagged, but the residual is already 1.4e-14‖b‖ and GMRES stops on its
  tolerance at k = 1 (Step 5.6).
- `c > 0`: nonsymmetric, with a full upper triangle in `H̄`, and *nonnormal*: the eigenvalues
  stop predicting convergence (Step 6.6). For cΔs/2 < 1 the eigenvalues are real. With
  b = ones, A·ones is nonzero only on cells 1 and n, so the residual is exactly 1 on the middle
  n − 2k cells and ‖r_k‖ ≥ √(n − 2k) (c = 0 sits exactly on this floor and breaks down at
  k = n/2). At c = 20 only the inflow front (s = 0) clears cells, and the residual follows the
  one-sided curve √(1 − k/n): 0.836 at k = 5, 0.670 at k = 10, a mean factor of 0.961 per step
  (the 0.96 of Step 4.7), while the diagonal twin with the same eigenvalues is at 0.076 at
  k = 5. At c = 42 (cΔs/2 = 1) the super-diagonal is 0 up to round-off (5.7e-14), A is lower
  bidiagonal with the single eigenvalue 2/Δs², and ‖r_k‖/‖b‖ = √(1 − k/n): one dot, all n
  steps. With c = 10, 20, 42 or 100, none of the three right-hand sides broke down before
  k = n.

It is the closest example to the simulation (a 1D grid, a tridiagonal stencil), large enough to
show restart stagnation, and its structure makes tridiagonal preconditioning and probing
(Step 8) natural. A variant `I + Δt·L` (one backward-Euler step of the heat equation, L =
`convdiff_n(n, c=0)`) adds a second knob: small Δt pushes the eigenvalues toward 1 and GMRES
gets easy, which is why implicit steps with moderate Δt converge quickly. Built inline in
Step 6.7 with n = 400 and `rhs="ramp"` (ones and sine break down early at c = 0): steps to
rtol = 1e-10 ≈ 22.4·√Δt/Δs for Δt/Δs² from 1 to 300 (23 to 377 steps), 5–13 steps below
that, and capped at n = 400 from Δt/Δs² = 1000. Rows of (I + Δt·L)⁻¹ fall by a factor e every
√Δt/Δs cells.

**`shift_n`: the worst case.** The cyclic shift, `A e_j = e_{j+1}` and `A e_n = e_1`, with
`b = e_1`. A permutation matrix: orthogonal, condition number 1. GMRES still makes **no
progress at all** until step n: `K_k = span{e_1, …, e_k}`, `A K_k = span{e_2, …, e_{k+1}}`, and
that is perpendicular to b, so the best residual is b itself (confirmed at n = 8: residual
exactly 1 for k = 1–7, 0 at k = 8). Its eigenvalues are the n-th roots of unity, and no
polynomial with `q(0) = 1` can be small on all of them until its degree reaches n. Shows that
the condition number alone doesn't predict GMRES, and that GMRES(m) with m < n never converges.

**`eig_rhs`: a lucky breakdown.** `A = [[2, 1, 0], [0, 3, 1], [0, 0, 4]]`, `b = e_1`, so
`Ab = 2b`. The first Arnoldi step finds `A v_1` already in `span{v_1}`, `h_21 = 0`, and GMRES
returns the exact answer `x = b/2` after one step. Shows that breakdown is usually good news.
`gmres_v3` already handles it correctly.

**`singular3`: a bad breakdown.** `A = diag(1, 2, 0)`, `b = (1, 1, 1)`. A squashes everything
onto the xy-plane, so `Ax = b` has no solution: the z-component of b is out of reach. The best
possible residual is `(0, 0, 1)`, of norm 1. GMRES's residual is 1.095 at step 1 and 1.0 at
step 2. At step 3 the Krylov space is all of R³, Arnoldi breaks down, and `H_3` is singular to
round-off, so back-substitution meets a zero pivot. `gmres_v3` reports convergence with
residual 0 and returns x₃ = −1.47e16 (true residual 2.24). `singular3(rhs="in_range")` keeps
A and takes `b = (1, 1, 0)`, inside the range of A: a lucky breakdown at k = 2 with the exact
answer (1, 1/2, 0), showing that a singular A alone doesn't make a breakdown bad.

**`nilpotent2`: failure on the first step.** `A = [[0, 1], [0, 0]]`, `b = e_1`. Then `Ab = 0`,
so `h_11 = h_21 = 0` and the pivot is zero before anything has happened. The system *does* have
a solution, `x = e_2`, but it isn't in `span{b} = span{e_1}`, so GMRES cannot find it. Shows
that a zero-pivot breakdown is a property of the Krylov space, not only of whether a solution
exists. `gmres_v3` reports convergence and returns `x = (inf, nan)`.

**`embree3`: the restart paradox.** From Embree, "The tortoise and the hare restart GMRES"
(SIAM Review 45(2), 2003). `A = [[1, 1, 1], [0, 1, 3], [0, 0, 1]]`, `b = (2, −4, 1)`.
Confirmed: GMRES(1) reaches the exact solution after 3 cycles (residual 4.58 → 4.24 → 3.00 → 0),
while GMRES(2) drops to 1.73 and stalls (1.7285, 1.7256, 1.7254, …). Full GMRES finishes in
3 steps, as on any 3×3.

**`two_scale`: unknowns at different scales.** Two coupled fields on n cells, so 2n×2n,
standing in for the simulation's `Te` (order 1) and whistler energies (order 1e-10). Built from
a well-behaved block matrix (e.g. two `convdiff_n` blocks with weak off-diagonal coupling) and
a true solution whose second field is ~1e-10 times the first, with `b = A x_true`. GMRES
minimises the 2-norm of the whole residual, which the first field's rows dominate, so it
"converges" while the second field is still wrong by 100%. Rescaling each field to order 1
(diagonal scaling, the simplest preconditioner) fixes it. Mirrors the argument in
`convergence.py` and the operator scaling in `newton_krylov.py: _gmres_solve`.

---

# Part I — A correct GMRES (notebooks 1–5)

## Step 0. The problem

**Notebook:** `GMRES_1_krylov.ipynb` (done)

**As built:** states `Ax = b` with A nonsingular and available only as `matvec: v -> Av`,
with a 2×2 example of what the solver sees. Explains why: in the simulation A is the Jacobian
`J = ∂F/∂u` of an implicit time step, `Jv` is a directional derivative costing about one
evaluation of F (`jax.jvp`), while forming J takes n of those and a dense LU costs `O(n³)` work
and `O(n²)` memory. Defines the residual `r_k = b − A x_k` as the measurable target and the stopping
rule `‖r_k‖₂ ≤ rtol·‖b‖₂`. Cost is counted in matvecs.

**Hands off:** the only thing the solver can do is apply A to vectors it chooses (Step 1).

## Step 1. The bare idea (`gmres_v0`)

**Notebook:** `GMRES_1_krylov.ipynb` (done)

**Starts from:** the solver can only apply A to vectors.

**Builds:** `CountingMatvec` (also in `tutorial_utils.py`), `gmres_v0` (forms
`K_k = [b, Ab, …, A^{k−1}b]` literally and solves `min_c ‖b − (A K_k) c‖₂` with `lstsq`),
`krylov_matrix`, `angle_degrees`.

**As built:**
- 1.1 Which vectors can the solver build: the Krylov space; the example matrix is `geo3`.
- 1.2 Krylov vectors are polynomials in A: every element is `p(A) b` with deg p < k. Cayley–
  Hamilton puts `A⁻¹b` in `K_n`, so GMRES finishes in at most n steps.
- 1.3 The GMRES rule: the smallest residual in `K_k`.
- 1.4 What the least-squares solution looks like geometrically.
- 1.5 `gmres_v0` as a loop. 1.6 a bigger problem: `diag_narrow`, Figure 1.1 (about 10× per
  step, rtol reached after 11 steps).
- 1.7 Where it breaks: `diag_spread`, Figure 1.2. cond(K_k) reaches about 3e15 at k = 5,
  essentially 1/ε, and from there the residual creeps back up to about 0.95 and stays,
  although theory says it never increases and is 0 by k = 20. The cause is power iteration:
  `A^j b` turns toward the dominant eigenvector, and the printed angle between successive
  Krylov vectors shrinks toward 0.

**Ties to code:** none yet; this is the idea `gmres.py` implements.

**Hands off:** the search space is right, the basis describing it is not (Step 2).

## Step 2. Arnoldi: an orthonormal Krylov basis (`gmres_v1`) — *same math, computed better*

**Notebook:** `GMRES_2_arnoldi_orthogonalise.ipynb` (done). Loads `krylov_matrix`, `gmres_v0`
from notebook 1.

**Starts from:** `gmres_v0`'s basis goes parallel.

**Builds:** `normalise_remainder` (the shared tail, including production's breakdown test),
three Arnoldi step functions with the same arguments,
`arnoldi_step_classical_gram_schmidt`, `arnoldi_step_modified_gram_schmidt` and
`arnoldi_step_classical_gram_schmidt_twice`, the driver `arnoldi` (default step: classical
Gram–Schmidt applied twice), and `gmres_v1` (the same basis written inline, still `lstsq` on
`A V_k`).

**As built:**
- 2.1 Same space, better basis: any basis of `K_k` gives the same iterate; only its
  conditioning changes.
- 2.2 The Arnoldi process, written neutrally (compute all components, subtract them). Why
  orthonormalising afterwards is too late: in `A^j b` the small-eigenvalue components are
  already below round-off.
- 2.3 Two steps on `geo3` by hand with one classical pass, and what a second pass would remove
  (2.3e-15, 3.1e-16).
- 2.4 Gram–Schmidt in floating point. Conventions (0-based, arrays preallocated with zeros and
  filled in place; `H̄_k` as a record of what Arnoldi computed, given its meaning in Step 3).
  The default reproduces the by-hand V₃ and H̄₂ to 1.8e-15 and 3.3e-15. `geo3`'s third step is
  a flagged breakdown: the remainder after two passes is 2.0e-31 against the threshold
  ε‖A v₃‖ = 3.8e-16, so h₄₃ = 0 exactly (the test is explained in Step 5). Figure 2.1 compares
  `‖I − V_kᵀV_k‖₂` for the three methods on `diag_spread` at 4 and 12 decades. A table follows
  each method on the 12-decade matrix (loss of V_j, error of the new vector, cancellation
  `‖Av_j‖/h_{j+1,j}` ≈ 6.6) and explains why the one-pass loss compounds by 4–5× per step and
  why "twice is enough" (Kahan, Parlett).
- 2.5 `gmres_v1`, which writes the two passes inline because the step function does its own
  matvec and doesn't return `A v_k`. Figure 2.2: v0 against v1 on `diag_spread`.

**Confirmed numbers:**
- `diag_spread` (4 decades): `gmres_v1` reaches 2.8e-13 at k = 20 and never rises.
  ‖x_v0 − x_v1‖/‖x_v1‖ is 3e-16 for k ≤ 2, 3.6e-12 at k = 3, 1.1e-7 at k = 4 and O(1) from
  k = 5, always under cond(K_k)·ε. `‖I − V_kᵀV_k‖₂` ≤ 3.7e-16 for k ≤ 20 with two passes; one
  pass at k = 20 gives 6.6e-12 (classical) and 5.0e-12 (modified). k = 20 = n is a flagged
  breakdown.
- `diag_spread(20, decades=12)`: `‖I − V₂₀ᵀV₂₀‖₂` is 1.1 (classical, one pass), 5.4e-6
  (modified), 3.6e-16 (twice). From a clean basis one pass leaves ≤ 1.8e-15, two passes
  ≤ 6.4e-17.

**Ties to code:** `arnoldi.py: arnoldi_step` is `arnoldi_step_classical_gram_schmidt_twice`
(`project`/`combine` are the two products, `first_pass + second_pass` the accumulated h), apart
from what JAX requires (rows instead of columns, a returned basis, whole-basis products; Step 9).

**Hands off:** `gmres_v1` still solves an n×k problem every step and throws away the `h_ij`
(Step 3).

## Step 3. The Arnoldi relation: a smaller least-squares problem (`gmres_v2`) — *same math, computed better*

**Notebook:** `GMRES_3_arnoldi_least_squares.ipynb` (done). Loads `gmres_v1`, `arnoldi`, the
three step functions (the one-pass ones only for 3.3) from notebook 2, and `krylov_matrix`
from notebook 1.

**Starts from:** `gmres_v1` re-solves an n×k least-squares problem, stores `A V_k`, and
discards the `h_ij`.

**Builds:** `gmres_v2`, which solves `min_y ‖β e_1 − H̄_k y‖₂` and forms x_k once, at the end.

**As built:**
- 3.1 What the `h_ij` record: `geo3`'s first step rearranged as `A v_1 = h_11 v_1 + h_21 v_2`,
  then the Arnoldi relation `A V_k = V_{k+1} H̄_k`, and why `H̄` is upper Hessenberg. Explains
  `geo3`'s accidental `h_13 = 0`.
- 3.2 Shrinking the problem: `b = β v_1` and `V_{k+1}` preserves lengths, so
  `‖b − A V_k y‖₂ = ‖β e_1 − H̄_k y‖₂`. Figure 3.1: v1's residual, v2's small-problem residual
  and the true residual of v2's iterates coincide on `diag_spread`.
- 3.3 A look inside `H̄_k`. Figure 3.2: the Hessenberg shape filling in on `convdiff_n`
  (c = 20) after 2, 4, 8 and 16 steps. Figure 3.3: the relation error next to
  `‖I − V_kᵀV_k‖₂`, since the residual equality needs both; with two passes both stay at
  round-off. Then the 12-decade `diag_spread` demonstration (table and Figure 3.4): with one
  pass the small problem's iterate stalls at 0.25 while `gmres_v1`'s problem on the same basis
  still reaches 5e-6. What fails is `V_{k+1}ᵀV_{k+1} = I`, not the basis.
- 3.4 Aside: symmetric A gives a tridiagonal `H̄` (Lanczos / MINRES), Figure 3.5 (`convdiff_n`
  c = 0 against c = 20, k = 8).

**Confirmed numbers:** v1 against v2 agree to ≤ 5.3e-13 on `diag_spread`; cond(H̄₂₀) = 1.0e4 =
cond(A). On `convdiff_n` (n = 20), the largest entry of `H̄_8` above the superdiagonal, relative
to its largest entry, is 1.7e-2 for c = 20 (upper triangle filled) and 6.7e-17 for c = 0
(tridiagonal); with c = 0 and b = ones `arnoldi` stops at a breakdown after 10 steps. On the 12-decade matrix the true residual at k = 20 is 0.25 (classical, one
pass), 8.7e-6 (modified), 1.5e-5 (twice), against a floor of ε‖A‖‖x‖/‖b‖ = 5.1e-5; the methods
agree to two digits up to k = 18. `gmres_v1`'s problem on the one-pass classical basis reaches
4.9e-6.

**Ties to code:** `gmres.py: _arnoldi_steps` is the loop of `gmres_v2`, calling `arnoldi_step`
on a basis allocated once with `restart + 1` zero rows (`vector_ops.py: empty_basis`).
Production never forms `A V_k` and never calls a least-squares solver; it uses `givens.py`
(Step 4).

**Hands off:** `gmres_v2` re-solves the whole small problem each step just to learn the
residual norm, and each step costs more than the last (Step 4).

## Step 4. Givens rotations: the residual for free (`gmres_v3`) — *same math, computed better*

**Notebook:** `GMRES_4_givens.ipynb` (done). Loads `gmres_v2` from notebook 3 and
`arnoldi_step_classical_gram_schmidt_twice`, `arnoldi` from notebook 2.

**Starts from:** `H̄_k` is `H̄_{k−1}` plus one column, but v2 solves from scratch.

**Builds:** `givens_rotation(top, bottom) -> (radius, cosine, sine)` (the identity when both
are zero, as production's `_rotation`), `givens_matrix` (for display only),
`back_substitution`, and `gmres_v3`: rotations stored as a list of `(c_j, s_j)` pairs, `H̄`
never stored, stop when `|g_{k+1}| ≤ max(rtol·‖b‖, atol)`, back-substitute once at the end.
Adds `atol`.

**As built:**
- 4.1 A least-squares problem doesn't change under an orthogonal matrix. With
  `Ω_k H̄_k = R̄_k` upper triangular and `g = Ω_k β e_1`, `‖r_k‖₂ = |g_{k+1}|` and
  `R_k y = g_{1:k}` (checked with `np.linalg.qr`).
- 4.2 One Givens rotation: `c = p/ρ`, `s = q/ρ`, `ρ = hypot(p, q)`, Figure 4.1.
- 4.3 Triangularising `H̄_k`, one rotation per column: `geo3`'s `H̄₃` by hand, then Figure 4.2
  (`convdiff_n`'s `H̄₆` after each rotation).
- 4.4 One column at a time: a new column takes all old rotations and one new one; earlier
  columns of R and entries of g never change again. `‖r_k‖₂ = |s_k|·‖r_{k−1}‖₂`.
- 4.5 `gmres_v3`, checked against v2 with `compare_versions`.
- 4.6 How far down the residual can go, Figure 4.3: `|g_{k+1}|`, v2's small-problem residual
  and the true residual agree down to round-off. Below about 1e-15 `|g_{k+1}|` keeps falling
  (a product of sines) while the true residual doesn't, so a tolerance set there is met on
  paper only.
- 4.7 What `|s_k|` says, Figure 4.4 (`diag_narrow`, `convdiff_n` c = 20, `shift_n` n = 8),
  read off the record as `‖r_k‖/‖r_{k−1}‖` (no new record key). `shift_n` previewed with the
  `A K_k ⟂ b` argument.
- 4.8 What it saves, Figure 4.5: time per step of `lstsq` against the Givens update on
  `convdiff_n` (n = 400, 200 steps). Order k³ against order k per step.
- 4.9 Production's form: the rotations kept as one matrix `Ω_k`, updated by rotating two of
  its rows and applied with one matrix-vector product, because on a GPU each separate
  operation costs a launch. Checked against v3: residuals to 9.9e-17, x₂₀ to 4.3e-16.

**Confirmed numbers:**
- `geo3`: s₁ = 0.1400, s₂ = 0.3845, s₃ = 0 exactly (flagged breakdown); R₃ diagonal 3.3665,
  2.2524, 1.7144; residuals 1.732 → 0.2425 → 0.0933 → 0; x = (3, 4, 5)/13.
- v2 against v3: ≤ 8.9e-14 on `diag_spread`, about 1e-14 or less on `geo3`, `diag_narrow` and
  `convdiff_n` (c = 20).
- `diag_spread` and `convdiff_n` (c = 20) end in flagged breakdowns at k = n: `|g_{n+1}| = 0`
  exactly, while the true residual is 1.5e-13 on `diag_spread`.
- `diag_narrow` with rtol = 0: `|g_{k+1}|` reaches 1.6e-20 at k = 20 while the true residual
  stays at 1.6e-16 (v2's `lstsq` residual about 1e-15). rtol = 1e-16 stops after 17 steps with
  `|g|` = 1.6e-17 and a true residual of 1.6e-16.
- `|s_k|`: `diag_narrow` 0.10–0.12; `convdiff_n` (c = 20) ≈ 0.96 for k ≤ 10 (residual still
  0.67‖b‖ at k = 10), 0 at k = 20; `shift_n` (n = 8) exactly 1 for k = 1–7, 0 at k = 8.
- Small-problem time per step (`convdiff_n`, n = 400), `lstsq` over Givens: ≈ 3 at k = 10,
  30–40 at k = 200 (varies run to run). 200 two-pass Arnoldi steps run without a breakdown.

**Ties to code:** `givens.py`: `GivensFactor` (`rotation` = Ω_k, `triangular` = R,
`rhs` = g), `start`, `fold` (one step of 4.4 in the product form of 4.9), `residual_estimate`
(`|g_{k+1}|`), `solve` (back-substitution via `solve_triangular`). `gmres.py: gmres` sets
`tolerance = max(rtol * norm(rhs), atol)`. Production's triangular factor starts as the
identity and `solve` masks unused steps (fixed shapes, Step 9). Production also checks each
new pivot `R_kk` before counting a step (Step 5).

**Hands off:** "residual zero exactly when s_k = 0" fails when `R_k` is singular (Step 5).

## Step 5. Breakdown (`gmres_v4`) — *same math, computed safely*

**Notebook:** `GMRES_5_breakdown.ipynb` (done). Loads `gmres_v3`, `givens_rotation`,
`back_substitution` from notebook 4 and the three Arnoldi step functions and `arnoldi` from
notebook 2.

**Starts from:** Arnoldi can produce a zero new vector (`h_{k+1,k} = 0`). Steps 2–4 flagged
several, all harmless; Step 4's "residual zero exactly when s_k = 0" assumed `R_k` invertible.

**Builds:** `eigenvalues_touched_by` and `rotation_pairs` (helpers for looking only),
`gmres_v4` (production's two checks, record keys `breakdown` and `residual_norm`), and, for
5.9 only, `gmres_v4_relative_pivot`.

**As built:** organised as good breakdowns against bad ones, with the rotation pair
`(h̃_kk, h_{k+1,k})` that builds G_k discussed in every case, and a map of which matrices give
which.
- 5.1 `gmres_v3` on `eig_rhs`, `singular3`, `nilpotent2`: all three report convergence with
  residual 0; only `eig_rhs` is right. Why it matters: Newton applies even a failed correction
  (`newton_krylov.py`).
- 5.2 What a breakdown is (K_k invariant, `A V_k = V_k H_k`) and when: at k = d, the degree of
  b's minimal polynomial μ_b; for diagonalisable A, the number of distinct eigenvalues b
  *touches*. H_d's eigenvalues are the roots of μ_b (checked on `eig_rhs`, `convdiff_n` c = 0,
  `singular3`). Step 1.2's Cayley–Hamilton argument with μ_b: μ_b(0) ≠ 0 ⇒ x_d exact (even for
  singular A); μ_b(0) = 0 ⇒ no x in K_d solves it. At a breakdown the pair is (h̃_kk, 0),
  R_kk = |h̃_kk|, and H_k singular ⇔ h̃_kk = 0. A 2×2 map of the pair: ordinary step, good
  breakdown, stagnation (0, ≠0; `shift_n`, not a breakdown), bad breakdown.
- 5.3 Good breakdowns. Nonsingular A ⇒ always good (Saad, Prop. 6.10, one-line proof). Kinds:
  b an eigenvector (`eig_rhs`, d = 1), b sharing a symmetry (`convdiff_n` c = 0 ones, d = 10:
  grid reversal), repeated eigenvalues (diag(1,1,2,2,3,3) built inline, d = 3), nothing
  special (d = n), singular A with b in its range (`singular3(rhs="in_range")`, d = 2).
  Figure 5.1: residual histories dropping to exactly 0 at k = d.
- 5.4 Bad breakdowns. Pair (0, 0): the smallest residual over K_k is |g_k| = ‖r_{k−1}‖, so the
  step makes no progress and x_{k−1} is the best GMRES can find. Cause: K_d contains a null
  vector of A. `singular3` (no solution; x₂ already optimal; round-off pivot) and
  `nilpotent2` (solution exists but unreachable; exact zero pivot). Figure 5.2: `singular3`
  residuals on a linear axis; Figure 5.3: R_k heatmaps for `geo3`, `singular3`, `nilpotent2`.
- 5.5 Table: kind of A × kind of b → breakdown step, pair, good/bad, example.
- 5.6 The breakdown test: relative, needs two passes, `not` so NaN counts. Missed lucky
  breakdowns (the `sine` case) are harmless.
- 5.7 `gmres_v4`: check 1 (a step with a non-finite column or `R_kk = 0` is not used), check 2
  (after a breakdown or a non-finite candidate, the true residual decides between the
  candidate and x_0). Checked against v3 with `compare_versions`.
- 5.8 The `converged` × `breakdown` table, each cell run live, and how `newton_krylov.py`
  counts failures and breakdowns.
- 5.9 A relative pivot test, `R_kk > ε‖A v_k‖`, compared with v4 on every example.

**Confirmed numbers:**
- Rotation pair at the breakdown: `eig_rhs` (2, 0) at k = 1; `convdiff_n` c = 0 ones
  (623.7, 0) at k = 10; diag(1,1,2,2,3,3) (1.686, 0) at k = 3; `geo3` (1.714, 0) at k = 3;
  `diag_spread` (1.872, 0) and `convdiff_n` c = 20 (1077, 0) at k = 20; `singular3` in range
  (1.265, 0) at k = 2; `singular3` (−1.67e-16, 0) at k = 3; `nilpotent2` exactly (0, 0) at
  k = 1. Every good case is exact to ≤ 1.5e-13‖b‖.
- `singular3` under v3, stopped after k = 1, 2, 3 steps: true residuals 1.0954, 1.0000, 2.2361;
  x₂ = (1, 0.5, 1.5); `lstsq` on H̄₃ gives the minimum 1.0000.
- Remainder at a true breakdown along each method's own run, in units of ε‖A v_j‖: `geo3`
  step 3, 11.7 (classical, one pass) and 13.0 (modified), with loss of orthogonality 2.9e-15
  and 3.1e-15; `convdiff_n` c = 0 step 10, 23.3 and 26.9, loss 8.5e-15 and 1.1e-14. Two
  passes are flagged at both, with loss 3.3e-16 and 6.7e-16. (From a clean two-pass basis
  instead, one pass misses only `geo3`'s classical step, at 1.2 ε.)
- v4: `eig_rhs` 1 iteration, 2 matvecs; `geo3` 3, 4; `singular3` in range 2, 3; `singular3`
  3 iterations, 4 matvecs, returns x = 0 with residual_norm 1.73 (converged False, breakdown
  True); `nilpotent2` 0 iterations, 2 matvecs, x = 0, residual_norm 1.
- `compare_versions(gmres_v3, gmres_v4)` on `geo3`, `diag_narrow` (k ≤ 11), `diag_spread`,
  `convdiff_n` c = 20: x and residual history identical (difference exactly 0); matvecs differ
  only at the breakdown k = n, by one. With rtol = 0, v4 reports `diag_spread`'s breakdown as
  not converged (true residual 6.9e-13 = 1.5e-13‖b‖).
- Relative pivot test: identical to v4 on all 13 examples except `singular3`, where it returns
  x₂ (residual 1.0, 2 iterations, 4 matvecs). Smallest R_kk/‖A v_k‖ over every step of the
  good examples is 0.155 (12-decade `diag_spread`); `singular3`'s round-off pivot is 0.65 ε.

**Ties to code:** `arnoldi.py` (breakdown flag, zero row, NaN), `givens.py` (`_rotation`'s
identity, `fold`'s docstring on a zero pivot), `gmres.py: _arnoldi_steps` (`usable`, the
`jnp.where` revert, `steps + usable`), `_restart_cycle` (`broke_down`, `verified` against
`trusted`), the `GMRESDiagnostics` docstring, `newton_krylov.py` (`gmres_failures`,
`gmres_breakdowns`). Files are named without relative links into the sibling repo.

**Hands off (end of Part I):** `gmres_v4` is a complete, correct GMRES: same iterates as
v0 in exact arithmetic, accurate in floating point, cheap per step, and honest at a
breakdown. What it doesn't control is how many steps it needs (Step 6).

---

# Part II — A fast GMRES (notebooks 6–8)

## Step 6. Why and when GMRES converges — *analysis, no new version*

**Notebook:** `GMRES_6_convergence.ipynb` (done). Loads `gmres_v4` from notebook 5.

**Starts from:** Step 4's `|s_k|`: about 0.1 per step on `diag_narrow`, about 0.96 on
`convdiff_n`, exactly 1 on `shift_n`. What decides it?

**Builds:** `residual_polynomial(A, b, k, *, solver=gmres_v4, **solver_options)`, a helper
for looking only (no solver version): it runs the solver for k steps, forms r_k with A, and
fits q_k through the heights r_k[i]/b_i at the eigenvalues of a diagonal A. Self-contained,
so Steps 7–8 can load it with `import_from_notebook` and pass `solver=gmres_v5, restart=m`.
In `tutorial_utils.py`: `diag_clustered`, `plot_residual_polynomial` (the dots-and-curve
picture), and `convdiff_n`'s default c = 20.

**The central picture ("dots and curve"):** `r_k = q_k(A) b` with deg q_k ≤ k and
`q_k(0) = 1`; the eigenvalues are dots, q_k is a curve held at 1 at z = 0, and GMRES picks
the lowest curve on the dots. No eigenvalue bound with κ(X), no Ritz or harmonic Ritz values.

**As built:**
- Opening: Figure 6.1 and a table of n, cond(A), steps to 1e-10 and the mean factor per step
  over k = 1..7 for `diag_narrow`, `diag_spread`, `convdiff_n`, `shift_n` (grey, dashed, as
  the reference worst case). Neither n nor cond(A) orders them.
- 6.1 The residual is a curve through (0, 1): derivation from Step 1.2, q(0) = 1 in words; for
  diagonal A, `r_k[i] = q_k(λ_i) b_i` and ‖r_k‖/‖b‖ is the RMS height of the dots. Figure 6.2
  (q_k for k = 1, 2, 3 on `diag_narrow`, linear and |q| on a log axis). Check: RMS against
  `residual_history`, and |q_k(0) − 1| for k ≤ 8.
- 6.2 GMRES beats any curve you can draw: the hand curve (1 − z/1.25)^k, the bound
  ‖r_k‖ ≤ max|q(λ_i)|·‖b‖ for diagonal A, Richardson as an aside. Figure 6.3. Shifted bunches
  [a, a + 0.5] for a = 0.1, 1, 10, Figure 6.4. Rule of thumb: width against distance from 0.
  Scaling A leaves residuals unchanged (iterates divided by α, checked); shifting A changes
  convergence.
- 6.3 Dots near 0 are expensive: `diag_spread`, Figure 6.5 (log z axis, |q_k| for k = 2, 5, 8,
  and the history against 0.9998^k). GMRES removes the large isolated dots first;
  superlinear convergence; tie to Step 1.7's power iteration.
- 6.4 A few bunches take a few steps: `diag_clustered` at n = 30 and 3000, Figure 6.6; tie to
  Step 5.3's diag(1,1,2,2,3,3). Outliers, Figure 6.7.
- 6.5 A ring around 0: `shift_n`'s eigenvalues are the 8th roots of unity; the mean of q over
  them is q(0) = 1, and b = e₁ weighs every orthonormal eigenvector equally, so
  ‖r_k‖ ≥ ‖b‖ for k < 8. Figure 6.8 (complex plane with log|q| contours for a hand curve with
  roots on three dots, and the circle unrolled). Ties to Step 4.7 (A K_k ⟂ b) and Step 5's
  stagnation cell; preview of restarting with m < 8.
- 6.6 When the dots mislead (`convdiff_n`): the diagonal twin and the sweep over c, Figure 6.9.
  Grid language: A^j·ones lives within j cells of the ends, so ‖r_k‖ ≥ √(n − 2k); Figure 6.10
  (|A⁻¹| with the band of a degree-5 polynomial, the residual cell by cell, the history
  against √(1 − 2k/n) and √(1 − k/n)). Nonnormal introduced as a word. c = 42 in one
  paragraph.
- 6.7 The time-step knob: A = I + Δt·L, dots in (1, 1 + 4Δt/Δs²), steps ∝ √Δt/Δs.
  Figure 6.11 (steps against Δt/Δs², n = 400, `rhs="ramp"`), Figure 6.12 (|A⁻¹| at
  Δt/Δs² = 10 and 1000). Tie to production: `preconditioner.py` registers only `"none"`.
- 6.8 Why many steps hurt: about 4nj multiply-adds and j + 1 stored vectors in step j, about
  2nk² in total, counted (not timed) for the 377-step solve, Figure 6.13.

**Confirmed numbers:**
- Opening table: cond(A) 1.5, 1e4, 50.8, 1; steps to 1e-10: 11, 20, 20, 8; mean factor over
  k = 1..7: 0.105, 0.930, 0.964, 1.000.
- 6.1 (`diag_narrow`): RMS heights 0.116, 0.0123, 0.00127 for k = 1, 2, 3, equal to
  `residual_history` to ≤ 3e-17 for k ≤ 8. |q_k(0) − 1| ≤ 3e-10 for k ≤ 8, 1.0e-3 at k = 14
  (so curves are drawn only for k ≤ 8; ≤ 6 for clusters, ≤ 4 for bunches far from 0). With
  warnings as errors, no `RankWarning` at the k used.
- 6.2: GMRES 0.116 then 0.102–0.106 per step, 11 steps to 1e-10; the hand curve's bound 0.2^k
  needs 15, its actual residual 14 (factor 0.117 → 0.19). GMRES's roots at k = 5: 1.022,
  1.118, 1.255, 1.389, 1.480; largest height on the dots 4.2e-5 against the hand curve's
  3.2e-4 (RMS 1.3e-5 against 1.0e-4). Shifted bunches: a = 0.1, 27 steps (0.42 per step);
  a = 1, 11 (0.10); a = 10, 6 (0.013). 7.3·A: residuals equal to 1.4e-15 relative, iterates
  divided by 7.3 to 3.7e-16.
- 6.3 (`diag_spread`): k = 1 root at 6189; a root on 10⁴ from k = 5; at k = 8 roots on 2336,
  3793, 6158, 10⁴ with q₈(1) = 0.98. Factor per step ≈ 0.94 for k = 4–6, 0.35 on the last
  step before the breakdown at k = 20.
- 6.4: `diag_clustered` 1.8e-4, 3.3e-8, 5.9e-12 at k = 3, 6, 9 (n = 30); 1.7e-4, 2.8e-8,
  4.6e-12 at n = 3000; 9 steps to 1e-10 at both. Bunch [1, 1.5] alone 11 steps; with 5, 20,
  100: 14; with 0.01: 14.
- 6.5 (`shift_n`): residual exactly 1 for k ≤ 7, 0 at k = 8. Hand curve with roots at ω₀, ω₁,
  ω₇: heights 0, 0, 2, 5.23, 6.83, 5.23, 2, 0; mean 1.000; RMS 3.70.
- 6.6 (`convdiff_n`, n = 20): at c = 20 the eigenvalues are real (imaginary parts exactly 0),
  λ_max/λ_min = 14.3, cond(A) = 50.8. Twin against `convdiff_n`: 0.076 against 0.836 at k = 5,
  0.006 against 0.670 at k = 10. Sweep c = 10, 20, 30: spread 49.5, 14.3, 5.49; cond 89.1,
  50.8, 35.4; ‖r₁₀‖/‖b‖ 0.604, 0.670, 0.693 (the earlier check: 1.86 and 0.705 at c = 40,
  where `eigvals` is off by 1e-2, so the notebook stops at 30). At c = 0 the b = ones twin
  also separates (0.567 against 0.949 at k = 1), because b touches only 10 eigenvectors.
  A^j·ones nonzero on cells 1..j and n−j+1..n at c = 0 and 20; middle cells of r₅ exactly 1;
  ‖r_k‖/√(n − 2k) is 1.000 for all k ≤ 9 at c = 0 and 1.01 → 2.23 at c = 20. c = 20 against
  √(1 − k/n): 0.836/0.866 at k = 5, 0.670/0.707 at k = 10; mean factor over 10 steps 0.961
  against 0.966. c = 42: super-diagonal 5.7e-14, diagonal 882, residual equal to √(1 − k/n)
  to 1e-15.
- 6.7 (n = 400, `rhs="ramp"`, rtol = 1e-10): Δt/Δs² = 0.01, 0.03, 0.1, 0.3 → 5, 6, 9, 13
  steps; 1, 3, 10, 30, 100, 300 → 23, 39, 70, 123, 226, 377; 1000 and above → 400 (the cap,
  flagged as a breakdown at k = n, converged). C = 22.4 fitted over 1 ≤ Δt/Δs² ≤ 300, against
  ln(2/rtol) = 23.7. No early breakdown. Rows of A⁻¹ fall by e every 3.18 and 31.62 cells at
  Δt/Δs² = 10 and 1000 (√Δt/Δs = 3.16, 31.62).
- 6.8 (Δt/Δs² = 300, 377 steps): orthogonalisation 1.14e8 multiply-adds against 4.52e5 for
  the stencil matvecs (×252); basis 151,200 numbers against 1,200.

**Ties to code:** no production changes. `gmres.py: gmres` has the two knobs Steps 7 and 8
explain, `restart` and `preconditioner`; `preconditioner.py` registers only `"none"`, so
today production relies on the time step alone.

**Hands off:** convergence can take many steps (eigenvalues near or around 0, or a stencil
matrix clearing one cell per step), and many steps cost about 2nk² work and (k + 1)n memory
(Step 7). Where the eigenvalues sit is what Step 8 changes.

**Proposed for the author, not applied:** drop Step 7's Elman bound and field-of-values plot,
so that Part II stays in the dots-and-curve picture: restarting as a product of degree-m
curves, each held at (0, 1). Steps 7–8 can load `residual_polynomial` from notebook 6 to
draw them.

## Step 7. Restarting: GMRES(m) (`gmres_v5`) — *changes the math*

**Notebook:** `GMRES_7_restart.ipynb` (planned). Loads `gmres_v4` from notebook 5.

**Starts from:** memory and per-step cost grow with k (Step 6).

**Builds:** `gmres_v5`: outer cycles, an inner Arnoldi of at most m steps, the true residual
recomputed at each restart, the iteration cap summed over cycles. Adds `restart=None` (no
restart) and the `cycles` record key. Step 5's acceptance rule becomes "no worse than the
start of the cycle". Sanity check: v5 with `restart ≥ max_iterations` equals v4.

**Derivations to include:**
- Cost: step j costs `O(n·j)` to orthogonalise, so a cycle costs `O(n·m²)`; memory `O(n·m)`
- Restarting = a new low-degree polynomial each cycle; the residual polynomial becomes a
  product of degree-m pieces instead of one degree-km polynomial
- The residual never increases across cycles, but it can stagnate
- A convergence guarantee when the field of values excludes 0 (Elman bound, stated)
- Matvec count `iterations + cycles − 1`: the first cycle starts from b for free, each restart
  pays one

**Examples:** a larger `convdiff_n` (cost and several m), `shift_n` (m < n: flat forever),
`embree3` (new builder).

**Plots to include:**
- Time (or flops) per iteration against k for full GMRES on a larger `convdiff_n`
- Residual histories for several m on one plot, with cycle boundaries marked
- `shift_n` with m < n
- `embree3`: GMRES(1) against GMRES(2)
- Field of values of a matrix where GMRES(1) stagnates

**Ties to code:** `gmres.py: gmres` (outer `while_loop`), `_restart_cycle`, `step_limit`.

**Hands off:** restarting caps the cost but can stall; the cause is the spectrum, so change it
(Step 8).

## Step 8. Preconditioning (`gmres_v6`, the full algorithm) — *changes the math*

**Notebook:** `GMRES_8_preconditioning.ipynb` (planned). Loads `gmres_v5` from notebook 7.

**Starts from:** Step 6 says convergence depends on the spectrum; Step 7 shows restarting
can't fix a bad one.

**Builds:** `gmres_v6`: solve `A M u = b` with `M ≈ A⁻¹`, then `x = M u` (right-preconditioned,
restarted, Givens, classical Gram–Schmidt applied twice, breakdown-checked). Signature
`gmres_v6(matvec, b, preconditioner=None, *, ...)` lines up with production's
`(operator, rhs, preconditioner, *, rtol, atol, restart, max_iterations)`, so
`solver(matvec, b, **kwargs)` still works. Sanity check: v6 with no preconditioner equals v5.
Note that `preconditioner.py` registers only `"none"` today.

**Derivations to include:**
- Right preconditioning: `b − A M u = b − A x`, so GMRES still minimises the true residual
- Left preconditioning minimises `‖M(b − Ax)‖` instead, a different norm
- Recovering x: `x = x_0 + M V_k y` (one preconditioner application per cycle)
- `M = A⁻¹` converges in one step; `M = I` is plain GMRES
- "A poor preconditioner costs iterations, never accuracy" (from `gmres.py`)
- Scaling as diagonal preconditioning: with `two_scale`, the 2-norm can't see the small field
- Jacobi preconditioning as the simplest example

**Examples:** `convdiff_n`, `two_scale` (new builder).

**Plots to include:**
- Eigenvalues of A against AM in the complex plane
- Residual histories with and without M
- Left against right: preconditioned residual and true residual for each
- Per-field residuals for `two_scale` before and after scaling

**Ties to code:** `gmres.py` (`preconditioned_operator`, `preconditioner(direction)`),
`preconditioner.py` (builder registry, `"none"`), `newton_krylov.py: _gmres_solve` (operator
scaling), `convergence.py` (why one norm can't cover fields at different scales).

**Hands off (end of Part II):** `gmres_v6` is the full algorithm in NumPy (Step 9).

### Appendix to Step 8: building a preconditioner from matvecs only

**Notebook:** `GMRES_8a_probing.ipynb` (proposed; or inside notebook 8, still open)

**Starts from:** a tridiagonal M is cheap to apply, but you only have `v -> Av`, and building A
column by column costs n matvecs.

**Derivations to include:**
- Probing: 3 vectors (cells `i % 3 == 0, 1, 2`) recover all three diagonals in 3 matvecs,
  whatever n (Curtis–Powell–Reid)
- Why 3 groups: the neighbours `i − 1, i, i + 1` always fall in different groups
- Block systems: 3 × (number of fields) matvecs
- What goes wrong when A is wider than tridiagonal

**Plots to include:**
- The probe vectors
- `A · probe`, coloured by which diagonal each entry came from
- Error of the recovered diagonals when A has extra couplings

**Ties to code:** `jacobian_blocks.py: tridiagonal_blocks`, `TridiagonalBlocks`. It is tested
but not yet registered as a preconditioner in `preconditioner.py`, so say so. SciPy is in the
`jaxsim` environment, so `scipy.linalg.solve_banded` can apply the tridiagonal M.

---

# Part III — The production code (notebook 9)

## Step 9. From NumPy to the production JAX code

**Notebook:** `GMRES_9_jax.ipynb` (planned). Loads `gmres_v6` from notebook 8.

**Starts from:** `gmres_v6` in NumPy. Each JAX constraint explains a design choice in
production. The notebook ends by running production `gmres` against `gmres_v6` on every entry
of `EXAMPLES`.

**Setup:** `jax.config.update("jax_enable_x64", True)`, as production's tests do;
`time_integration` is importable from the `jaxsim` kernel.

**Section: why production orthogonalises with two classical passes.** Time one Arnoldi
orthogonalisation against k basis vectors: modified Gram–Schmidt as a loop against two classical
passes as matrix products, in NumPy on the CPU and as jitted JAX on the GPU, for several n and
k. Two passes do about twice the arithmetic and win by replacing k dependent dot products with
two products (two kernel launches instead of k dependent ones; matrix-vector products instead
of vector operations). Also time production's own `project`/`combine`, which multiply the whole
`restart + 1`-row basis, zero rows included. Quick measurements on this machine, to set
expectations only (not for the notebook): NumPy on the CPU, two passes faster by 4–5× at
n = 1e4 and 2–3× at n = 1e6; JAX on the A100 in float64, modified Gram–Schmidt as a
`lax.fori_loop`, two passes faster by 2–14× at n = 1e4 but only 1.1–1.35× at n = 1e6
(memory-bound). Report what is measured, since the margin is not a fixed factor. `arnoldi.py`'s
docstring says the loop "would cost one host round trip per basis vector", which is not what a
`lax.fori_loop` does, so describe what is measured (dependent kernel launches) rather than
repeat the docstring.

**Topics to include:**
- The operator is just a function: dense `A @ v`, a stencil function and `jax.jvp` of a
  nonlinear F all give identical iterates
- `jax.jacfwd(op)(x)` to look at a function operator as a matrix
- Immutable arrays → `.at[i].set(...)` (`vector_ops.store`); `arnoldi_step` returns a new basis
  where the NumPy versions fill V in place
- Production stores basis vectors as rows, not columns
- Static shapes → a fixed `restart + 1`-row basis that starts at zero, so whole-basis products
  ignore unused rows (`project`, `combine`)
- Static shapes → the triangular factor starts as the identity; `givens.solve` masks unused
  entries
- The rotation product `Ω_k` instead of a list (built in Step 4.9)
- No `break` → `lax.while_loop` with `unfinished` predicates (outer and inner)
- No `if` on values → `lax.cond`, `jnp.where(usable, new, old)`
- Pytrees → `vector_ops` as the only way to touch a vector, so multi-field states never get
  flattened
- `GMRESDiagnostics` and `recompute_residual=True` (the true-residual recheck that Step 4.6
  motivates)

**Plots to include:**
- Comparison table: iterations, cycles, residual, flags for v6 against production on each
  example
- Timing: modified Gram–Schmidt loop against two classical passes, NumPy CPU and JAX GPU,
  against n and k
- Timing: NumPy v6 against jitted production as n grows (optional)

**Ties to code:** all of `gmres.py`, `vector_ops.py`, `givens.py`, `arnoldi.py`.
