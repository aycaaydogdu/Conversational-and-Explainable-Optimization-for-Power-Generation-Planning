"""Solver adapter registry.

Here we wrap every solver behind the same signature:  f(lp: LP, tolerans: float) -> Cozum
The resource planner (kaynak.py) picks an adapter using the properties stored here.

  highs        CPU, simplex, exact duals            (reference solver)
  ortools_glop CPU, simplex, exact duals            (Google OR-Tools)
  ortools_pdlp CPU, first-order (PDLP)              (Google OR-Tools, looser tolerance)
  cuopt        GPU, PDLP + batch solve              (NVIDIA cuOpt, needs a GPU)
"""
from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .model import LP, Cozum, coz_highs


# ---------------------------------------------------------------- OR-Tools
def _ortools(lp: LP, motor: str, tolerans: float) -> Cozum:
    """Here we rebuild the LP in OR-Tools and solve it with GLOP or PDLP."""
    from ortools.linear_solver import pywraplp

    s = pywraplp.Solver.CreateSolver(motor)
    inf = s.infinity()
    xs = [s.NumVar(float(l) if np.isfinite(l) else -inf, float(h) if np.isfinite(h) else inf, "")
          for l, h in zip(lp.col_lo, lp.col_hi)]
    A = lp.A.tocsr()
    satirlar = []
    for i in range(A.shape[0]):
        lo, hi = lp.row_lo[i], lp.row_hi[i]
        c = s.Constraint(float(lo) if np.isfinite(lo) else -inf, float(hi) if np.isfinite(hi) else inf)
        for j, v in zip(A.indices[A.indptr[i]:A.indptr[i + 1]], A.data[A.indptr[i]:A.indptr[i + 1]]):
            c.SetCoefficient(xs[j], float(v))
        satirlar.append(c)
    amac = s.Objective()
    for j, cj in enumerate(lp.c):
        if cj:
            amac.SetCoefficient(xs[j], float(cj))
    amac.SetMinimization()
    if motor == "PDLP":
        # Here we set PDLP tolerances; OR-Tools takes them as a text protobuf
        s.SetSolverSpecificParametersAsString(
            f"termination_criteria {{ simple_optimality_criteria {{ "
            f"eps_optimal_relative: {tolerans} eps_optimal_absolute: {tolerans} }} }}")
    t0 = time.perf_counter()
    st = s.Solve()
    sure = (time.perf_counter() - t0) * 1000
    ad = f"OR-Tools {motor}"
    if st == pywraplp.Solver.OPTIMAL or (motor == "PDLP" and st == pywraplp.Solver.FEASIBLE):
        x = np.array([v.solution_value() for v in xs])
        y = np.array([c.dual_value() for c in satirlar])
        d = np.array([v.reduced_cost() for v in xs])
        return Cozum("optimal", amac.Value(), x, y, d, ad, sure, int(s.iterations()))
    if st == pywraplp.Solver.INFEASIBLE:
        return Cozum("cozumsuz", None, None, None, None, ad, sure, 0)
    return Cozum(f"hata:{st}", None, None, None, None, ad, sure, 0)


def coz_glop(lp: LP, tolerans: float = 1e-9) -> Cozum:
    return _ortools(lp, "GLOP", tolerans)


def coz_pdlp(lp: LP, tolerans: float = 1e-4) -> Cozum:
    return _ortools(lp, "PDLP", tolerans)


# ---------------------------------------------------------------- cuOpt (GPU)
def _cuopt_model(lp: LP):
    """Here we convert our LP into a cuOpt DataModel (CSR matrix + bounds)."""
    from cuopt.linear_programming import data_model

    A = lp.A.tocsr()
    dm = data_model.DataModel()
    dm.set_csr_constraint_matrix(A.data.astype(np.float64), A.indices.astype(np.int32),
                                 A.indptr.astype(np.int32))
    dm.set_constraint_lower_bounds(lp.row_lo.astype(np.float64))
    dm.set_constraint_upper_bounds(lp.row_hi.astype(np.float64))
    dm.set_variable_lower_bounds(lp.col_lo.astype(np.float64))
    dm.set_variable_upper_bounds(lp.col_hi.astype(np.float64))
    dm.set_objective_coefficients(lp.c.astype(np.float64))
    return dm


def _cuopt_ayar(tolerans: float):
    """Here we build cuOpt solver settings (tolerance, quiet logging)."""
    from cuopt.linear_programming import solver_settings

    ss = solver_settings.SolverSettings()
    for ad in ("set_optimality_tolerance",):
        if hasattr(ss, ad):
            getattr(ss, ad)(tolerans)
    # Here we turn off console logging: every solve prints ~15 lines, which means
    # millions of lines for 100k scenarios and a very slow notebook.
    for anahtar, deger in (("log_to_console", False), ("CUOPT_LOG_TO_CONSOLE", False)):
        try:
            ss.set_parameter(anahtar, deger)
            break
        except Exception:
            continue
    return ss


def _cuopt_cevir(sol, sure, ad="cuOpt PDLP (GPU)") -> Cozum:
    """Here we convert a cuOpt solution back into our Cozum object."""
    durum = str(sol.get_termination_status()).lower() if hasattr(sol, "get_termination_status") else ""
    if "infeas" in durum:
        return Cozum("cozumsuz", None, None, None, None, ad, sure, 0)
    x = np.asarray(sol.get_primal_solution())
    y = np.asarray(sol.get_dual_solution())
    d = np.asarray(sol.get_reduced_cost()) if hasattr(sol, "get_reduced_cost") else None
    return Cozum("optimal", float(sol.get_primal_objective()), x, y, d, ad, sure, 0)


def coz_cuopt(lp: LP, tolerans: float = 1e-4) -> Cozum:
    """Here we solve one LP on the GPU with cuOpt (tested on Colab T4, cuOpt 26.6)."""
    from cuopt.linear_programming import solver

    t0 = time.perf_counter()
    sol = solver.Solve(_cuopt_model(lp), _cuopt_ayar(tolerans))
    return _cuopt_cevir(sol, (time.perf_counter() - t0) * 1000)


def coz_cuopt_toplu(lps: list[LP], tolerans: float = 1e-4) -> list[Cozum]:
    """Here we send many LPs to the GPU in one call (cuOpt BatchSolve).

    Note: measured on T4, cuOpt still solves each small LP separately, so this is
    not a true single-kernel batch.
    """
    from cuopt.linear_programming import solver

    t0 = time.perf_counter()
    sonuc = solver.BatchSolve([_cuopt_model(lp) for lp in lps], _cuopt_ayar(tolerans))
    sols = sonuc[0] if isinstance(sonuc, tuple) else sonuc
    sure = (time.perf_counter() - t0) * 1000 / max(1, len(lps))
    return [_cuopt_cevir(s, sure) for s in sols]


# ---------------------------------------------------------------- registry
@dataclass(frozen=True)
class Adaptor:
    ad: str
    fonk: Callable[[LP, float], Cozum]
    cihaz: str              # device: "cpu" | "gpu"
    kesin_dual: bool        # simplex-based: duals / reduced costs are exact
    en_iyi_tolerans: float  # tightest tolerance the solver reaches reliably
    toplu: Callable | None = None
    paket: str = ""


ADAPTORLER: dict[str, Adaptor] = {
    "highs": Adaptor("highs", lambda lp, t=1e-9: coz_highs(lp), "cpu", True, 1e-9, paket="highspy"),
    "ortools_glop": Adaptor("ortools_glop", coz_glop, "cpu", True, 1e-9, paket="ortools"),
    "ortools_pdlp": Adaptor("ortools_pdlp", coz_pdlp, "cpu", False, 1e-6, paket="ortools"),
    "cuopt": Adaptor("cuopt", coz_cuopt, "gpu", False, 1e-6, toplu=coz_cuopt_toplu, paket="cuopt"),
}


# Here we cache which solvers actually work on this machine
_KULLANILABILIR: dict[str, tuple[bool, str]] = {}


def kullanilabilir(a: Adaptor) -> tuple[bool, str]:
    """Here we check that the package is installed AND can really solve a small LP.

    find_spec alone is not enough: a library can be installed but crash on load
    (e.g. version clash, no GPU, incompatible CUDA driver). The result is cached.
    """
    if a.ad in _KULLANILABILIR:
        return _KULLANILABILIR[a.ad]
    if importlib.util.find_spec(a.paket) is None:
        sonuc = (False, "kurulu değil")
    else:
        try:
            from .model import kur_model, veri_yukle
            c = a.fonk(kur_model(veri_yukle()), max(1e-6, a.en_iyi_tolerans))
            sonuc = (c.durum == "optimal", "ok" if c.durum == "optimal" else c.durum)
        except Exception as e:  # ImportError, CUDA error, etc.
            sonuc = (False, f"{type(e).__name__}: {str(e)[:160]}")
    _KULLANILABILIR[a.ad] = sonuc
    return sonuc


def kurulu_mu(a: Adaptor) -> bool:
    return kullanilabilir(a)[0]
