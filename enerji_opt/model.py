"""Model builder and the HiGHS / scipy solver adapters.

Here we turn the input data into a solver-independent LP and solve it.

Flow:  data + changes  ->  kur_model()  ->  LP (solver-independent)
       LP  ->  coz_highs() / coz_scipy()  ->  Cozum (x, duals, status)

We keep the LP structure solver-independent on purpose: later cuOpt or our own
batched PDLP solver will take the same LP object and return the same Cozum object.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

VERI_YOLU = Path(__file__).with_name("veri_5bolge.json")
INF = float("inf")


def veri_yukle(yol: str | Path = VERI_YOLU) -> dict:
    with open(yol, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- what-if changes
def degisiklik_uygula(veri: dict, d: dict | None) -> dict:
    """Here we apply what-if changes to a copy of the data (the original is untouched).

    Supported keys:
      ruzgar_carpani: 0.7            -> scale all wind capacities by 0.7
      talep_carpani: 1.25            -> scale all demands by 1.25
      bolge_talep_carpani: {"MAR":1.1}  -> scale demand of one region
      hat_kapasite: {"DOG-ICA": 4500}  -> set a line limit (MW, absolute)
      hat_kapasite_degisim: {"DOG-ICA": 500}  -> add to a line limit (MW)
      santral_disi: ["MAR_DGKC"]     -> take a unit out (outage / maintenance)
      santral_kapasite: {"EGE_GT": 500}
      santral_maliyet: {"MAR_DGKC": 3000}
      zorla_min: {"EGE_GT": 300}     -> lower bound, used by the why-not counterfactual
    """
    v = copy.deepcopy(veri)
    if not d:
        return v
    # Here we reject unknown keys so a typo never silently does nothing
    bilinen = {"ruzgar_carpani", "talep_carpani", "bolge_talep_carpani", "hat_kapasite",
               "hat_kapasite_degisim", "santral_disi", "santral_kapasite",
               "santral_maliyet", "zorla_min"}
    bilinmeyen = set(d) - bilinen
    if bilinmeyen:
        raise ValueError(f"Bilinmeyen değişiklik anahtarı: {sorted(bilinmeyen)}")

    bolge_ids = {b["id"] for b in v["bolgeler"]}
    santraller = {s["id"]: s for s in v["santraller"]}
    hatlar = {h["id"]: h for h in v["hatlar"]}

    def kontrol(ids, sozluk, ad):
        for i in ids:
            if i not in sozluk:
                raise ValueError(f"Bilinmeyen {ad}: {i}")

    if "ruzgar_carpani" in d:
        for s in v["santraller"]:
            if s["tip"] == "rüzgâr":
                s["kapasite"] *= d["ruzgar_carpani"]
    if "talep_carpani" in d:
        for b in v["bolgeler"]:
            b["talep"] *= d["talep_carpani"]
    if "bolge_talep_carpani" in d:
        kontrol(d["bolge_talep_carpani"], bolge_ids, "bölge")
        for b in v["bolgeler"]:
            b["talep"] *= d["bolge_talep_carpani"].get(b["id"], 1.0)
    if "hat_kapasite" in d:
        kontrol(d["hat_kapasite"], hatlar, "hat")
        for k, val in d["hat_kapasite"].items():
            hatlar[k]["kapasite"] = val
    if "hat_kapasite_degisim" in d:
        kontrol(d["hat_kapasite_degisim"], hatlar, "hat")
        for k, val in d["hat_kapasite_degisim"].items():
            hatlar[k]["kapasite"] = max(0.0, hatlar[k]["kapasite"] + val)
    if "santral_disi" in d:
        kontrol(d["santral_disi"], santraller, "santral")
        for k in d["santral_disi"]:
            santraller[k]["kapasite"] = 0.0
    if "santral_kapasite" in d:
        kontrol(d["santral_kapasite"], santraller, "santral")
        for k, val in d["santral_kapasite"].items():
            santraller[k]["kapasite"] = val
    if "santral_maliyet" in d:
        kontrol(d["santral_maliyet"], santraller, "santral")
        for k, val in d["santral_maliyet"].items():
            santraller[k]["maliyet"] = val
    # Here we force a minimum output on a unit (why-not: "what if we ran it anyway?")
    if "zorla_min" in d:
        kontrol(d["zorla_min"], santraller, "santral")
        for k, val in d["zorla_min"].items():
            santraller[k]["min"] = val
    return v


# ---------------------------------------------------------------- LP representation
@dataclass
class LP:
    """Solver-independent LP:  min c'x  s.t.  row_lo <= A x <= row_hi,  col_lo <= x <= col_hi"""
    c: np.ndarray
    A: sp.csc_matrix
    row_lo: np.ndarray
    row_hi: np.ndarray
    col_lo: np.ndarray
    col_hi: np.ndarray
    col_ad: list[str]
    row_ad: list[str]
    meta: dict = field(default_factory=dict)  # variable/constraint groups -> column/row indices


def kur_model(veri: dict, onarim: dict | None = None) -> LP:
    """Here we build a lossless transport-type (NTC) economic dispatch LP.

    Decision variables:
      p_g  >= 0        generation of unit g          (MW)
      f_l  free        flow on line l, + means a -> b (MW)
    Constraints:
      regional balance: sum_g p_g + inflow - outflow = demand_r   (dual = regional price)
      capacity:         min_g <= p_g <= cap_g                     (column bound)
      line limit:       -cap_l <= f_l <= cap_l                    (column bound)

    onarim: None gives the plain model. If a dict is given, we add flexibility
    (repair) variables, e.g.
      {"soguk_yedek": True, "talep_katilimi": True, "hat_acil": True, "yuk_atma": True}
    """
    onarim = onarim or {}
    ok = veri.get("onarim_kaynaklari", {})
    bolgeler = veri["bolgeler"]
    r_idx = {b["id"]: i for i, b in enumerate(bolgeler)}
    R = len(bolgeler)

    c, lo, hi, ad = [], [], [], []
    rows, cols, vals = [], [], []
    meta: dict[str, list[int]] = {"uretim": [], "akis": [], "soguk_yedek": [],
                                  "talep_katilimi": [], "hat_acil": [], "yuk_atma": []}
    ek_satirlar = []  # extra rows: (lo, hi, name, [(col, val)])

    # Here we add one variable and remember which group it belongs to
    def degisken(isim, maliyet, l, h, grup):
        j = len(c)
        c.append(maliyet); lo.append(l); hi.append(h); ad.append(isim)
        meta[grup].append(j)
        return j

    # Here we add one generation variable per unit, feeding its region's balance row
    for s in veri["santraller"]:
        kap = float(s["kapasite"])
        mn = float(s.get("min", 0.0))
        j = degisken(s["id"], s["maliyet"], mn, kap, "uretim")
        rows.append(r_idx[s["bolge"]]); cols.append(j); vals.append(1.0)

    # Here we add one flow variable per line (leaves region a, enters region b)
    acil = onarim.get("hat_acil", False)
    acil_oran = ok.get("hat_acil_kapasite", {}).get("oran", 0.0)
    for h in veri["hatlar"]:
        kap = float(h["kapasite"])
        sinir = kap * (1 + acil_oran) if acil else kap
        j = degisken(h["id"], 0.0, -sinir, sinir, "akis")
        rows += [r_idx[h["a"]], r_idx[h["b"]]]; cols += [j, j]; vals += [-1.0, 1.0]
        if acil:
            e = degisken(h["id"] + "_ACIL", ok["hat_acil_kapasite"]["maliyet"], 0.0,
                         kap * acil_oran, "hat_acil")
            # Here we allow emergency rating e: f - e <= cap and -f - e <= cap  ->  |f| <= cap + e
            ek_satirlar.append((-INF, kap, h["id"] + "_ust", [(j, 1.0), (e, -1.0)]))
            ek_satirlar.append((-INF, kap, h["id"] + "_alt", [(j, -1.0), (e, -1.0)]))

    # Here we add the optional repair resources (only used by feasibility repair)
    if onarim.get("soguk_yedek"):
        for s in ok.get("soguk_yedek", []):
            j = degisken(s["id"], s["maliyet"], 0.0, s["kapasite"], "soguk_yedek")
            rows.append(r_idx[s["bolge"]]); cols.append(j); vals.append(1.0)
    if onarim.get("talep_katilimi"):
        tk = ok["talep_katilimi"]
        for b in bolgeler:
            j = degisken(b["id"] + "_TK", tk["maliyet"], 0.0, tk["oran"] * b["talep"],
                         "talep_katilimi")
            rows.append(r_idx[b["id"]]); cols.append(j); vals.append(1.0)
    if onarim.get("yuk_atma"):
        for b in bolgeler:
            j = degisken(b["id"] + "_YUKATMA", ok["yuk_atma"]["maliyet"], 0.0, b["talep"],
                         "yuk_atma")
            rows.append(r_idx[b["id"]]); cols.append(j); vals.append(1.0)

    # Here we set the balance rows equal to demand (row_lo == row_hi)
    row_lo = [b["talep"] for b in bolgeler]
    row_hi = [b["talep"] for b in bolgeler]
    row_ad = [f"denge_{b['id']}" for b in bolgeler]
    for k, (l, h, isim, giris) in enumerate(ek_satirlar):
        i = R + k
        row_lo.append(l); row_hi.append(h); row_ad.append(isim)
        for j, v in giris:
            rows.append(i); cols.append(j); vals.append(v)

    A = sp.csc_matrix((vals, (rows, cols)), shape=(len(row_lo), len(c)))
    meta["denge"] = list(range(R))
    return LP(np.array(c, float), A, np.array(row_lo, float), np.array(row_hi, float),
              np.array(lo, float), np.array(hi, float), ad, row_ad, meta)


# ---------------------------------------------------------------- solution
@dataclass
class Cozum:
    durum: str                 # "optimal" | "cozumsuz" (infeasible) | "sinirsiz" (unbounded) | "hata"
    amac: float | None
    x: np.ndarray | None
    row_dual: np.ndarray | None   # d(objective)/d(rhs)  -> regional price
    col_dual: np.ndarray | None   # reduced cost
    cozucu: str
    sure_ms: float
    iterasyon: int
    ek: dict = field(default_factory=dict)   # extras, e.g. sensitivity ranges


def coz_highs(lp: LP, yontem: str = "simplex", aralik: bool = False) -> Cozum:
    """Here we solve the LP with HiGHS (simplex by default) and return exact duals."""
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", yontem)
    m = highspy.HighsLp()
    m.num_col_ = len(lp.c)
    m.num_row_ = len(lp.row_lo)
    m.col_cost_ = lp.c
    m.col_lower_ = lp.col_lo
    m.col_upper_ = lp.col_hi
    m.row_lower_ = np.where(np.isinf(lp.row_lo), -highspy.kHighsInf, lp.row_lo)
    m.row_upper_ = np.where(np.isinf(lp.row_hi), highspy.kHighsInf, lp.row_hi)
    m.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    m.a_matrix_.start_ = lp.A.indptr
    m.a_matrix_.index_ = lp.A.indices
    m.a_matrix_.value_ = lp.A.data
    # Here we pass the column-wise sparse matrix straight to HiGHS
    h.passModel(m)
    t0 = time.perf_counter()
    h.run()
    sure = (time.perf_counter() - t0) * 1000
    ms = h.getModelStatus()
    info = h.getInfo()
    iters = int(getattr(info, "simplex_iteration_count", 0) or 0) + \
        int(getattr(info, "ipm_iteration_count", 0) or 0)
    surum = f"HiGHS {h.version()}"
    if ms == highspy.HighsModelStatus.kOptimal:
        s = h.getSolution()
        ek = {}
        # Here we ask HiGHS for ranging: how far each demand can move before the price changes
        if aralik:
            st, rg = h.getRanging()
            if st == highspy.HighsStatus.kOk:
                ek["satir_ust"] = np.array(rg.row_bound_up.value_)
                ek["satir_alt"] = np.array(rg.row_bound_dn.value_)
        return Cozum("optimal", h.getObjectiveValue(), np.array(s.col_value),
                     np.array(s.row_dual), np.array(s.col_dual), surum, sure, iters, ek)
    if ms in (highspy.HighsModelStatus.kInfeasible,):
        return Cozum("cozumsuz", None, None, None, None, surum, sure, iters)
    if ms == highspy.HighsModelStatus.kUnbounded:
        return Cozum("sinirsiz", None, None, None, None, surum, sure, iters)
    return Cozum("hata:" + h.modelStatusToString(ms), None, None, None, None, surum, sure, iters)


def akis_tekillestir(lp: LP, cozum: Cozum, tolerans: float = 1e-10) -> np.ndarray:
    """Second stage: here we fix the cost at its optimum and minimise total |flow|.

    When regional prices are equal, the flows have infinitely many optima (the
    solver may return circular flows). This step picks the simplest flow pattern
    with the same cost. Prices (duals) are still taken from the first stage.
    """
    n, m = len(lp.c), len(lp.row_lo)
    akis = lp.meta["akis"]
    k = len(akis)
    # Here we add new columns t_l >= |f_l|
    A = lp.A.tocoo()
    rows, cols, vals = list(A.row), list(A.col), list(A.data)
    for q, j in enumerate(akis):
        t = n + q
        rows += [m + 2 * q, m + 2 * q, m + 2 * q + 1, m + 2 * q + 1]
        cols += [j, t, j, t]
        vals += [1.0, -1.0, -1.0, -1.0]
    # Here we add a cost row c'x <= z* so the cost cannot get worse
    r_c = m + 2 * k
    for j in np.nonzero(lp.c)[0]:
        rows.append(r_c); cols.append(int(j)); vals.append(float(lp.c[j]))
    z = cozum.amac
    A2 = sp.csc_matrix((vals, (rows, cols)), shape=(m + 2 * k + 1, n + k))
    lp2 = LP(np.r_[np.zeros(n), np.ones(k)], A2,
             np.r_[lp.row_lo, np.full(2 * k, -INF), -INF],
             np.r_[lp.row_hi, np.zeros(2 * k), z + tolerans * max(1.0, abs(z))],
             np.r_[lp.col_lo, np.zeros(k)], np.r_[lp.col_hi, np.full(k, INF)],
             lp.col_ad + [f"|{lp.col_ad[j]}|" for j in akis], lp.row_ad, lp.meta)
    s2 = coz_highs(lp2)
    if s2.durum != "optimal":
        return cozum.x
    return s2.x[:n]


def coz_scipy(lp: LP) -> Cozum:
    """Second adapter for cross-checking (scipy linprog with HiGHS interior point)."""
    from scipy.optimize import linprog

    # Here we split rows into equalities and <= inequalities, as linprog expects
    eq = lp.row_lo == lp.row_hi
    A = lp.A.tocsr()
    A_ub, b_ub = [], []
    for i in np.where(~eq)[0]:
        if np.isfinite(lp.row_hi[i]):
            A_ub.append(A[i]); b_ub.append(lp.row_hi[i])
        if np.isfinite(lp.row_lo[i]):
            A_ub.append(-A[i]); b_ub.append(-lp.row_lo[i])
    t0 = time.perf_counter()
    r = linprog(lp.c, A_ub=sp.vstack(A_ub) if A_ub else None, b_ub=b_ub or None,
                A_eq=A[eq], b_eq=lp.row_lo[eq], bounds=list(zip(lp.col_lo, lp.col_hi)),
                method="highs-ipm")
    sure = (time.perf_counter() - t0) * 1000
    if r.status != 0:
        return Cozum("cozumsuz" if r.status == 2 else "hata", None, None, None, None,
                     "scipy-ipm", sure, int(r.nit))
    y = np.zeros(len(lp.row_lo)); y[np.where(eq)[0]] = r.eqlin.marginals
    return Cozum("optimal", r.fun, r.x, y, None, "scipy-ipm", sure, int(r.nit))


def dual_amac(lp: LP, y: np.ndarray) -> float:
    """Here we compute the dual objective for given row duals (strong duality check).

    If primal objective == dual objective, the solution is provably optimal.
    """
    d = lp.c - lp.A.T @ y
    tot = 0.0
    for i, yi in enumerate(y):
        b = lp.row_lo[i] if yi > 0 else lp.row_hi[i]
        if abs(yi) > 1e-12:
            tot += yi * b
    for j, dj in enumerate(d):
        b = lp.col_lo[j] if dj > 0 else lp.col_hi[j]
        if abs(dj) > 1e-9:
            tot += dj * b
    return float(tot)
