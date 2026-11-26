"""Uncertainty scenarios: generation, batch solving and statistics.

Here every scenario has the same LP structure (only bounds/demands change).
That is the ideal case for a batched PDLP on the GPU: same matrix, different right-hand sides.
"""
from __future__ import annotations

import numpy as np

from .model import degisiklik_uygula, kur_model, veri_yukle


def senaryo_uret(n: int, tohum: int = 7, ruzgar_ort: float = 1.0, ruzgar_ss: float = 0.30,
                 talep_ss: float = 0.04, temel: dict | None = None) -> list[dict]:
    """Here we sample random scenarios: a wind factor and correlated regional demands."""
    rng = np.random.default_rng(tohum)
    bolgeler = [b["id"] for b in veri_yukle()["bolgeler"]]
    R = len(bolgeler)
    # Here we use 0.6 correlation between regional demands (shared weather effect)
    kov = talep_ss ** 2 * (0.6 * np.ones((R, R)) + 0.4 * np.eye(R))
    L = np.linalg.cholesky(kov)
    out = []
    for _ in range(n):
        d = dict(temel or {})
        d["ruzgar_carpani"] = float(np.clip(rng.normal(ruzgar_ort, ruzgar_ss), 0.05, 1.0))
        z = 1.0 + L @ rng.standard_normal(R)
        d["bolge_talep_carpani"] = {b: float(z[i]) for i, b in enumerate(bolgeler)}
        out.append(d)
    return out


# Here we allow load shedding (penalised at VOLL) in every batch scenario, so every
# scenario is solvable and we can count how often supply falls short.
TOPLU_ONARIM = {"yuk_atma": True}


def senaryo_lp(veri: dict, d: dict):
    """Here we build the LP for one scenario."""
    return kur_model(degisiklik_uygula(veri, d), TOPLU_ONARIM)


def ozet(veri: dict, senaryolar: list[dict], cozumler: list) -> dict:
    """Here we summarise many solutions: cost percentiles, prices, congestion, shedding."""
    lp0 = senaryo_lp(veri, senaryolar[0])
    bolgeler = [b["id"] for b in veri["bolgeler"]]
    R = len(bolgeler)
    ok = [s for s in cozumler if s.durum == "optimal"]
    maliyet = np.array([s.amac for s in ok])
    fiyat = np.array([s.row_dual[:R] for s in ok])
    akis = np.array([s.x[lp0.meta["akis"]] for s in ok])
    kap = np.array([[h["kapasite"] for h in veri["hatlar"]]])
    # Here a line counts as congested only if it is at its limit AND the prices at its two
    # ends differ (if prices are equal, sitting at the limit is just an alternative optimum)
    ri = {b: i for i, b in enumerate(bolgeler)}
    a_idx = [ri[h["a"]] for h in veri["hatlar"]]; b_idx = [ri[h["b"]] for h in veri["hatlar"]]
    fark = np.abs(fiyat[:, b_idx] - fiyat[:, a_idx])
    tikanik = (np.abs(akis) >= kap - 1e-3) & (fark > 1e-3 * np.maximum(1.0, np.abs(fiyat[:, a_idx])))
    atma = np.array([s.x[lp0.meta["yuk_atma"]].sum() for s in ok])
    p = lambda a, q: float(np.percentile(a, q))
    return {
        "senaryo_sayisi": len(cozumler), "cozulen": len(ok),
        "maliyet_tl": {"P5": p(maliyet, 5), "P50": p(maliyet, 50), "P95": p(maliyet, 95),
                       "ortalama": float(maliyet.mean())},
        "bolge_fiyati": {b: {"ortalama": round(float(fiyat[:, i].mean()), 1),
                             "P95": round(p(fiyat[:, i], 95), 1)} for i, b in enumerate(bolgeler)},
        "hat_tikanma_olasiligi": {h["id"]: round(float(tikanik[:, q].mean()), 3)
                                  for q, h in enumerate(veri["hatlar"])},
        "yuk_atma": {"olasilik": round(float((atma > 1e-3).mean()), 3),
                     "kosullu_ortalama_mw": round(float(atma[atma > 1e-3].mean()), 1)
                     if (atma > 1e-3).any() else 0.0},
    }
