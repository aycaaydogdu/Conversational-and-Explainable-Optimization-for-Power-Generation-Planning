"""Evidence object and explanation procedures (what-if, why-not, feasibility repair).

Rule: every number that goes to the LLM comes from solver output in this module.
If a claim cannot be verified numerically, the evidence object says
"dogrulandi": False and the LLM must not present it as a firm result.
"""
from __future__ import annotations

import itertools

import numpy as np

from .model import (LP, akis_tekillestir, coz_highs, coz_scipy, degisiklik_uygula,
                    dual_amac, kur_model, veri_yukle)

TOL = 1e-6


# Here we round numbers for the evidence object (None stays None)
def _r(x, n=1):
    return None if x is None else round(float(x), n)


# ================================================================ evidence object
def kanit(veri: dict | None = None, degisiklik: dict | None = None,
          onarim: dict | None = None, capraz_kontrol: bool = False,
          hat_degeri_kontrol: bool = True) -> dict:
    """Here we solve one scenario and package everything the LLM may cite."""
    veri0 = veri or veri_yukle()
    v = degisiklik_uygula(veri0, degisiklik)
    lp = kur_model(v, onarim)
    s = coz_highs(lp, aralik=True)
    k = {"degisiklik": degisiklik or {}, "onarim": onarim or None, "durum": s.durum,
         "cozucu": {"ad": s.cozucu, "sure_ms": _r(s.sure_ms, 2), "iterasyon": s.iterasyon}}
    if s.durum != "optimal":
        return k

    # Here we pick the simplest flow pattern and note whether alternative optima exist
    x_tekil = akis_tekillestir(lp, s)
    akis_farkli = bool(np.max(np.abs(x_tekil - s.x)[lp.meta["akis"]], initial=0) > 1e-3)
    x = x_tekil
    lmp = {b["id"]: s.row_dual[i] for i, b in enumerate(v["bolgeler"])}

    # --- regions: demand, generation, net import, price and its valid demand range
    bolgeler = []
    for i, b in enumerate(v["bolgeler"]):
        uretim = sum(x[j] for j in lp.meta["uretim"] if v["santraller"][j]["bolge"] == b["id"])
        aralik = None
        if "satir_ust" in s.ek:
            aralik = [_r(s.ek["satir_alt"][i]), _r(s.ek["satir_ust"][i])]
        bolgeler.append({"id": b["id"], "ad": b["ad"], "talep": _r(b["talep"]),
                         "uretim": _r(uretim), "net_ithalat": _r(b["talep"] - uretim),
                         "fiyat": _r(lmp[b["id"]], 2),
                         "fiyat_gecerli_talep_araligi": aralik})

    # --- units: output, reduced cost and a status label
    santraller = []
    for j in lp.meta["uretim"]:
        sn = v["santraller"][j]
        rc = s.col_dual[j]
        kap, mn = lp.col_hi[j], lp.col_lo[j]
        if kap <= TOL:
            durum = "devre_disi"
        elif x[j] >= kap - TOL:
            durum = "tam_kapasite"
        elif x[j] <= mn + TOL:
            durum = "kapali" if mn <= TOL else "alt_sinirda"
        else:
            durum = "marjinal"
        santraller.append({"id": sn["id"], "bolge": sn["bolge"], "tip": sn["tip"],
                           "uretim": _r(x[j]), "kapasite": _r(kap), "maliyet": sn["maliyet"],
                           "indirgenmis_maliyet": _r(rc, 2), "durum": durum})

    # --- lines: flow, direction, utilisation, binding or not, shadow price
    hatlar, baglayici = [], []
    for q, j in enumerate(lp.meta["akis"]):
        h = v["hatlar"][q]
        f, kap = x[j], h["kapasite"]
        dolu = kap > TOL and abs(f) >= kap - 1e-3
        golge = abs(lmp[h["b"]] - lmp[h["a"]]) if dolu else 0.0
        yon = f"{h['a']}->{h['b']}" if f >= 0 else f"{h['b']}->{h['a']}"
        kayit = {"id": h["id"], "akis": _r(abs(f)), "yon": yon if abs(f) > TOL else None,
                 "kapasite": _r(kap), "doluluk": _r(abs(f) / kap if kap else 0, 3),
                 "baglayici": dolu, "golge_fiyat": _r(golge, 2)}
        if dolu and hat_degeri_kontrol:
            # Here we double-check the dual: it may not be unique (degenerate solution),
            # so we re-solve with +-1 MW of line capacity and compare.
            kayit["marjinal_deger"] = _hat_marjinal(veri0, degisiklik, onarim, h["id"], s.amac)
            kayit["dual_tutarli"] = (kayit["marjinal_deger"]["artirma"] is not None and
                                     abs(kayit["marjinal_deger"]["artirma"] - golge) < 1e-3)
        hatlar.append(kayit)
        if dolu:
            baglayici.append({"tur": "hat", "id": h["id"], "golge_fiyat": _r(golge, 2)})
    for sn in santraller:
        if sn["durum"] == "tam_kapasite" and sn["indirgenmis_maliyet"] < -TOL:
            baglayici.append({"tur": "kapasite", "id": sn["id"],
                              "golge_fiyat": _r(-sn["indirgenmis_maliyet"], 2)})

    # --- repair variables that ended up being used
    onarim_kullanim = {}
    for grup in ("soguk_yedek", "talep_katilimi", "hat_acil", "yuk_atma"):
        kul = {lp.col_ad[j]: _r(x[j]) for j in lp.meta[grup] if x[j] > 1e-3}
        if kul:
            onarim_kullanim[grup] = kul

    # --- verification: balance error, bound violation, duality gap, optional cross-check
    Ax = lp.A @ x
    esit = lp.row_lo == lp.row_hi
    denge_hata = float(np.max(np.abs(Ax[esit] - lp.row_lo[esit]), initial=0))
    sinir_ihlal = float(max(np.max(lp.col_lo - x, initial=0), np.max(x - lp.col_hi, initial=0), 0))
    d_amac = dual_amac(lp, s.row_dual)
    dogrulama = {"denge_hatasi_mw": denge_hata, "sinir_ihlali_mw": sinir_ihlal,
                 "primal_amac": _r(s.amac, 2), "dual_amac": _r(d_amac, 2),
                 "dualite_boslugu_goreli": abs(s.amac - d_amac) / max(1.0, abs(s.amac)),
                 "alternatif_akis_deseni_var": akis_farkli}
    if capraz_kontrol:
        s2 = coz_scipy(lp)
        dogrulama["capraz_kontrol"] = {
            "cozucu": s2.cozucu, "amac_farki": _r(abs(s2.amac - s.amac), 4),
            "fiyat_farki_max": _r(float(np.max(np.abs(s2.row_dual[:len(lmp)] -
                                                      s.row_dual[:len(lmp)]))), 4)}
    dogrulama["dogrulandi"] = (denge_hata < 1e-4 and sinir_ihlal < 1e-4 and
                               dogrulama["dualite_boslugu_goreli"] < 1e-7)

    # --- here we find the price-setting (marginal) unit in each region
    marjinal = {}
    for b in bolgeler:
        aday = [sn["id"] for sn in santraller
                if sn["bolge"] == b["id"] and sn["durum"] in ("marjinal",)
                and abs(sn["maliyet"] - b["fiyat"]) < 1e-3]
        marjinal[b["id"]] = aday or None

    k.update({"toplam_maliyet_tl": _r(s.amac, 0), "bolgeler": bolgeler,
              "santraller": santraller, "hatlar": hatlar, "baglayici_kisitlar": baglayici,
              "fiyati_belirleyen_santral": marjinal, "onarim_kullanimi": onarim_kullanim,
              "dogrulama": dogrulama})
    return k


def _hat_marjinal(veri, degisiklik, onarim, hat_id, z0, delta=1.0):
    """Here we measure the right/left marginal value of a line limit (TL/MW) by re-solving."""
    out = {}
    for ad, dd in (("artirma", delta), ("azaltma", -delta)):
        d = dict(degisiklik or {})
        d["hat_kapasite_degisim"] = {**d.get("hat_kapasite_degisim", {}), hat_id: dd}
        s = coz_highs(kur_model(degisiklik_uygula(veri, d), onarim))
        out[ad] = _r((z0 - s.amac) / delta if ad == "artirma" else (s.amac - z0) / delta, 2) \
            if s.durum == "optimal" else None
    return out


# ================================================================ what-if
def what_if(degisiklik: dict, veri: dict | None = None) -> dict:
    """Here we solve before and after the change and report the difference.

    If the changed model is infeasible, we hand over to feasibility_repair.
    """
    veri = veri or veri_yukle()
    once = kanit(veri)
    sonra = kanit(veri, degisiklik)
    out = {"tur": "what_if", "degisiklik": degisiklik, "temel": once}
    if sonra["durum"] != "optimal":
        out["sonuc"] = "cozumsuz"
        out["onarim"] = feasibility_repair(degisiklik, veri)
        return out
    out["yeni"] = sonra
    out["fark"] = karsilastir(once, sonra)
    return out


def karsilastir(a: dict, b: dict) -> dict:
    """Here we diff two evidence objects: cost, prices, outputs, flows, binding set."""
    fiyat = {x["id"]: _r(y["fiyat"] - x["fiyat"], 2)
             for x, y in zip(a["bolgeler"], b["bolgeler"]) if abs(y["fiyat"] - x["fiyat"]) > 1e-6}
    uretim = {x["id"]: _r(y["uretim"] - x["uretim"])
              for x, y in zip(a["santraller"], b["santraller"]) if abs(y["uretim"] - x["uretim"]) > 0.05}
    akis = {x["id"]: [x["akis"], y["akis"]]
            for x, y in zip(a["hatlar"], b["hatlar"]) if abs(y["akis"] - x["akis"]) > 0.05}
    ba = {c["id"] for c in a["baglayici_kisitlar"]}
    bb = {c["id"] for c in b["baglayici_kisitlar"]}
    return {"maliyet_farki_tl": _r(b["toplam_maliyet_tl"] - a["toplam_maliyet_tl"], 0),
            "fiyat_degisimi": fiyat, "uretim_degisimi": uretim, "akis_degisimi": akis,
            "yeni_baglayici": sorted(bb - ba), "artik_baglayici_degil": sorted(ba - bb)}


# ================================================================ why-not
def why_not(santral_id: str, miktar: float | None = None, veri: dict | None = None,
            degisiklik: dict | None = None) -> dict:
    """Here we answer "why wasn't unit X used (more)?"

    1) Reduced cost: unit cost minus regional price.
    2) Why the regional price is low/high: binding lines leaving the region.
    3) Counterfactual: force the unit to run +miktar MW, re-solve, measure the extra cost.
    """
    veri = veri or veri_yukle()
    k = kanit(veri, degisiklik)
    sn = next((s for s in k["santraller"] if s["id"] == santral_id), None)
    if sn is None:
        raise ValueError(f"Bilinmeyen santral: {santral_id}")
    bolge = next(b for b in k["bolgeler"] if b["id"] == sn["bolge"])
    bosluk = sn["kapasite"] - sn["uretim"]
    out = {"tur": "why_not", "santral": sn, "bolge_fiyati": bolge["fiyat"], "kanit_temel": k}
    if bosluk <= 1e-6:
        out["sonuc"] = "zaten_tam_kapasite"
        return out

    miktar = min(miktar or 500.0, bosluk)
    out["fiyat_farki"] = _r(sn["maliyet"] - bolge["fiyat"], 2)

    # Here we check if the region is cut off: full lines leaving it vs. prices elsewhere
    diger_max = max(b["fiyat"] for b in k["bolgeler"])
    tikali = [h for h in k["hatlar"] if h["baglayici"] and sn["bolge"] in h["id"].split("-")
              and h["yon"] and h["yon"].startswith(sn["bolge"])]
    if sn["maliyet"] < diger_max - 1e-6 and tikali:
        neden = "iletim_kisiti"
    elif sn["maliyet"] > bolge["fiyat"] + 1e-6:
        neden = "daha_ucuz_alternatif_var"
    else:
        neden = "esit_maliyetli_alternatif"  # rc = 0: multiple optima
    out["neden"] = neden
    out["ilgili_dolu_hatlar"] = [{"id": h["id"], "golge_fiyat": h["golge_fiyat"]} for h in tikali]

    # Here we run the counterfactual: force the unit on and re-solve
    d = dict(degisiklik or {})
    d["zorla_min"] = {**d.get("zorla_min", {}), santral_id: sn["uretim"] + miktar}
    k2 = kanit(veri, d, hat_degeri_kontrol=False)
    if k2["durum"] == "optimal":
        ek = k2["toplam_maliyet_tl"] - k["toplam_maliyet_tl"]
        out["karsi_olgu"] = {"zorla_ek_mw": _r(miktar), "ek_maliyet_tl": _r(ek, 0),
                             "mw_basina_tl": _r(ek / miktar, 2),
                             "yerine_azalan": karsilastir(k, k2)["uretim_degisimi"]}
        # Here we check that the reduced-cost prediction matches the measured cost
        out["karsi_olgu"]["indirgenmis_maliyetle_uyumlu"] = \
            abs(ek / miktar - max(0.0, sn["indirgenmis_maliyet"])) < 1e-3
    else:
        out["karsi_olgu"] = {"zorla_ek_mw": _r(miktar), "durum": k2["durum"]}
    return out


# ================================================================ feasibility repair
ONARIM_SECENEKLERI = {
    "soguk_yedek": "Soğuk yedek santralleri devreye al",
    "talep_katilimi": "Talep tarafı katılımı (sanayi yük kısma anlaşmaları)",
    "hat_acil": "Hatları kısa süreli acil kapasiteyle çalıştır (+%10)",
}


def feasibility_repair(degisiklik: dict, veri: dict | None = None, en_fazla: int = 3) -> dict:
    """Here we handle an infeasible model: (1) measure the gap and its cause,
    (2) generate alternative repair plans ranked by cost."""
    veri = veri or veri_yukle()
    v = degisiklik_uygula(veri, degisiklik)

    # (1) Diagnosis: only load shedding is allowed, at cost 1 -> minimum unmet demand.
    lp = kur_model(v, {"yuk_atma": True})
    lp.c[:] = 0.0
    lp.c[lp.meta["yuk_atma"]] = 1.0
    s = coz_highs(lp)
    acik = s.amac
    nerede = {lp.col_ad[j].split("_")[0]: _r(s.x[j]) for j in lp.meta["yuk_atma"] if s.x[j] > 1e-3}
    # Here we read the diagnosis like a Farkas certificate: bounds with non-zero duals
    # are the cause of the gap. Lines are listed one by one, units are summarised by region.
    darbogaz_hat = [lp.col_ad[j] for j in lp.meta["akis"] if abs(s.col_dual[j]) > 1e-6]
    bolge_ozet = {}
    for b in v["bolgeler"]:
        js = [j for j in lp.meta["uretim"] if v["santraller"][j]["bolge"] == b["id"]]
        bos = sum(lp.col_hi[j] - s.x[j] for j in js)
        bolge_ozet[b["id"]] = {"bos_kapasite_mw": _r(bos),
                               "tum_santraller_tam": bool(bos < 1e-3)}
    toplam_talep = sum(b["talep"] for b in v["bolgeler"])
    toplam_kap = sum(sn["kapasite"] for sn in v["santraller"])
    teshis = {"karsilanamayan_talep_mw": _r(acik), "bolgelere_gore": nerede,
              "toplam_talep": _r(toplam_talep), "toplam_kapasite": _r(toplam_kap),
              "mahsur_kapasite_mw": _r(toplam_kap - (toplam_talep - acik)),
              "darbogaz_hatlar": darbogaz_hat, "bolge_kapasite_durumu": bolge_ozet,
              "not": "Açığın bölgelere dağılımı tekil olmayabilir; toplam açık kesindir."}

    # (2) Repair plans: here we try combinations of resources and rank the feasible ones by cost.
    planlar = []
    anahtarlar = list(ONARIM_SECENEKLERI)
    for n in range(1, len(anahtarlar) + 1):
        for komb in itertools.combinations(anahtarlar, n):
            k = kanit(veri, degisiklik, {a: True for a in komb}, hat_degeri_kontrol=False)
            if k["durum"] != "optimal":
                continue
            kullanilan = tuple(sorted(k["onarim_kullanimi"]))
            if any(set(p["kullanilan"]) <= set(kullanilan) for p in planlar):
                continue  # a plan using fewer resources already exists
            planlar.append({"kullanilan": kullanilan,
                            "aciklama": [ONARIM_SECENEKLERI[a] for a in kullanilan],
                            "miktarlar": k["onarim_kullanimi"],
                            "toplam_maliyet_tl": k["toplam_maliyet_tl"],
                            "fiyatlar": {b["id"]: b["fiyat"] for b in k["bolgeler"]},
                            "dogrulandi": k["dogrulama"]["dogrulandi"]})
    planlar.sort(key=lambda p: p["toplam_maliyet_tl"])
    # Here we fall back to load shedding only if nothing else works
    son_care = None
    if not planlar:
        k = kanit(veri, degisiklik, {"soguk_yedek": True, "talep_katilimi": True,
                                     "hat_acil": True, "yuk_atma": True}, hat_degeri_kontrol=False)
        son_care = {"yuk_atma": k["onarim_kullanimi"].get("yuk_atma"),
                    "toplam_maliyet_tl": k["toplam_maliyet_tl"]}
    for p in planlar:
        p["kullanilan"] = list(p["kullanilan"])
    return {"tur": "feasibility_repair", "degisiklik": degisiklik, "teshis": teshis,
            "planlar": planlar[:en_fazla], "son_care_yuk_atma": son_care}
