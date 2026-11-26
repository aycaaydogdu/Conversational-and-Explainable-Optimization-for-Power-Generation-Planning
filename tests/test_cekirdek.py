"""Core tests. Run with: python -m pytest -q"""
import numpy as np
import pytest

from enerji_opt.aciklama import feasibility_repair, kanit, what_if, why_not
from enerji_opt.cozuculer import ADAPTORLER, kullanilabilir, kurulu_mu
from enerji_opt.kaynak import planla
from enerji_opt.model import coz_highs, degisiklik_uygula, kur_model, veri_yukle
from enerji_opt.senaryo import ozet, senaryo_lp, senaryo_uret

V = veri_yukle()


def test_temel_dogrulama():
    # Here we check the base plan is verified and matches the scipy cross-check
    k = kanit(V, capraz_kontrol=True)
    d = k["dogrulama"]
    assert k["durum"] == "optimal" and d["dogrulandi"]
    assert d["capraz_kontrol"]["amac_farki"] < 1e-3
    # in every region: generation + net import = demand
    for b in k["bolgeler"]:
        assert abs(b["uretim"] + b["net_ithalat"] - b["talep"]) < 1e-6


def test_fiyat_dual_sonlu_farkla_ayni():
    """Here we check that the regional price (dual) equals the cost increase for +1 MW demand."""
    z0 = coz_highs(kur_model(V)).amac
    k = kanit(V, hat_degeri_kontrol=False)
    for b in k["bolgeler"]:
        z1 = coz_highs(kur_model(degisiklik_uygula(V, {"bolge_talep_carpani":
                                                       {b["id"]: (b["talep"] + 1) / b["talep"]}}))).amac
        assert abs((z1 - z0) - b["fiyat"]) < 1e-3, b["id"]


def test_hat_golge_fiyati_dogrulanir():
    # Here we check the DOG-ICA shadow price is confirmed by re-solving
    k = kanit(V)
    h = next(h for h in k["hatlar"] if h["id"] == "DOG-ICA")
    assert h["baglayici"] and h["dual_tutarli"]
    assert h["golge_fiyat"] == pytest.approx(1750.0)


def test_dejenere_durum_yakalanir():
    """Here we check we catch degeneracy: if the hydro limit and the line limit bind at the
    same time, the dual alone cannot be trusted (left and right values differ)."""
    k = kanit(V, {"santral_kapasite": {"DOG_HES": 10000}})
    h = next(h for h in k["hatlar"] if h["id"] == "DOG-ICA")
    assert h["marjinal_deger"]["artirma"] != h["marjinal_deger"]["azaltma"]


def test_why_not_iletim_kisiti():
    # Here we check why-not finds the transmission limit as the cause
    r = why_not("DOG_LINY")
    assert r["neden"] == "iletim_kisiti"
    assert r["karsi_olgu"]["indirgenmis_maliyetle_uyumlu"]


def test_repair_acik_ve_planlar():
    # Here we check the +25% demand gap (1700 MW) and that repair plans are minimal and sorted
    r = feasibility_repair({"talep_carpani": 1.25})
    assert r["teshis"]["karsilanamayan_talep_mw"] == pytest.approx(1700.0)
    assert r["teshis"]["darbogaz_hatlar"] == ["DOG-ICA"]
    assert len(r["planlar"]) >= 2
    maliyetler = [p["toplam_maliyet_tl"] for p in r["planlar"]]
    assert maliyetler == sorted(maliyetler)
    # no plan uses a superset of another plan's resources
    for p in r["planlar"]:
        for q in r["planlar"]:
            if p is not q:
                assert not set(q["kullanilan"]) < set(p["kullanilan"])


def test_what_if_ruzgar():
    # Here we check less wind makes the plan more expensive
    r = what_if({"ruzgar_carpani": 0.7})
    assert r["fark"]["maliyet_farki_tl"] > 0


@pytest.mark.parametrize("ad", [k for k in ADAPTORLER if k != "highs"])
def test_cozuculer_ayni_sonuc(ad):
    # Here we check every other solver agrees with HiGHS (skipped if unavailable)
    a = ADAPTORLER[ad]
    ok, neden = kullanilabilir(a)
    if not ok:
        pytest.skip(f"{ad} bu makinede kullanılamıyor: {neden}")
    lp = kur_model(degisiklik_uygula(V, {"ruzgar_carpani": 0.6}))
    ref = coz_highs(lp)
    s = a.fonk(lp, a.en_iyi_tolerans)
    assert abs(s.amac - ref.amac) / ref.amac < (1e-9 if a.kesin_dual else 1e-4)
    assert np.allclose(s.row_dual, ref.row_dual, rtol=1e-3, atol=1.0)


def test_planlayici_kurallari():
    # Here we check the planner's rules on a simulated 4-GPU + SLURM machine
    sc = senaryo_uret(20)
    d = {"cpu_cekirdek": 2, "gpular": [{"index": i, "ad": "A100", "bellek_mb": 81920,
                                        "kullanim_yuzde": 0} for i in range(4)],
         "bos_gpu_sayisi": 4, "cozuculer": {k: True for k in ADAPTORLER}, "slurm": True,
         "slurm_icinde": False}
    # a question that needs exact duals never goes to PDLP/GPU
    p = planla(1, "why_not", V, ornek=sc, donanim=d)
    assert ADAPTORLER[p["secim"]["cozucu"]].kesin_dual
    # huge job + short deadline -> more than one GPU
    p = planla(1_000_000, "senaryo_istatistik", V, son_tarih_s=60, ornek=sc, donanim=d)
    assert p["secim"]["cihaz"] == "gpu" and p["secim"]["isci"] >= 2
    # small job -> no GPU needed
    p = planla(100, "senaryo_istatistik", V, son_tarih_s=60, ornek=sc, donanim=d)
    assert p["secim"]["cihaz"] == "cpu"


def test_toplu_ozet():
    # Here we check the scenario summary is complete and percentiles are ordered
    sc = senaryo_uret(200)
    cz = [coz_highs(senaryo_lp(V, d)) for d in sc]
    o = ozet(V, sc, cz)
    assert o["cozulen"] == 200
    assert o["maliyet_tl"]["P5"] <= o["maliyet_tl"]["P50"] <= o["maliyet_tl"]["P95"]


def test_cuopt_toplu():
    # Here we check cuOpt's batch results match HiGHS (skipped without a GPU)
    a = ADAPTORLER["cuopt"]
    ok, neden = kullanilabilir(a)
    if not ok:
        pytest.skip(f"cuopt kullanılamıyor: {neden}")
    sc = senaryo_uret(50)
    lps = [senaryo_lp(V, d) for d in sc]
    ref = [coz_highs(lp).amac for lp in lps]
    gpu = a.toplu(lps, 1e-6)
    for r, g in zip(ref, gpu):
        assert g.durum == "optimal" and abs(g.amac - r) / r < 1e-4
