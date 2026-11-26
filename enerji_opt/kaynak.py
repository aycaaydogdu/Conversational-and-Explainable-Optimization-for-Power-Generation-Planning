"""Resource planner: here the system decides by itself which solver, how many
CPUs/GPUs and where (local / HPC) to run a job, and records why.

Decision steps
  1. Hardware detection : CPU count, GPUs (nvidia-smi), working solvers, SLURM.
  2. Accuracy need      : question type -> required tolerance and whether exact duals
                          are needed (why-not / shadow price -> simplex;
                          scenario statistics -> PDLP is enough).
  3. Pilot measurement  : every suitable solver that runs locally (CPU and GPU) solves a
                          few LPs and the time per LP is MEASURED. Anything that cannot be
                          measured (e.g. remote HPC) uses an assumption, flagged "varsayim".
  4. Candidate configs  : estimated time and resource cost for (solver, workers/GPUs, location).
  5. Choice             : the cheapest config that meets the deadline;
                          if none does, the fastest one.
  6. Run + feedback     : if the real time differs from the estimate by >2x,
                          the calibration file is updated.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .cozuculer import ADAPTORLER, kullanilabilir, kurulu_mu
from .senaryo import senaryo_lp

# ---------------------------------------------------------------- assumptions
# Here we keep starting assumptions for resources we cannot measure. Once a pilot
# measurement exists on a GPU/HPC, the real values replace these (calibration file).
VARSAYIM = {
    "gpu_baslatma_s": 3.0,          # CUDA context + data transfer
    "gpu_lp_basina_ms_toplu": 0.05,  # per small LP in cuOpt BatchSolve (assumption; T4 measured ~45 ms)
    "gpu_lp_basina_ms_tekil": 15.0,  # single LP, excluding GPU start-up (assumption)
    "gpu_coklu_verim": 0.90,         # scaling efficiency per doubling of GPUs
    "cpu_coklu_verim": 0.85,
    "hpc_kuyruk_bekleme_s": 120.0,   # estimated SLURM queue wait
    "hpc_max_gpu": 8, "hpc_max_cpu": 128,
    # resource cost unit: 1 GPU-second = 30 core-seconds (energy / quota weight)
    "gpu_saniye_agirlik": 30.0,
}
KALIBRASYON_DOSYASI = Path(os.environ.get("ENERJI_OPT_KALIBRASYON",
                                          Path.home() / ".enerji_opt_kalibrasyon.json"))

# Here we map each question type to the accuracy it needs
HASSASIYET = {
    "why_not":    {"tolerans": 1e-9, "kesin_dual": True,
                   "neden": "indirgenmiş maliyet ve gölge fiyat kesin olmalı"},
    "what_if":    {"tolerans": 1e-9, "kesin_dual": True,
                   "neden": "iki çözümün farkı raporlanıyor; küçük hatalar farkı bozar"},
    "repair":     {"tolerans": 1e-9, "kesin_dual": False,
                   "neden": "uygunluk kararı kesin olmalı"},
    "senaryo_istatistik": {"tolerans": 1e-4, "kesin_dual": False,
                           "neden": "P5-P95 için 1e-4 göreli hata, senaryo örnekleme hatasından çok küçük"},
}


# ---------------------------------------------------------------- 1. hardware
def donanim_tespit() -> dict:
    """Here we detect CPUs, free GPUs, working solvers and whether SLURM is available."""
    gpular = []
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            for satir in r.stdout.strip().splitlines():
                i, ad, mem, kul = [p.strip() for p in satir.split(",")]
                gpular.append({"index": int(i), "ad": ad, "bellek_mb": int(mem), "kullanim_yuzde": int(kul)})
        except Exception:
            pass
    bos_gpu = [g for g in gpular if g["kullanim_yuzde"] < 50]
    return {
        "cpu_cekirdek": os.cpu_count() or 1,
        "gpular": gpular, "bos_gpu_sayisi": len(bos_gpu),
        "cozuculer": {k: kurulu_mu(a) for k, a in ADAPTORLER.items()},
        "cozucu_durumu": {k: kullanilabilir(a)[1] for k, a in ADAPTORLER.items()},
        "slurm": shutil.which("sbatch") is not None,
        "slurm_icinde": "SLURM_JOB_ID" in os.environ,
    }


# ---------------------------------------------------------------- 3. pilot measurement
# Here we read/write past measurements so later decisions can reuse them
def _kalibrasyon_oku() -> dict:
    try:
        return json.loads(KALIBRASYON_DOSYASI.read_text())
    except Exception:
        return {}


def _kalibrasyon_yaz(k: dict):
    try:
        KALIBRASYON_DOSYASI.write_text(json.dumps(k, indent=1))
    except Exception:
        pass


def pilot_olcum(veri: dict, ornek: list[dict], adaptorler: list[str], tolerans: float,
                gpu: bool = True) -> dict:
    """Here we measure the time per LP (ms) for every solver that runs locally."""
    lps = [senaryo_lp(veri, d) for d in ornek]
    olcum = {}
    for ad in adaptorler:
        a = ADAPTORLER[ad]
        a.fonk(lps[0], max(tolerans, a.en_iyi_tolerans))  # warm-up
        t0 = time.perf_counter()
        for lp in lps:
            a.fonk(lp, max(tolerans, a.en_iyi_tolerans))
        olcum[ad] = (time.perf_counter() - t0) * 1000 / len(lps)
    # Here we really measure GPU batch solvers if available (never trust the assumption)
    for ad, a in ADAPTORLER.items():
        if not gpu or a.cihaz != "gpu" or a.toplu is None or not kurulu_mu(a):
            continue
        n = 200
        toplu = [lps[i % len(lps)] for i in range(n)]
        t0 = time.perf_counter()
        a.toplu(toplu[:5], max(tolerans, a.en_iyi_tolerans))       # warm-up + start-up
        olcum[ad + "_baslatma_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        a.toplu(toplu, max(tolerans, a.en_iyi_tolerans))
        olcum[ad + "_toplu"] = (time.perf_counter() - t0) * 1000 / n
    # Here we also measure LP build time, since it is part of the job
    t0 = time.perf_counter()
    for d in ornek:
        senaryo_lp(veri, d)
    olcum["_lp_kurma"] = (time.perf_counter() - t0) * 1000 / len(ornek)
    return olcum


# ---------------------------------------------------------------- 4. candidates
@dataclass
class Aday:
    cozucu: str
    cihaz: str
    konum: str            # location: "yerel" (local) | "hpc"
    isci: int             # number of CPU processes or GPUs
    tahmini_sure_s: float
    kaynak_maliyeti: float  # in core-second equivalents
    kaynak: str           # "olcum" (measured) | "onceki_olcum" (past) | "varsayim" (assumed)
    not_: str = ""


def adaylar_uret(n_lp: int, olcum: dict, donanim: dict, ihtiyac: dict, kal: dict) -> list[Aday]:
    """Here we list every feasible (solver, workers, location) and estimate its time and cost."""
    V = {**VARSAYIM, **kal.get("varsayim_guncel", {})}
    adaylar = []
    kurma_ms = olcum.get("_lp_kurma", 1.0)

    def cpu_sure(ms, k, baslatma):
        verim = V["cpu_coklu_verim"] if k > 1 else 1.0
        return baslatma + n_lp * (ms + kurma_ms) / 1000 / (k * verim)

    for ad, a in ADAPTORLER.items():
        # Here we drop solvers that cannot meet the accuracy need or do not work here
        if ihtiyac["kesin_dual"] and not a.kesin_dual:
            continue
        if a.en_iyi_tolerans > ihtiyac["tolerans"] * 10:
            continue
        if not donanim["cozuculer"].get(ad):
            continue  # not installed or not working on this machine
        if a.cihaz == "cpu":
            ms = olcum.get(ad)
            kaynak = "olcum"
            if ms is None:
                if ad not in kal.get("olcum", {}):
                    continue
                ms, kaynak = kal["olcum"][ad], "onceki_olcum"
            for k in sorted({1, 2, 4, 8, 16, donanim["cpu_cekirdek"]}):
                if k > donanim["cpu_cekirdek"]:
                    continue
                t = cpu_sure(ms, k, 0.3 if k > 1 else 0.0)
                adaylar.append(Aday(ad, "cpu", "yerel", k, t, t * k, kaynak))
            if donanim["slurm"]:
                for k in (32, 64, V["hpc_max_cpu"]):
                    t = V["hpc_kuyruk_bekleme_s"] + cpu_sure(ms, k, 2.0)
                    adaylar.append(Aday(ad, "cpu", "hpc", k, t, (t - V["hpc_kuyruk_bekleme_s"]) * k,
                                        kaynak, "kuyruk süresi varsayım"))
        else:  # GPU: prefer a fresh measurement, then a past one, then the assumption
            if ad + "_toplu" in olcum:
                gpu_ms, kaynak = olcum[ad + "_toplu"], "olcum"
            elif kal.get("olcum", {}).get(ad + "_toplu"):
                gpu_ms, kaynak = kal["olcum"][ad + "_toplu"], "onceki_olcum"
            else:
                gpu_ms, kaynak = V["gpu_lp_basina_ms_toplu"], "varsayim"

            baslatma = olcum.get(ad + "_baslatma_s", V["gpu_baslatma_s"])

            def gpu_sure(k):
                verim = V["gpu_coklu_verim"] ** math.log2(k) if k > 1 else 1.0
                return baslatma + n_lp * (gpu_ms + kurma_ms / 8) / 1000 / (k * verim)

            yerel_gpu = donanim["bos_gpu_sayisi"] if donanim["cozuculer"].get(ad) else 0
            for k in (1, 2, 4, 8):
                if k <= yerel_gpu:
                    t = gpu_sure(k)
                    adaylar.append(Aday(ad, "gpu", "yerel", k, t, t * k * V["gpu_saniye_agirlik"], kaynak))
                if donanim["slurm"] and k <= V["hpc_max_gpu"]:
                    t = V["hpc_kuyruk_bekleme_s"] + gpu_sure(k)
                    adaylar.append(Aday(ad, "gpu", "hpc", k, t,
                                        (t - V["hpc_kuyruk_bekleme_s"]) * k * V["gpu_saniye_agirlik"],
                                        kaynak, "kuyruk süresi varsayım"))
    return adaylar


# ---------------------------------------------------------------- 5. decision
def planla(n_lp: int, soru_turu: str, veri: dict, son_tarih_s: float = 60.0,
           ornek: list[dict] | None = None, donanim: dict | None = None) -> dict:
    """Here we run the whole decision: detect, measure, list candidates, choose, explain."""
    donanim = donanim or donanim_tespit()
    ihtiyac = HASSASIYET[soru_turu]
    kal = _kalibrasyon_oku()
    yerel_cpu = [k for k, a in ADAPTORLER.items() if a.cihaz == "cpu" and donanim["cozuculer"].get(k)
                 and kurulu_mu(a) and (a.kesin_dual or not ihtiyac["kesin_dual"])]
    print(f"[planlayıcı] pilot ölçüm: {', '.join(yerel_cpu)}"
          + (" + GPU" if not ihtiyac["kesin_dual"] and donanim["bos_gpu_sayisi"] else ""), flush=True)
    olcum = pilot_olcum(veri, (ornek or [{}])[:10], yerel_cpu, ihtiyac["tolerans"],
                        gpu=not ihtiyac["kesin_dual"])
    kal.setdefault("olcum", {}).update({k: v for k, v in olcum.items() if not k.startswith("_")})
    _kalibrasyon_yaz(kal)

    adaylar = adaylar_uret(n_lp, olcum, donanim, ihtiyac, kal)
    # Here we choose: cheapest config that meets the deadline, otherwise the fastest
    uygun = [a for a in adaylar if a.tahmini_sure_s <= son_tarih_s]
    if uygun:
        secim = min(uygun, key=lambda a: (a.kaynak_maliyeti, a.tahmini_sure_s))
        kural = (f"Son tarihi ({son_tarih_s:.0f} sn) karşılayan {len(uygun)} yapılandırma içinde "
                 f"kaynak maliyeti en düşük olan seçildi.")
    else:
        secim = min(adaylar, key=lambda a: a.tahmini_sure_s)
        kural = "Hiçbir yapılandırma son tarihi karşılamıyor; en hızlı olan seçildi."
    siralama = sorted(adaylar, key=lambda a: (a.tahmini_sure_s > son_tarih_s, a.kaynak_maliyeti))
    return {
        "is": {"lp_sayisi": n_lp, "soru_turu": soru_turu, "son_tarih_s": son_tarih_s},
        "hassasiyet": ihtiyac, "donanim": donanim,
        "pilot_olcum_ms_lp_basina": {k: round(v, 4) for k, v in olcum.items()},
        "secim": asdict(secim), "kural": kural,
        "alternatifler": [asdict(a) for a in siralama[:6]],
        "varsayimlar_kullanildi": any(a.kaynak == "varsayim" for a in siralama[:6]),
    }


# ---------------------------------------------------------------- 6. execution
def _isci(arg):
    """Here we solve one chunk of scenarios (in its own process; sees only its own GPU)."""
    veri, parca, cozucu, tolerans, gpu_id = arg
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from .cozuculer import ADAPTORLER as AD
    a = AD[cozucu]
    if a.toplu is None:
        return [a.fonk(senaryo_lp(veri, d), tolerans) for d in parca]
    out, P = [], 2000  # here we go in chunks to limit GPU memory and time per call
    for i in range(0, len(parca), P):
        out += a.toplu([senaryo_lp(veri, d) for d in parca[i:i + P]], tolerans)
        print(f"  [{cozucu}{'' if gpu_id is None else f' gpu{gpu_id}'}] {len(out)}/{len(parca)}", flush=True)
    return out


def calistir(plan: dict, veri: dict, senaryolar: list[dict], cikti_dizini: str = "hpc_is") -> dict:
    """Here we execute the chosen plan locally (or write a SLURM script) and feed timing back."""
    s = plan["secim"]
    tol = max(plan["hassasiyet"]["tolerans"], ADAPTORLER[s["cozucu"]].en_iyi_tolerans)
    t0 = time.perf_counter()
    if s["konum"] == "hpc":
        return {"durum": "hpc_gonderilecek", "betik": slurm_betigi_yaz(plan, senaryolar, cikti_dizini)}
    k = s["isci"]
    parcalar = [senaryolar[i::k] for i in range(k)]
    gpu_ids = [g["index"] for g in plan["donanim"]["gpular"]][:k] if s["cihaz"] == "gpu" else [None] * k
    args = [(veri, p, s["cozucu"], tol, g) for p, g in zip(parcalar, gpu_ids)]
    if k == 1:
        sonuc = [_isci(args[0])]
    else:
        import multiprocessing as mp
        # Here we use spawn for GPU workers, because CUDA can hang after fork
        ctx = mp.get_context("spawn" if s["cihaz"] == "gpu" else None)
        with ProcessPoolExecutor(max_workers=k, mp_context=ctx) as ex:
            sonuc = list(ex.map(_isci, args))
    # Here we put the chunks back into the original scenario order
    cozumler = [None] * len(senaryolar)
    for i, r in enumerate(sonuc):
        for j, c in enumerate(r):
            cozumler[i + j * k] = c
    gercek = time.perf_counter() - t0
    tahmin = s["tahmini_sure_s"]
    geri_besleme = {"tahmini_s": round(tahmin, 3), "gercek_s": round(gercek, 3),
                    "oran": round(gercek / tahmin, 2) if tahmin else None}
    # Here we update the calibration if the estimate was off by more than 2x
    if tahmin and not (0.5 <= gercek / tahmin <= 2.0):
        kal = _kalibrasyon_oku()
        ms = gercek * 1000 * k / len(senaryolar)
        kal.setdefault("olcum", {})[s["cozucu"] + ("_toplu" if s["cihaz"] == "gpu" else "")] = ms
        _kalibrasyon_yaz(kal)
        geri_besleme["kalibrasyon_guncellendi"] = True
    return {"durum": "tamam", "cozumler": cozumler, "geri_besleme": geri_besleme}


def slurm_betigi_yaz(plan: dict, senaryolar: list[dict], dizin: str) -> str:
    """Here we write a SLURM array job for HPC: each task solves one chunk on its own GPU/CPU."""
    s = plan["secim"]
    d = Path(dizin); d.mkdir(parents=True, exist_ok=True)
    k = s["isci"]
    for i in range(k):
        (d / f"parca_{i}.json").write_text(json.dumps(senaryolar[i::k]))
    kaynak = "#SBATCH --gres=gpu:1" if s["cihaz"] == "gpu" else "#SBATCH --cpus-per-task=1"
    betik = f"""#!/bin/bash
#SBATCH --job-name=enerji_opt
#SBATCH --array=0-{k - 1}
#SBATCH --ntasks=1
{kaynak}
#SBATCH --time=00:30:00
#SBATCH --output={d}/log_%a.txt
# Generated by the resource planner: {s['cozucu']} x {k} {s['cihaz']}
python -m enerji_opt.hpc_gorev --parca {d}/parca_${{SLURM_ARRAY_TASK_ID}}.json \\
    --cozucu {s['cozucu']} --tolerans {plan['hassasiyet']['tolerans']} \\
    --cikti {d}/sonuc_${{SLURM_ARRAY_TASK_ID}}.json
"""
    yol = d / "is.slurm"
    yol.write_text(betik)
    return str(yol)
