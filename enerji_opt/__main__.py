"""Demo:  python -m enerji_opt [--senaryo N] [--son-tarih SN]

Here we answer the prototype's four questions with the real solver, then solve
the uncertainty scenarios with the configuration chosen by the resource planner.
All evidence objects are written as JSON into ornek_ciktilar/.
"""
import argparse
import json
from pathlib import Path

from .aciklama import feasibility_repair, kanit, what_if, why_not
from .kaynak import calistir, planla
from .model import veri_yukle
from .senaryo import ozet, senaryo_uret


# Here we save an evidence object as JSON
def _yaz(d: Path, ad: str, obj):
    (d / ad).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--senaryo", type=int, default=1000)
    ap.add_argument("--son-tarih", type=float, default=60.0)
    ap.add_argument("--cikti", default="ornek_ciktilar")
    a = ap.parse_args()
    d = Path(a.cikti); d.mkdir(exist_ok=True)
    v = veri_yukle()
    tl = lambda x: f"{x / 1e6:,.2f} mn TL"

    # [1] Here we solve the base plan and cross-check it with a second solver
    k = kanit(v, capraz_kontrol=True); _yaz(d, "1_temel.json", k)
    print(f"[1] Temel plan: {tl(k['toplam_maliyet_tl'])}, doğrulandı={k['dogrulama']['dogrulandi']}")
    for h in k["hatlar"]:
        if h["baglayici"]:
            print(f"    Dolu hat {h['id']}: gölge fiyat {h['golge_fiyat']:,.0f} TL/MWh")

    # [2] Here we ask: what if wind is 30% lower?
    w = what_if({"ruzgar_carpani": 0.7}); _yaz(d, "2_whatif_ruzgar.json", w)
    print(f"[2] Rüzgâr -%30: maliyet +{tl(w['fark']['maliyet_farki_tl'])}, "
          f"fiyat değişimi {w['fark']['fiyat_degisimi']}")

    # [3] Here we ask: why isn't the Doğu lignite unit running?
    n = why_not("DOG_LINY"); _yaz(d, "3_whynot_dogu_linyit.json", n)
    print(f"[3] Neden Doğu linyit çalışmıyor? neden={n['neden']}, "
          f"500 MW zorlamak +{n['karsi_olgu']['ek_maliyet_tl']:,.0f} TL")

    # [4] Here we ask: demand +25% is infeasible, how do we repair it?
    r = feasibility_repair({"talep_carpani": 1.25}); _yaz(d, "4_repair_talep125.json", r)
    print(f"[4] Talep +%25: açık {r['teshis']['karsilanamayan_talep_mw']:,.0f} MW, "
          f"darboğaz {r['teshis']['darbogaz_hatlar']}, {len(r['planlar'])} onarım planı")
    for p in r["planlar"]:
        print(f"    - {' + '.join(p['kullanilan'])}: {tl(p['toplam_maliyet_tl'])}")

    # [5] Here we let the planner pick resources, then solve all scenarios
    sc = senaryo_uret(a.senaryo)
    plan = planla(len(sc), "senaryo_istatistik", v, son_tarih_s=a.son_tarih, ornek=sc)
    _yaz(d, "5_kaynak_plani.json", plan)
    s = plan["secim"]
    for al in plan["alternatifler"][:4]:
        print(f"    aday: {al['cozucu']} x{al['isci']} {al['cihaz']} ~{al['tahmini_sure_s']:.1f} sn ({al['kaynak']})")
    print(f"    pilot ölçüm (ms/LP): {plan['pilot_olcum_ms_lp_basina']}")
    print(f"[5] Kaynak kararı ({len(sc)} senaryo): {s['cozucu']} x{s['isci']} {s['cihaz']} "
          f"({s['konum']}), tahmini {s['tahmini_sure_s']:.2f} sn. {plan['kural']}")
    sonuc = calistir(plan, v, sc)
    if sonuc["durum"] == "tamam":
        o = ozet(v, sc, sonuc["cozumler"]); o["geri_besleme"] = sonuc["geri_besleme"]
        _yaz(d, "6_senaryo_ozeti.json", o)
        m = o["maliyet_tl"]
        print(f"    Maliyet P5/P50/P95: {tl(m['P5'])} / {tl(m['P50'])} / {tl(m['P95'])}; "
              f"gerçek süre {sonuc['geri_besleme']['gercek_s']} sn")
    else:
        print(f"    HPC betiği yazıldı: {sonuc['betik']}")


if __name__ == "__main__":
    main()
