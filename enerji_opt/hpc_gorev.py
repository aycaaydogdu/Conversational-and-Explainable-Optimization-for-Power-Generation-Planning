"""One task of the HPC (SLURM) array job: here we solve one chunk of scenarios and save the results."""
import argparse
import json

from .kaynak import _isci
from .model import veri_yukle

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parca"); p.add_argument("--cozucu"); p.add_argument("--cikti")
    p.add_argument("--tolerans", type=float, default=1e-4)
    a = p.parse_args()
    senaryolar = json.load(open(a.parca))
    cozumler = _isci((veri_yukle(), senaryolar, a.cozucu, a.tolerans, None))
    json.dump([{"durum": c.durum, "amac": c.amac,
                "x": None if c.x is None else c.x.tolist(),
                "row_dual": None if c.row_dual is None else c.row_dual.tolist()} for c in cozumler],
              open(a.cikti, "w"))
