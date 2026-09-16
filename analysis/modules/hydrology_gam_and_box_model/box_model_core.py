"""Two-box equations from the revision analysis, without figure-layout dependencies.

The scenario grid and random draws below retain the original seed and ranges.
The curated literature evidence table is contextual evidence; this calculation
uses the explicit parameter ranges below, not a fit to that table.
"""
from pathlib import Path
import numpy as np
import pandas as pd

def steady_state(
    rt,
    source=1.0,
    k_sed=1.5e-3,
    k_resus=3.5e-4,
    k_bur=2.5e-4,
    H=10.0,
    Hs=0.10,
):
    """Normalized two-box steady state under local fishery loading.

    dCw/dt = source - (k_flush + k_sed) Cw + k_resus * (Hs/H) Cs
    dCs/dt = (k_sed / (Hs/H)) Cw - (k_resus + k_bur) Cs
    """
    rt_arr = np.asarray(rt, dtype=float)
    k_flush = 1.0 / np.maximum(rt_arr, 1e-12)
    R = Hs / H
    a11 = -(k_flush + k_sed)
    a12 = k_resus * R
    a21 = k_sed / R
    a22 = -(k_resus + k_bur)
    det = a11 * a22 - a12 * a21
    Cw = (-source * a22) / np.maximum(det, 1e-15)
    Cs = (source * a21) / np.maximum(det, 1e-15)
    return Cw, Cs

def dynamics(rt, years=8, source=1.0, k_sed=1.5e-3, k_resus=3.5e-4,
             k_bur=2.5e-4, H=10.0, Hs=0.10, dt=1.0):
    t = np.arange(0, years * 365.25 + dt, dt)
    Cw = np.zeros_like(t)
    Cs = np.zeros_like(t)
    R = Hs / H
    k_flush = 1.0 / rt
    for i in range(t.size - 1):
        dCw = source - (k_flush + k_sed) * Cw[i] + k_resus * R * Cs[i]
        dCs = (k_sed / R) * Cw[i] - (k_resus + k_bur) * Cs[i]
        Cw[i + 1] = max(Cw[i] + dCw * dt, 0)
        Cs[i + 1] = max(Cs[i] + dCs * dt, 0)
    return t / 365.25, Cw, Cs

def flux_budget(rt, source=1.0, **kwargs):
    Cw, Cs = steady_state(rt, source=source, **kwargs)
    rt_arr = np.asarray(rt, dtype=float)
    R = kwargs.get("Hs", 0.10) / kwargs.get("H", 10.0)
    k_flush = 1.0 / rt_arr
    k_sed = kwargs.get("k_sed", 1.5e-3)
    k_resus = kwargs.get("k_resus", 3.5e-4)
    flush = k_flush * Cw
    settling = k_sed * Cw
    resus = k_resus * R * Cs
    net_sed = np.maximum(settling - resus, 0)
    total_loss = np.maximum(flush + net_sed, 1e-12)
    return flush / total_loss, net_sed / total_loss, resus / np.maximum(source, 1e-12)

def main():
    rt = np.logspace(np.log10(30), np.log10(5000), 220)
    rng = np.random.default_rng(42)
    curves = []
    for _ in range(1200):
        k_sed = 10 ** rng.uniform(-3.5, -2.0)
        k_resus = 10 ** rng.uniform(-4.2, -3.0)
        k_bur = 10 ** rng.uniform(-4.2, -3.0)
        c, _ = steady_state(rt, k_sed=k_sed, k_resus=k_resus, k_bur=k_bur)
        c100, _ = steady_state(np.array([100.0]), k_sed=k_sed, k_resus=k_resus, k_bur=k_bur)
        curves.append(c / c100[0])
    qs = np.percentile(np.vstack(curves), [5, 25, 50, 75, 95], axis=0)
    out = Path(__file__).resolve().parents[3] / 'outputs/box_model'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(rt_days=rt, q05=qs[0], q25=qs[1], median=qs[2], q75=qs[3], q95=qs[4])).to_csv(out/'residence_time_amplification.csv', index=False)
    print(out/'residence_time_amplification.csv')

if __name__ == '__main__':
    main()
