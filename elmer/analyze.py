"""Extract neutral plane, curvature and MoS2 strain from an Elmer slice result (results/lines.dat)."""
import sys
import numpy as np

Y_MOS2 = 25.0e-6 + 50e-9 + 0.65e-9 / 2      # MoS2 mid-plane height above the substrate bottom (baseline stack)


def analyze(path, t_sub=25e-6, y_mos2=Y_MOS2):
    d = np.loadtxt(path)
    x, y, exx, exy = d[:, 3], d[:, 4], d[:, 13], d[:, 16]
    out = {}
    for tag, x0 in (("centre", 1e-9), ("x50um", 50e-6)):
        m = np.isclose(x, x0, atol=1e-10)
        sub = m & (y <= t_sub)
        k, c = np.polyfit(y[sub], exx[sub], 1)
        order = np.argsort(y[m])
        e_mos2 = np.interp(y_mos2, y[m][order], exx[m][order])
        out[tag] = dict(kappa=k, zn_um=-c / k * 1e6, eps_mos2=e_mos2, epk_um=e_mos2 / k * 1e6,
                        max_shear=float(np.abs(exy[m]).max()),
                        fit_resid=float(np.abs(np.polyval([k, c], y[sub]) - exx[sub]).max() / abs(k * t_sub)))
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = analyze(p)
        print(p)
        for tag, v in r.items():
            print(f"  {tag:7s} kappa={v['kappa']:.4f} 1/m  z_n={v['zn_um']:.4f} um  eps_MoS2/kappa={v['epk_um']:.4f} um  "
                  f"(eta=0.8 -> {0.8*v['epk_um']:.4f} um)  shear={v['max_shear']:.1e}  nonlinearity={v['fit_resid']:.1e}")
