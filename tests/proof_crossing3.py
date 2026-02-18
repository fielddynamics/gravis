"""
Proof v3: Pure geometric throat detection from GFD base + observations.

Method 1 (Ratio): v_obs / v_base ratio at each observation point.
  Inner region: ratio ~ 1 (GFD base matches).
  Outer region: ratio > 1 (structural correction lifts observations).
  Throat = where ratio crosses 1.0 (observations diverge from base).

Method 2 (Band crossing): Using the diagnostic band (4/pi)^(1/4):
  Curve A: line from lower-band at inner edge to upper-band at outer edge
  Curve B: line from upper-band at inner edge to lower-band at outer edge
  Crossing = geometric throat.

Method 3 (Residual sign change): v_obs - v_base changes sign.
  Where the residual goes from negative to positive = throat.
"""
import math
import sys
sys.path.insert(0, '.')

from physics.constants import G, M_SUN, KPC_TO_M, A0
from physics.aqual import solve_x as aqual_solve_x
from data.galaxies import PREDICTION_GALAXIES

THROAT_YN = (4.0 / 13.0) * (9.0 / 10.0)
GF = (4.0 / math.pi) ** 0.25  # 1.0622


def enc_bulge(r, M, a):
    if M <= 0 or a <= 0 or r <= 0:
        return 0.0
    return M * r * r / ((r + a) ** 2)


def enc_disk(r, M, Rd):
    if M <= 0 or Rd <= 0 or r <= 0:
        return 0.0
    x = r / Rd
    return M * (1.0 - (1.0 + x) * math.exp(-x))


def gfd_vel(r, mm, a0):
    b = mm.get('bulge', {})
    d = mm.get('disk', {})
    g = mm.get('gas', {})
    enc = (enc_bulge(r, b.get('M', 0), b.get('a', 0))
           + enc_disk(r, d.get('M', 0), d.get('Rd', 0))
           + enc_disk(r, g.get('M', 0), g.get('Rd', 0)))
    r_m = r * KPC_TO_M
    if r_m <= 0 or enc <= 0:
        return 0.0
    gN = G * enc * M_SUN / (r_m * r_m)
    y = gN / a0
    x = aqual_solve_x(y)
    return math.sqrt(a0 * x * r_m) / 1000.0


def find_topo_throat(mm, a0):
    b = mm.get('bulge', {})
    d = mm.get('disk', {})
    g = mm.get('gas', {})

    def _enc(r):
        return (enc_bulge(r, b.get('M', 0), b.get('a', 0))
                + enc_disk(r, d.get('M', 0), d.get('Rd', 0))
                + enc_disk(r, g.get('M', 0), g.get('Rd', 0)))

    def _yN(r):
        e = _enc(r)
        r_m = r * KPC_TO_M
        if r_m <= 0 or e <= 0:
            return 0.0
        return G * e * M_SUN / (r_m * r_m * a0)

    if _yN(0.01) < THROAT_YN:
        return None
    lo, hi = 0.01, 500.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if _yN(mid) > THROAT_YN:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def main():
    print('=' * 82)
    print('PROOF v3: GEOMETRIC THROAT FROM GFD BASE + OBSERVATIONS')
    print()
    print('At each observation point, compute v_obs/v_base.')
    print('Throat = where this ratio crosses 1.0')
    print('(observations start exceeding GFD base = structural correction)')
    print('=' * 82)
    print()

    summary = []

    for gal in PREDICTION_GALAXIES:
        name = gal['name'].split('(')[0].strip()
        mm = gal.get('mass_model')
        obs = gal.get('observations', [])
        if not mm or not obs or len(obs) < 6:
            continue

        a0 = A0 * gal.get('accel', 1.0)

        pts = [(o['r'], o['v'], max(o.get('err', 5.0), 1.0))
               for o in obs if o.get('r', 0) > 0 and o.get('v', 0) > 0]
        pts.sort(key=lambda x: x[0])

        # Compute ratio and residual at each point
        data = []
        for r, v_obs, err in pts:
            v_base = gfd_vel(r, mm, a0)
            if v_base > 0:
                ratio = v_obs / v_base
                resid = v_obs - v_base
                data.append((r, v_obs, v_base, ratio, resid, err))

        if len(data) < 4:
            continue

        # METHOD 1: Find where ratio crosses 1.0
        # Look for the LAST crossing from below to above (or near 1.0)
        ratio_crossing = None
        for i in range(len(data) - 1):
            r1, _, _, rat1, _, _ = data[i]
            r2, _, _, rat2, _, _ = data[i + 1]
            if rat1 <= 1.0 and rat2 > 1.0:
                # Linear interpolation
                if abs(rat2 - rat1) > 1e-6:
                    frac = (1.0 - rat1) / (rat2 - rat1)
                    ratio_crossing = r1 + frac * (r2 - r1)
                else:
                    ratio_crossing = (r1 + r2) / 2.0

        # METHOD 2: Find where residual crosses zero (negative -> positive)
        resid_crossing = None
        for i in range(len(data) - 1):
            r1, _, _, _, res1, _ = data[i]
            r2, _, _, _, res2, _ = data[i + 1]
            if res1 <= 0 and res2 > 0:
                if abs(res2 - res1) > 1e-6:
                    frac = (0 - res1) / (res2 - res1)
                    resid_crossing = r1 + frac * (r2 - r1)
                else:
                    resid_crossing = (r1 + r2) / 2.0

        # METHOD 3: Band crossing
        # Line A: from v_base/GF at r_min to v_base*GF at r_max
        # Line B: from v_base*GF at r_min to v_base/GF at r_max
        r_min = data[0][0]
        r_max = data[-1][0]
        v_base_min = data[0][2]
        v_base_max = data[-1][2]
        band_lo_start = v_base_min / GF
        band_hi_start = v_base_min * GF
        band_lo_end = v_base_max / GF
        band_hi_end = v_base_max * GF

        # Line A: (r_min, band_lo_start) to (r_max, band_hi_end)
        # Line B: (r_min, band_hi_start) to (r_max, band_lo_end)
        # y_A(r) = band_lo_start + (band_hi_end - band_lo_start)
        #          * (r - r_min) / (r_max - r_min)
        # y_B(r) = band_hi_start + (band_lo_end - band_hi_start)
        #          * (r - r_min) / (r_max - r_min)
        # Crossing: y_A = y_B
        dr = r_max - r_min
        if dr > 0:
            slope_A = (band_hi_end - band_lo_start) / dr
            slope_B = (band_lo_end - band_hi_start) / dr
            if abs(slope_A - slope_B) > 1e-10:
                t = (band_hi_start - band_lo_start) / (slope_A - slope_B)
                band_crossing = r_min + t
                if band_crossing < r_min or band_crossing > r_max:
                    band_crossing = None
            else:
                band_crossing = None
        else:
            band_crossing = None

        R_t_topo = find_topo_throat(mm, a0)
        gr = gal.get('galactic_radius', 0)
        R_t_cat = 0.30 * gr

        print('--- %s ---' % name)
        print()
        print('  r(kpc)  v_obs  v_base  ratio   residual')
        print('  ------  -----  ------  -----   --------')
        for r, vo, vb, rat, res, err in data:
            marker = ''
            if ratio_crossing and abs(r - ratio_crossing) < 1.0:
                marker = '  <-- ratio~1'
            print('  %6.1f  %5.0f  %6.1f  %.3f   %+6.1f%s'
                  % (r, vo, vb, rat, res, marker))
        print()

        # Compute mean inner/outer ratios
        mid_idx = len(data) // 2
        inner_rats = [d[3] for d in data[:mid_idx]]
        outer_rats = [d[3] for d in data[mid_idx:]]
        mean_inner = sum(inner_rats) / len(inner_rats) if inner_rats else 0
        mean_outer = sum(outer_rats) / len(outer_rats) if outer_rats else 0

        print('  Mean inner ratio: %.3f (should be ~1.0 if mass model OK)'
              % mean_inner)
        print('  Mean outer ratio: %.3f (>1 = structural boost)'
              % mean_outer)
        print('  Boost: %+.1f%%' % ((mean_outer / mean_inner - 1) * 100)
              if mean_inner > 0 else '')
        print()

        rc_s = '%.2f' % ratio_crossing if ratio_crossing else '--'
        rr_s = '%.2f' % resid_crossing if resid_crossing else '--'
        bc_s = '%.2f' % band_crossing if band_crossing else '--'
        topo_s = '%.2f' % R_t_topo if R_t_topo else 'deep'
        print('  Ratio crossing (v_obs/v_base = 1):  %s kpc' % rc_s)
        print('  Residual zero-cross:                %s kpc' % rr_s)
        print('  Band diagonal crossing:             %s kpc' % bc_s)
        print('  Topological R_t (y_N = 18/65):      %s kpc' % topo_s)
        print('  Catalog R_t (0.30 * R_env):         %.2f kpc' % R_t_cat)
        print()

        summary.append({
            'name': name,
            'ratio_x': ratio_crossing,
            'resid_x': resid_crossing,
            'band_x': band_crossing,
            'topo': R_t_topo,
            'cat': R_t_cat,
            'boost': (mean_outer / mean_inner - 1) * 100
            if mean_inner > 0 else 0,
        })

    # Summary table
    print()
    print('=' * 82)
    print('SUMMARY: Three geometric throat estimates vs topological vs catalog')
    print('=' * 82)
    print()
    fmt = '%-16s  %7s  %7s  %7s  %7s  %7s  %6s'
    print(fmt % ('Galaxy', 'ratio=1', 'resid=0', 'band_x',
                 'topo', 'cat', 'boost'))
    print('-' * 82)
    for s in summary:
        def _f(v):
            return '%7.2f' % v if v is not None else '     --'
        print('%-16s  %s  %s  %s  %s  %7.2f  %+.0f%%' % (
            s['name'][:16],
            _f(s['ratio_x']),
            _f(s['resid_x']),
            _f(s['band_x']),
            _f(s['topo']),
            s['cat'],
            s['boost'],
        ))


if __name__ == '__main__':
    main()
