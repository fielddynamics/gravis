"""
Proof v2: Fit GFD base shape separately to inner vs outer observations.

The key insight: if you fit the mass DISTRIBUTION (not just total mass)
to inner observations, you get a curve that peaks early and falls off.
If you fit to outer observations, you get a more extended curve.

These two curves MUST cross somewhere. That crossing is the throat:
the geometric transition from inner-dominated to outer-dominated field.

Method: vary total mass AND disk scale length Rd independently for
inner vs outer halves. This changes the SHAPE, not just amplitude.
"""
import math
import sys
sys.path.insert(0, '.')

from physics.constants import G, M_SUN, KPC_TO_M, A0
from physics.aqual import solve_x as aqual_solve_x
from data.galaxies import PREDICTION_GALAXIES

THROAT_YN = (4.0 / 13.0) * (9.0 / 10.0)


def enc_bulge(r, M, a):
    if M <= 0 or a <= 0 or r <= 0:
        return 0.0
    return M * r * r / ((r + a) ** 2)


def enc_disk(r, M, Rd):
    if M <= 0 or Rd <= 0 or r <= 0:
        return 0.0
    x = r / Rd
    return M * (1.0 - (1.0 + x) * math.exp(-x))


def gfd_vel(r, Mb, ab, Md, Rd, Mg, Rg, a0):
    enc = enc_bulge(r, Mb, ab) + enc_disk(r, Md, Rd) + enc_disk(r, Mg, Rg)
    r_m = r * KPC_TO_M
    if r_m <= 0 or enc <= 0:
        return 0.0
    gN = G * enc * M_SUN / (r_m * r_m)
    y = gN / a0
    x = aqual_solve_x(y)
    return math.sqrt(a0 * x * r_m) / 1000.0


def fit_shape(obs_r, obs_v, obs_w, Mb, ab, Md_pub, Rd_pub, Mg, Rg, a0):
    """Fit total stellar mass AND disk scale Rd to observations.

    Search over mass scale s in [0.1, 5.0] and Rd multiplier in
    [0.3, 3.0]. Returns (best_s, best_rd_mult).
    """
    best_s = 1.0
    best_rm = 1.0
    best_chi2 = 1e30
    for si in range(5, 251, 5):
        s = si * 0.02  # 0.10 to 5.00
        for ri in range(3, 31):
            rm = ri * 0.1  # 0.3 to 3.0
            Rd = Rd_pub * rm
            chi2 = 0.0
            for j in range(len(obs_r)):
                v = gfd_vel(obs_r[j], Mb * s, ab, Md_pub * s, Rd,
                            Mg * s, Rg, a0)
                delta = obs_v[j] - v
                chi2 += obs_w[j] * delta * delta
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_s = s
                best_rm = rm
    return best_s, best_rm


def find_crossing(Mb, ab, Md, Rd_A, Rd_B, Mg, Rg, a0, s_A, s_B, max_r):
    """Find radius where curve A (inner fit) and curve B (outer fit) cross."""
    steps = 5000
    step = max_r / steps
    prev_diff = None
    for i in range(1, steps + 1):
        r = i * step
        vA = gfd_vel(r, Mb * s_A, ab, Md * s_A, Rd_A, Mg * s_A, Rg, a0)
        vB = gfd_vel(r, Mb * s_B, ab, Md * s_B, Rd_B, Mg * s_B, Rg, a0)
        diff = vA - vB
        if prev_diff is not None and prev_diff * diff < 0:
            return r - step / 2.0
        prev_diff = diff
    return None


def find_topo_throat(Mb, ab, Md, Rd, Mg, Rg, a0):
    def _enc(r):
        return (enc_bulge(r, Mb, ab) + enc_disk(r, Md, Rd)
                + enc_disk(r, Mg, Rg))

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
    print('=' * 80)
    print('PROOF v2: SHAPE FIT -> INNER vs OUTER GFD BASE CROSSING')
    print()
    print('Fit mass + disk scale Rd independently to inner/outer halves.')
    print('Inner fit concentrates mass -> peaks early, falls off.')
    print('Outer fit extends mass -> rises slowly, stays high.')
    print('Crossing = geometric throat.')
    print('=' * 80)
    print()

    summary = []

    for gal in PREDICTION_GALAXIES:
        name = gal['name'].split('(')[0].strip()
        mm = gal.get('mass_model')
        obs = gal.get('observations', [])
        if not mm or not obs or len(obs) < 6:
            continue

        a0 = A0 * gal.get('accel', 1.0)
        b = mm.get('bulge', {})
        d = mm.get('disk', {})
        g = mm.get('gas', {})
        Mb, ab = b.get('M', 0), b.get('a', 0)
        Md, Rd = d.get('M', 0), d.get('Rd', 0)
        Mg, Rg = g.get('M', 0), g.get('Rd', 0)

        pts = [(o['r'], o['v'], max(o.get('err', 5.0), 1.0))
               for o in obs if o.get('r', 0) > 0 and o.get('v', 0) > 0]
        pts.sort(key=lambda x: x[0])
        n = len(pts)
        if n < 6:
            continue

        mid_idx = n // 2
        inner = pts[:mid_idx]
        outer = pts[mid_idx:]

        inner_r = [p[0] for p in inner]
        inner_v = [p[1] for p in inner]
        inner_w = [1.0 / (p[2] ** 2) for p in inner]

        outer_r = [p[0] for p in outer]
        outer_v = [p[1] for p in outer]
        outer_w = [1.0 / (p[2] ** 2) for p in outer]

        s_in, rm_in = fit_shape(
            inner_r, inner_v, inner_w, Mb, ab, Md, Rd, Mg, Rg, a0)
        s_out, rm_out = fit_shape(
            outer_r, outer_v, outer_w, Mb, ab, Md, Rd, Mg, Rg, a0)

        Rd_in = Rd * rm_in
        Rd_out = Rd * rm_out

        max_r = max(p[0] for p in pts) * 1.2
        crossing = find_crossing(
            Mb, ab, Md, Rd_in, Rd_out, Mg, Rg, a0, s_in, s_out, max_r)

        R_t_topo = find_topo_throat(Mb, ab, Md, Rd, Mg, Rg, a0)
        gr = gal.get('galactic_radius', 0)
        R_t_cat = 0.30 * gr

        print('--- %s ---' % name)
        print('  Inner fit: mass=%.2fx, Rd=%.1f kpc (%.1fx pub)'
              % (s_in, Rd_in, rm_in))
        print('  Outer fit: mass=%.2fx, Rd=%.1f kpc (%.1fx pub)'
              % (s_out, Rd_out, rm_out))

        if crossing:
            print('  CROSSING:     %.2f kpc' % crossing)
        else:
            print('  No crossing found')
        topo_s = '%.2f kpc' % R_t_topo if R_t_topo else 'deep field'
        print('  Topo throat:  %s' % topo_s)
        print('  Catalog R_t:  %.2f kpc' % R_t_cat)

        if crossing and R_t_topo:
            d1 = (crossing - R_t_topo) / R_t_topo * 100
            d2 = (crossing - R_t_cat) / R_t_cat * 100
            print('  Cross/Topo: %+.1f%%,  Cross/Cat: %+.1f%%' % (d1, d2))
            summary.append((name, crossing, R_t_topo, R_t_cat, d1, d2))
        elif crossing:
            d2 = (crossing - R_t_cat) / R_t_cat * 100
            print('  Cross/Cat: %+.1f%%' % d2)
            summary.append((name, crossing, None, R_t_cat, None, d2))
        else:
            summary.append((name, None, R_t_topo, R_t_cat, None, None))

        # Show curves at key radii near the crossing
        if crossing:
            print()
            print('  r(kpc)  v_inner  v_outer  diff')
            check_radii = []
            for offset in [-3, -1, -0.5, 0, 0.5, 1, 3]:
                cr = crossing + offset
                if cr > 0.2:
                    check_radii.append(cr)
            for r in check_radii:
                vA = gfd_vel(r, Mb * s_in, ab, Md * s_in, Rd_in,
                             Mg * s_in, Rg, a0)
                vB = gfd_vel(r, Mb * s_out, ab, Md * s_out, Rd_out,
                             Mg * s_out, Rg, a0)
                marker = ' <-- CROSSING' if abs(r - crossing) < 0.3 else ''
                print('  %6.2f  %7.1f  %7.1f  %+5.1f%s'
                      % (r, vA, vB, vA - vB, marker))

        print()

    # Summary table
    print()
    print('=' * 80)
    print('SUMMARY: Crossing vs Topological vs Catalog throat')
    print('=' * 80)
    print()
    hdr = '%-22s  crossing  topo_Rt  cat_Rt  xing/topo  xing/cat'
    print(hdr % 'Galaxy')
    print('-' * 80)
    for name, xr, rt, rc, d1, d2 in summary:
        xr_s = '%5.2f' % xr if xr else '  --'
        rt_s = '%5.2f' % rt if rt else '  --'
        d1_s = '%+6.1f%%' % d1 if d1 is not None else '    --'
        d2_s = '%+6.1f%%' % d2 if d2 is not None else '    --'
        print('%-22s  %7s  %7s  %5.2f   %7s   %7s'
              % (name, xr_s, rt_s, rc, d1_s, d2_s))


if __name__ == '__main__':
    main()
