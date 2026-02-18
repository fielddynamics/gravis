"""
Proof: Inner vs Outer GFD base fit crossing reveals the throat.

Split observations into inner/outer halves.
Fit GFD base independently to each half (mass scaling).
Where the two curves cross = geometric throat.
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


def fit_mass_scale(obs_r, obs_v, obs_w, Mb, ab, Md, Rd, Mg, Rg, a0):
    """Find mass scale factor s that minimises weighted chi2."""
    best_s = 1.0
    best_chi2 = 1e30
    for si in range(1, 501):
        s = si * 0.01
        chi2 = 0.0
        for j in range(len(obs_r)):
            v_pred = gfd_vel(
                obs_r[j], Mb * s, ab, Md * s, Rd, Mg * s, Rg, a0)
            delta = obs_v[j] - v_pred
            chi2 += obs_w[j] * delta * delta
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_s = s
    return best_s


def yN_at(r, Mb, ab, Md, Rd, Mg, Rg, a0):
    enc = enc_bulge(r, Mb, ab) + enc_disk(r, Md, Rd) + enc_disk(r, Mg, Rg)
    r_m = r * KPC_TO_M
    if r_m <= 0 or enc <= 0:
        return 0.0
    return G * enc * M_SUN / (r_m * r_m * a0)


def find_topo_throat(Mb, ab, Md, Rd, Mg, Rg, a0):
    if yN_at(0.01, Mb, ab, Md, Rd, Mg, Rg, a0) < THROAT_YN:
        return None
    lo, hi = 0.01, 500.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if yN_at(mid, Mb, ab, Md, Rd, Mg, Rg, a0) > THROAT_YN:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def main():
    print('=' * 78)
    print('PROOF: INNER vs OUTER GFD BASE FIT -> CROSSING = THROAT')
    print('Split observations into inner/outer halves.')
    print('Fit GFD base independently to each half.')
    print('Where the two curves cross = geometric throat.')
    print('=' * 78)
    print()

    results = []

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

        # Extract valid observations sorted by radius
        pts = [(o['r'], o['v'], max(o.get('err', 5.0), 1.0))
               for o in obs if o.get('r', 0) > 0 and o.get('v', 0) > 0]
        pts.sort(key=lambda x: x[0])
        n = len(pts)
        if n < 6:
            continue

        # Split into inner and outer halves
        mid_idx = n // 2
        inner = pts[:mid_idx]
        outer = pts[mid_idx:]

        inner_r = [p[0] for p in inner]
        inner_v = [p[1] for p in inner]
        inner_w = [1.0 / (p[2] ** 2) for p in inner]

        outer_r = [p[0] for p in outer]
        outer_v = [p[1] for p in outer]
        outer_w = [1.0 / (p[2] ** 2) for p in outer]

        # Fit mass scale to each half
        s_inner = fit_mass_scale(
            inner_r, inner_v, inner_w, Mb, ab, Md, Rd, Mg, Rg, a0)
        s_outer = fit_mass_scale(
            outer_r, outer_v, outer_w, Mb, ab, Md, Rd, Mg, Rg, a0)

        # Find where the two curves cross
        max_r = max(p[0] for p in pts) * 1.2
        crossing_r = None
        prev_diff = None
        steps = 4000
        scan_step = max_r / steps
        for i in range(1, steps + 1):
            r = i * scan_step
            v_A = gfd_vel(
                r, Mb * s_inner, ab, Md * s_inner, Rd,
                Mg * s_inner, Rg, a0)
            v_B = gfd_vel(
                r, Mb * s_outer, ab, Md * s_outer, Rd,
                Mg * s_outer, Rg, a0)
            diff = v_A - v_B
            if prev_diff is not None and prev_diff * diff < 0:
                crossing_r = r - scan_step / 2.0
                break
            prev_diff = diff

        # Topological throat
        R_t_topo = find_topo_throat(Mb, ab, Md, Rd, Mg, Rg, a0)
        gr = gal.get('galactic_radius', 0)
        R_t_cat = 0.30 * gr

        print('--- %s ---' % name)
        print('  Inner half: r = [%.1f .. %.1f] kpc (%d pts), '
              'scale = %.2fx' % (
                  inner_r[0], inner_r[-1], len(inner), s_inner))
        print('  Outer half: r = [%.1f .. %.1f] kpc (%d pts), '
              'scale = %.2fx' % (
                  outer_r[0], outer_r[-1], len(outer), s_outer))

        if s_outer > s_inner:
            pct = (s_outer / s_inner - 1) * 100
            print('  Outer needs %.0f%% MORE mass -> structural boost' % pct)
        else:
            pct = (s_inner / s_outer - 1) * 100
            print('  Inner needs %.0f%% MORE mass -> concentrated' % pct)

        if crossing_r:
            print('  CROSSING POINT:  %.2f kpc' % crossing_r)
        else:
            if s_inner != s_outer:
                print('  No crossing (curves diverge, s_inner=%.2f '
                      's_outer=%.2f)' % (s_inner, s_outer))
            else:
                print('  Identical fits (no structural difference)')

        topo_str = ('%.2f kpc' % R_t_topo) if R_t_topo else 'deep field'
        print('  Topological R_t: %s' % topo_str)
        print('  Catalog R_t:     %.2f kpc' % R_t_cat)

        if crossing_r and R_t_topo:
            diff_topo = (crossing_r - R_t_topo) / R_t_topo * 100
            diff_cat = (crossing_r - R_t_cat) / R_t_cat * 100
            print('  Crossing vs Topo:    %+.1f%%' % diff_topo)
            print('  Crossing vs Catalog: %+.1f%%' % diff_cat)
            results.append((
                name, s_inner, s_outer, crossing_r,
                R_t_topo, R_t_cat, diff_topo, diff_cat))
        elif crossing_r:
            diff_cat = (crossing_r - R_t_cat) / R_t_cat * 100
            print('  Crossing vs Catalog: %+.1f%%' % diff_cat)
            results.append((
                name, s_inner, s_outer, crossing_r,
                None, R_t_cat, None, diff_cat))
        else:
            results.append((
                name, s_inner, s_outer, None,
                R_t_topo, R_t_cat, None, None))

        # Show the velocity profile at key radii for this galaxy
        if crossing_r:
            print()
            print('  r(kpc)   v_inner  v_outer  diff     zone')
            print('  ------   -------  -------  ------   ----')
            check_radii = sorted(set([
                max(0.5, crossing_r - 3),
                max(0.5, crossing_r - 1),
                crossing_r,
                crossing_r + 1,
                crossing_r + 3,
            ]))
            for r in check_radii:
                vA = gfd_vel(r, Mb * s_inner, ab, Md * s_inner, Rd,
                             Mg * s_inner, Rg, a0)
                vB = gfd_vel(r, Mb * s_outer, ab, Md * s_outer, Rd,
                             Mg * s_outer, Rg, a0)
                zone = 'INNER' if r < crossing_r else 'OUTER'
                print('  %6.2f   %7.1f  %7.1f  %+6.1f   %s' % (
                    r, vA, vB, vA - vB, zone))

        print()

    # Summary table
    print()
    print('=' * 78)
    print('SUMMARY')
    print('=' * 78)
    print()
    print('%-22s  s_in  s_out  crossing  R_t_topo  R_t_cat  '
          'xing/topo  xing/cat' % 'Galaxy')
    print('-' * 78)
    for row in results:
        name, si, so, xr, rt, rc, dt, dc = row
        xr_s = '%.2f' % xr if xr else '--'
        rt_s = '%.2f' % rt if rt else '--'
        dt_s = '%+.0f%%' % dt if dt is not None else '--'
        dc_s = '%+.0f%%' % dc if dc is not None else '--'
        print('%-22s  %.2f  %.2f   %6s    %6s    %5.2f    '
              '%6s    %6s' % (name, si, so, xr_s, rt_s, rc, dt_s, dc_s))


if __name__ == '__main__':
    main()
