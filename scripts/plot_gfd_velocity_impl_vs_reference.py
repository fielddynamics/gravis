"""
Side-by-side: our GFD field velocity implementation vs reference (gfd_velocity_with_poisson).

Chart 1 (left): Reference "Forward: Mass -> Field Velocity" from internal/big deal/
  gfd_velocity_with_poisson.py (smooth red line from M(r)).
Chart 2 (right): Our pipeline from sparse observations: gfd_field_velocity_curve_covariant_poisson
  (smooth via spline interpolation).

Run from repo root with PYTHONPATH=. (e.g. PowerShell: $env:PYTHONPATH="c:\\...\\gravis"; python scripts/plot_gfd_velocity_impl_vs_reference.py).
Uses same synthetic exponential-disk galaxy. No unicode (Windows charmap).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from physics.constants import G, M_SUN, KPC_TO_M, A0
from physics.sst_topological_velocity import gfd_field_velocity_curve_covariant_poisson

# Reference constants (match internal/big deal/gfd_velocity_with_poisson.py)
COUPLING = 17.0 / 13.0
SOLAR_MASS = M_SUN
KPC = KPC_TO_M


def forward_g_source(r, M):
    """Eq 75 + Gauss: g_source = (17/13) * G * M / r^2. r, M in SI."""
    return COUPLING * G * M / np.asarray(r) ** 2


def forward_g_total(g_source):
    """SST: g_total = sqrt(g_source * a0 + g_source^2)."""
    return np.sqrt(np.asarray(g_source) * A0 + np.asarray(g_source) ** 2)


def forward_velocity(r, M):
    """Forward: M(r) -> v(r) in m/s."""
    gs = forward_g_source(r, M)
    gt = forward_g_total(gs)
    return np.sqrt(np.asarray(r) * gt)


def exponential_disk_mass(r, M_total, R_d):
    """M(<r) = M_total * [1 - (1 + r/R_d) * exp(-r/R_d)]. r, R_d in m."""
    x = np.asarray(r) / R_d
    return M_total * (1.0 - (1.0 + x) * np.exp(-x))


def main():
    # Same setup as reference run_demonstration()
    M_total = 6.0e10 * SOLAR_MASS
    R_d = 3.0 * KPC
    r_kpc = np.linspace(0.5, 30.0, 200)
    r_m = r_kpc * KPC

    M_enc = exponential_disk_mass(r_m, M_total, R_d)
    v_predicted_m_s = forward_velocity(r_m, M_enc)
    v_ref_km = v_predicted_m_s / 1e3

    # Sparse "observations" (every 6th point ~= 33 points)
    step = 6
    obs_idx = np.arange(0, len(r_kpc), step)
    obs_r = r_kpc[obs_idx].tolist()
    obs_v = v_ref_km[obs_idx].tolist()

    # Our implementation: obs -> defraction -> spline interp -> forward equation
    target_radii = r_kpc.tolist()
    our_curve = gfd_field_velocity_curve_covariant_poisson(
        obs_r, obs_v, target_radii, A0, defraction_sigma_frac=0.062)

    # Plot: 2 panels side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Reference (chart 1 style)
    ax1.plot(r_kpc, v_ref_km, '-', color='crimson', linewidth=2.5,
            label='GFD field velocity (Eq 75 + SST)')
    ax1.scatter(obs_r, obs_v, s=24, color='black', alpha=0.6, zorder=5,
                label='Obs (subset)')
    ax1.set_xlabel('Radius [kpc]')
    ax1.set_ylabel('Velocity [km/s]')
    ax1.set_title('Reference: Forward Mass -> Field Velocity')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, None)
    ax1.grid(True, alpha=0.3)

    # Right: Our implementation
    ax2.plot(r_kpc, our_curve, '-', color='darkgreen', linewidth=2.5,
            label='Our impl (covariant + Poisson)')
    ax2.scatter(obs_r, obs_v, s=24, color='black', alpha=0.6, zorder=5,
                label='Obs (subset)')
    ax2.set_xlabel('Radius [kpc]')
    ax2.set_ylabel('Velocity [km/s]')
    ax2.set_title('Our implementation: Obs -> Field Velocity')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, None)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = 'scripts/compare_gfd_velocity_impl_vs_reference.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: %s' % out_path)


if __name__ == '__main__':
    main()
