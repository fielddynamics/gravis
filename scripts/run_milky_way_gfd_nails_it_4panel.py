"""
Milky Way: GFD field velocity (Eq 75 + SST) vs Newtonian, with MW observations from catalog.
Observations loaded via get_galaxy_by_id("milky_way"). No unicode (Windows charmap).
Run from repo root with PYTHONPATH=. (e.g. $env:PYTHONPATH="c:\\...\\gravis"; python scripts/run_milky_way_gfd_nails_it_4panel.py)
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt

from data.galaxies import get_galaxy_by_id

# Constants
G = 6.674e-11          # m^3 kg^-1 s^-2
a0 = 1.2e-10           # m s^-2
coupling = 17 / 13      # from Eq 75: (k^2+1)/(kd+1), k=4, d=3
Msun = 1.989e30        # kg
kpc = 3.086e19         # m

# MW baryonic disk
M_disk = 7.52e10 * Msun
R_d = 2.79 * kpc

# Observations from catalog
gal = get_galaxy_by_id("milky_way")
if not gal:
    raise RuntimeError("Milky Way galaxy not found")
obs_list = gal.get("observations", [])
if not obs_list:
    raise RuntimeError("No observations for Milky Way")
r_obs = np.array([float(o["r"]) for o in obs_list])
v_obs = np.array([float(o["v"]) for o in obs_list])
e_obs = np.array([float(o.get("err", 5.0)) for o in obs_list])

# Radial grid
r_kpc = np.linspace(0.5, 55, 500)
r = r_kpc * kpc

# Step 1: Enclosed mass
M = M_disk * (1 - (1 + r/R_d) * np.exp(-r/R_d))

# Step 2: Newtonian velocity
v_newt = np.sqrt(G * M / r) / 1e3

# Step 3: GFD field velocity (Eq 75 + SST)
g_source = coupling * G * M / r**2
g_total = np.sqrt(g_source * a0 + g_source**2)
v_gfd = np.sqrt(r * g_total) / 1e3

# Plot
plt.figure(figsize=(6, 4.5))
plt.plot(r_kpc, v_newt, '--', color='0.5', lw=1.2, label=r'Newtonian ($G_{\rm bare}$)')
plt.plot(r_kpc, v_gfd, '-', color='C3', lw=1.8, label='GFD (Eq. 1 + Eq. 4)')
plt.errorbar(r_obs, v_obs, yerr=e_obs, fmt='o', ms=3.5, color='k',
             ecolor='0.4', elinewidth=0.8, capsize=2, capthick=0.8,
             zorder=5, label='MW observations')
plt.xlabel(r'$r$ [kpc]')
plt.ylabel(r'$v$ [km s$^{-1}$]')
plt.xlim(0, 55)
plt.ylim(0, 290)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.2)
plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, 'panel_a.png')
plt.savefig(out_path, dpi=200)
plt.close()
print('Saved: %s' % out_path)
