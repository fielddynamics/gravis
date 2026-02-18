"""Verify auto-map output for Milky Way matches UI results."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data.galaxies import PREDICTION_GALAXIES
from physics.services.rotation.inference import optimize_inference

mw = next(g for g in PREDICTION_GALAXIES if g["id"] == "milky_way")
mm = mw["mass_model"]

print()
print("  BEFORE auto-map (published mass model):")
print("  Bulge: M = {:.4e}, a = {}".format(mm["bulge"]["M"], mm["bulge"]["a"]))
print("  Disk:  M = {:.4e}, Rd = {}".format(mm["disk"]["M"], mm["disk"]["Rd"]))
print("  Gas:   M = {:.4e}, Rd = {}".format(mm["gas"]["M"], mm["gas"]["Rd"]))
print("  Total: {:.4e}".format(
    mm["bulge"]["M"] + mm["disk"]["M"] + mm["gas"]["M"]))
print()

# Published B/T ratio (used by Stage 3)
pub_bt = mm["bulge"]["M"] / (mm["bulge"]["M"] + mm["disk"]["M"])
print("  Published B/T ratio: {:.4f}".format(pub_bt))
print()

result = optimize_inference(
    mass_model=mm,
    max_radius=mw["distance"],
    num_points=100,  # UI default
    observations=mw["observations"],
    accel_ratio=mw["accel"],
    galactic_radius_kpc=mw["galactic_radius"],
)

opt = result["mass_model"]
print("  AFTER auto-map (Stage 3 decomposition):")
print("  Bulge: M = {:.4e}, a = {}".format(
    opt["bulge"]["M"], opt["bulge"]["a"]))
print("  Disk:  M = {:.4e}, Rd = {}".format(
    opt["disk"]["M"], opt["disk"]["Rd"]))
print("  Gas:   M = {:.4e}, Rd = {}".format(
    opt["gas"]["M"], opt["gas"]["Rd"]))
m_total = opt["bulge"]["M"] + opt["disk"]["M"] + opt["gas"]["M"]
print("  Total: {:.4e}".format(m_total))
print()

# Verify B/T ratio preserved
fit_bt = opt["bulge"]["M"] / (opt["bulge"]["M"] + opt["disk"]["M"])
print("  Fitted B/T ratio:  {:.4f} (should match published {:.4f})".format(
    fit_bt, pub_bt))
print()

# Deltas
print("  Component deltas:")
for comp, pub_key, fit_key in [
    ("Bulge Mass", mm["bulge"]["M"], opt["bulge"]["M"]),
    ("Bulge Scale", mm["bulge"]["a"], opt["bulge"]["a"]),
    ("Disk Mass", mm["disk"]["M"], opt["disk"]["M"]),
    ("Disk Scale", mm["disk"]["Rd"], opt["disk"]["Rd"]),
    ("Gas Mass", mm["gas"]["M"], opt["gas"]["M"]),
    ("Gas Scale", mm["gas"]["Rd"], opt["gas"]["Rd"]),
]:
    delta = (fit_key - pub_key) / pub_key * 100 if pub_key else 0
    print("    {:<14s}  {:.4e} -> {:.4e}  ({:+.1f}%)".format(
        comp, pub_key, fit_key, delta))

print()
print("  Throughput: {}".format(result["throughput"]))
print("  Method:     {}".format(result["method"]))
print("  GFD RMS:    {} km/s".format(result["gfd_rms"]))
print("  Sigma RMS:  {} km/s".format(result["rms"]))
print("  Chi2/dof:   {}".format(result["chi2_dof"]))
print()

if result.get("gene_report"):
    print("  Gene Report:")
    print("  {:<14s}  {:>12s}  {:>12s}  {:>8s}  {:>6s}".format(
        "Param", "Prior", "Current", "Delta%", "sigma"))
    print("  " + "-" * 60)
    for g in result["gene_report"]:
        pub = g["published"]
        fit = g["fitted"]
        delta = (fit - pub) / pub * 100 if pub else 0
        print("  {:<14s}  {:>12.4e}  {:>12.4e}  {:>+7.1f}%  {:>5.2f}".format(
            g["name"], pub, fit, delta, g["sigma_excess"]))
print()
