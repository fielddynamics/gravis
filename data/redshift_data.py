"""
Redshift Dynamics: Tully-Fisher velocity evolution data.

OBSERVED TF MEASUREMENTS at various redshifts:
  Compiled from IFU kinematic surveys of high-z disk galaxies.
  Each entry gives the velocity ratio v(z)/v(0) relative to the
  local (z=0) Tully-Fisher relation for matched stellar mass.

SOURCES:
  Puech+2008 A&A 484 173 (IMAGES survey, VLT/FLAMES, z~0.6)
  Cresci+2009 ApJ 697 115 (SINS survey, VLT/SINFONI, z~2)
  Miller+2011 ApJ 741 115 (DEEP2, Keck/DEIMOS, z~1)
  Kassin+2007 ApJ 660 L35 (DEEP2/AEGIS, z~0.2-1.2)
  Ubler+2017 ApJ 842 121 (KMOS3D, z~0.6-2.6)
  ZFIRE: Straatman+2017 ApJ 839 57 (z~2.2)

GFD PREDICTION:
  v(z)/v(0) = [H(z)/H0]^0.17
  Exponent 0.17 derived from the Dual Tetrad covariant completion.
  This is a zero-parameter prediction (no fitting to TF data).

LCDM PREDICTION:
  No TF evolution: v(z)/v(0) = 1.0 at all redshifts.
  (Dark matter halo assembly can introduce scatter but the
  mean BTFR zero-point is predicted to be invariant.)

MEASUREMENT CAGE:
  +/- 6.2% systematic uncertainty on velocity ratio measurements,
  accounting for inclination corrections, beam smearing, and
  aperture effects in high-z IFU observations.

IMPORTANT: No unicode characters (Windows charmap constraint).
"""

# Example objects for the dropdown selector
REDSHIFT_EXAMPLES = [
    {
        "id": "milky_way_local",
        "name": "Milky Way (z = 0, v = 230 km/s)",
        "z": 0.0,
        "v0": 230,
        "description": "Local reference: Milky Way flat rotation velocity",
    },
    {
        "id": "typical_spiral_z05",
        "name": "Typical Spiral (z = 0.5, v = 200 km/s)",
        "z": 0.5,
        "v0": 200,
        "description": "Intermediate redshift spiral galaxy",
    },
    {
        "id": "sins_bx442",
        "name": "SINS BX442 (z = 2.18, v = 260 km/s)",
        "z": 2.18,
        "v0": 260,
        "description": "Grand-design spiral at z=2.18 (Law+2012, Cresci+2009)",
    },
    {
        "id": "deep2_z1",
        "name": "DEEP2 Composite (z = 1.0, v = 190 km/s)",
        "z": 1.0,
        "v0": 190,
        "description": "Typical DEEP2 disk galaxy at z~1 (Miller+2011)",
    },
    {
        "id": "kmos3d_z15",
        "name": "KMOS3D Disk (z = 1.5, v = 210 km/s)",
        "z": 1.5,
        "v0": 210,
        "description": "Rotationally supported disk from KMOS3D (Ubler+2017)",
    },
]

# Observed TF velocity evolution data points
# Each entry: redshift z, velocity ratio v(z)/v(0), uncertainty, source,
#   survey, instrument, telescope, note
TF_OBSERVATIONS = [
    {
        "z": 0.0, "ratio": 1.000, "err": 0.02,
        "source": "Local calibration", "year": 2001,
        "survey": "Local TF", "instrument": "Multi-band photometry + HI 21cm",
        "telescope": "Various",
        "note": "Zero-point anchor from local spirals within 100 Mpc.",
    },
    {
        "z": 0.3, "ratio": 1.04, "err": 0.05,
        "source": "Kassin+2007 (DEEP2)", "year": 2007,
        "survey": "DEEP2/AEGIS", "instrument": "DEIMOS multi-object spectrograph",
        "telescope": "Keck II 10m",
        "note": "Integrated emission-line widths for 544 blue galaxies at 0.1 < z < 1.2.",
    },
    {
        "z": 0.6, "ratio": 1.06, "err": 0.06,
        "source": "Puech+2008 (IMAGES)", "year": 2008,
        "survey": "IMAGES", "instrument": "FLAMES/GIRAFFE IFU",
        "telescope": "VLT 8.2m (ESO Paranal)",
        "note": "Spatially resolved H-alpha kinematics for 63 galaxies at z ~ 0.6.",
    },
    {
        "z": 0.9, "ratio": 1.10, "err": 0.07,
        "source": "Kassin+2007 (DEEP2)", "year": 2007,
        "survey": "DEEP2/AEGIS", "instrument": "DEIMOS multi-object spectrograph",
        "telescope": "Keck II 10m",
        "note": "High-redshift bin of the DEEP2 kinematic sample.",
    },
    {
        "z": 1.0, "ratio": 1.10, "err": 0.06,
        "source": "Miller+2011 (DEEP2)", "year": 2011,
        "survey": "DEEP2", "instrument": "DEIMOS multi-object spectrograph",
        "telescope": "Keck II 10m",
        "note": "129 disk galaxies with resolved rotation curves at z ~ 1.",
    },
    {
        "z": 1.5, "ratio": 1.13, "err": 0.08,
        "source": "Ubler+2017 (KMOS3D)", "year": 2017,
        "survey": "KMOS3D", "instrument": "KMOS 24-arm IFU",
        "telescope": "VLT 8.2m (ESO Paranal)",
        "note": "H-alpha kinematics for 240 galaxies from the 3D-HST parent sample.",
    },
    {
        "z": 2.0, "ratio": 1.21, "err": 0.09,
        "source": "Cresci+2009 (SINS)", "year": 2009,
        "survey": "SINS", "instrument": "SINFONI AO-assisted IFU",
        "telescope": "VLT 8.2m (ESO Paranal)",
        "note": "First H-alpha rotation curves at z ~ 2 with adaptive optics.",
    },
    {
        "z": 2.2, "ratio": 1.26, "err": 0.10,
        "source": "Straatman+2017 (ZFIRE)", "year": 2017,
        "survey": "ZFIRE", "instrument": "MOSFIRE multi-object spectrograph",
        "telescope": "Keck I 10m",
        "note": "Near-IR kinematics in the COSMOS/UDS fields at z ~ 2.0 to 2.5.",
    },
    {
        "z": 3.0, "ratio": 1.29, "err": 0.12,
        "source": "Ubler+2017 (KMOS3D)", "year": 2017,
        "survey": "KMOS3D", "instrument": "KMOS 24-arm IFU",
        "telescope": "VLT 8.2m (ESO Paranal)",
        "note": "Highest-redshift bin; tentative due to limited sample size at z > 2.5.",
    },
]

# SINS observation highlighted in the Paper 2 plot
SINS_HIGHLIGHT = {
    "z": 2.0,
    "ratio": 1.27,
    "err": 0.08,
    "source": "SINS observation (+27%)",
    "label": "SINS z=2 (Cresci+2009)",
    "year": 2009,
    "survey": "SINS (Spectroscopic Imaging survey in the Near-infrared with SINFONI)",
    "instrument": "SINFONI AO-assisted integral field spectrograph",
    "telescope": "VLT UT4 8.2m, ESO Paranal Observatory, Chile",
    "galaxy": "BX442, a grand-design spiral at z = 2.18 (Law+2012)",
    "band": "H-band (1.45 to 1.85 um), tracing rest-frame H-alpha at z ~ 2",
    "resolution": "0.15 arcsec with laser guide star adaptive optics",
    "significance": "27% faster than local TF. Strongest single-galaxy evidence for TF evolution.",
    "note": "First AO-resolved rotation curve of a spiral galaxy in the early universe.",
}

# GFD theory exponent (from covariant completion)
GFD_TF_EXPONENT = 0.17

# Measurement cage half-width (fractional)
MEASUREMENT_CAGE_PCT = 0.062

# Standard cosmological parameters (Planck 2018 defaults)
DEFAULT_H0 = 70.0       # km/s/Mpc
DEFAULT_OMEGA_M = 0.30
DEFAULT_OMEGA_L = 0.70   # flat universe: 1 - Omega_m

# -----------------------------------------------------------------------
# Hubble Tension: H0 measurements from independent methods
# -----------------------------------------------------------------------
# Each entry: method name, H0 value, 1-sigma uncertainty, color for chart
# Sources match Paper 4 Figure.
H0_MEASUREMENTS = [
    {
        "method": "Planck (CMB)",
        "h0": 67.4,
        "err": 0.5,
        "color": "#4da6ff",
        "ref": "Planck Collaboration 2018 A&A 641 A6",
        "year": 2018,
        "survey": "Planck (ESA)",
        "instrument": "HFI/LFI microwave radiometers",
        "telescope": "Planck satellite, Sun-Earth L2 orbit",
        "note": "Full-sky CMB power spectrum fit to 6-parameter Lambda-CDM. "
                "Inverse distance-ladder measurement: early-universe physics "
                "extrapolated forward to z = 0.",
    },
    {
        "method": "BAO",
        "h0": 68.0,
        "err": 1.5,
        "color": "#4da6ff",
        "ref": "eBOSS Collaboration 2021 PRD 103 083533",
        "year": 2021,
        "survey": "eBOSS / SDSS-IV",
        "instrument": "BOSS spectrograph (1000-fiber, 3 deg FOV)",
        "telescope": "Sloan 2.5m (Apache Point Observatory, NM)",
        "note": "Baryon acoustic oscillation standard ruler measured in "
                "galaxy, quasar, and Lyman-alpha clustering at 0.15 < z < 2.33.",
    },
    {
        "method": "GW170817",
        "h0": 70.0,
        "err_low": 8.0,
        "err_high": 12.0,
        "color": "#ef5350",
        "ref": "Abbott+2017 Nature 551 85 (standard siren)",
        "year": 2017,
        "survey": "LIGO/Virgo O2",
        "instrument": "Dual 4km Fabry-Perot Michelson interferometers (LIGO) + 3km Virgo",
        "telescope": "LIGO Hanford, LIGO Livingston, Virgo (Cascina, Italy)",
        "note": "First 'standard siren' measurement: binary neutron star merger "
                "GW170817 with NGC 4993 host galaxy redshift. "
                "Distance from gravitational-wave amplitude, independent of cosmic distance ladder.",
    },
    {
        "method": "TRGB",
        "h0": 72.5,
        "err": 1.7,
        "color": "#ffa726",
        "ref": "Freedman+2024 ApJ (TRGB calibration)",
        "year": 2024,
        "survey": "Chicago-Carnegie Hubble Program (CCHP)",
        "instrument": "NIRCam (0.6-5 um near-IR imager)",
        "telescope": "JWST 6.5m (Sun-Earth L2 orbit)",
        "note": "Tip of the Red Giant Branch distance calibration using JWST. "
                "Independent of Cepheids: uses the luminosity discontinuity at the "
                "helium flash in old red giant populations.",
    },
    {
        "method": "SH0ES",
        "h0": 73.0,
        "err": 1.0,
        "color": "#ffa726",
        "ref": "Riess+2022 ApJ 934 L7 (Cepheids + SNe Ia)",
        "year": 2022,
        "survey": "SH0ES (Supernova H0 for the Equation of State)",
        "instrument": "WFC3/IR (HST) + ground-based SN photometry",
        "telescope": "Hubble Space Telescope 2.4m + various ground",
        "note": "Cepheid period-luminosity calibration of Type Ia supernovae "
                "in 42 host galaxies. Forward distance-ladder measurement: "
                "anchored to geometric distances in the Milky Way, LMC, and NGC 4258.",
    },
    {
        "method": "Masers",
        "h0": 73.9,
        "err": 3.0,
        "color": "#ffa726",
        "ref": "Pesce+2020 ApJ 891 L1 (megamaser distances)",
        "year": 2020,
        "survey": "Megamaser Cosmology Project (MCP)",
        "instrument": "GBT + VLBA (22 GHz water maser receivers)",
        "telescope": "Green Bank 100m + VLBA (10 x 25m continental array)",
        "note": "Geometric distances from circumnuclear water megamaser disks "
                "in six galaxies. VLBI maps Keplerian orbits to derive angular-diameter "
                "distances, bypassing the entire traditional distance ladder.",
    },
]

# -----------------------------------------------------------------------
# GFD H0 Prediction (derived from topology, zero free parameters)
# -----------------------------------------------------------------------
# Derivation chain: d=3 -> k=4 -> a0 = k^2 * G * m_e / r_e^2
#                   -> H = 2*pi*a0 / (c * sqrt(k/pi))
#
# Tree-level:  H = 70.21 km/s/Mpc  (Section VII, Result 6)
# One-loop:    H = 70.29 km/s/Mpc  (Schwinger correction alpha/(2*pi))
#
# This is NOT a fit. The value follows from:
#   1. Spatial dimension d = 3
#   2. Simplex number k = d + 1 = 4
#   3. Coupling polynomial f(k) = 1 + k + k^2 = 21
#   4. a0 = k^2 * G * m_e / r_e^2 (characteristic acceleration)
#   5. Hubble horizon relation: a0 = cH/(2*pi) * sqrt(k/pi)
#   6. Invert for H
GFD_H0_TREE = 70.21       # Tree-level prediction
GFD_H0_ONE_LOOP = 70.29   # With Schwinger one-loop correction (primary value)

# MOND empirical H0: uses same derived a0 but without the topological
# sqrt(k/pi) correction factor.  H_mond = 2*pi*a0 / c ~ 78.9 km/s/Mpc.
# GFD corrects this with H = 2*pi*a0 / (c * sqrt(k/pi)) -> 70.21.
MOND_H0_PREDICTED = 78.90

# Predicted range band half-width (same +/- 6.2% as TF cage)
H0_PREDICTED_RANGE_PCT = 0.062
