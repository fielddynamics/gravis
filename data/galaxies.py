"""
Galaxy catalog with enterprise-grade observational data.

DISTRIBUTED MASS MODEL (Hernquist bulge + exponential disks):
  Each galaxy decomposed into 3 independently measured components:
    1. Stellar bulge: Hernquist profile, M(<r) = M * r^2 / (r+a)^2
    2. Stellar disk: Exponential, M(<r) = M * [1 - (1+r/Rd)*exp(-r/Rd)]
    3. Gas disk (HI+H2+He): Same form, typically more extended

BARYONIC MASS PROVENANCE (independently measured, NOT fitted to rotation curves):
  Milky Way: Bland-Hawthorn+2016, McMillan 2017, Kalberla+2009
  SPARC galaxies (NGC 3198, NGC 2403, NGC 6503, NGC 3109, DDO 154, UGC 2885):
    VizieR J/AJ/152/157/table1
    M_baryon = L[3.6] * (M*/L) + 1.33 * M_HI
    M*/L = 0.5 M_sun/L_sun at 3.6um (stellar population models, NOT fitted)
    1.33 factor = primordial He abundance correction (Big Bang nucleosynthesis)
  UGC 2885: SPARC Lelli+2016, Rubin+1980, Carvalho+2024 A&A 692 A105
  M31: Tamm+2012 A&A 546 A4, Barmby+2006, Braun+2009
  M33: Corbelli+2003/2014 (photometric + HI)

ROTATION CURVE DATA: 1-sigma error bars from published measurements
  Primary sources: SPARC (Lelli+2016), THINGS (de Blok+2008), Gaia DR3 (Jiao+2023)

IMPORTANT: No unicode characters (Windows charmap constraint).
"""

import math

# Each galaxy entry contains:
#   id: unique identifier
#   name: display name
#   mode: "prediction" or "inference"
#   distance: max radius for plotting (kpc)
#   mass: log10(M_baryon / M_sun)
#   accel: acceleration ratio (a/a0)
#   mass_model: distributed mass model dict
#   observations: list of {r, v, err} dicts (1-sigma error bars)
#   references: list of reference strings

PREDICTION_GALAXIES = [
    # === MAJOR SPIRALS ===
    {
        "id": "milky_way",
        "name": "Milky Way (M = 7.5x10^10 M_sun)",
        "distance": 30,
        "mass": 10.875,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 1.5e10, "a": 0.6},
            "disk":  {"M": 5.0e10, "Rd": 2.5},
            "gas":   {"M": 1.0e10, "Rd": 5.0},
        },
        "observations": [
            {"r": 2,  "v": 206, "err": 25},
            {"r": 5,  "v": 236, "err": 7},
            {"r": 8,  "v": 230, "err": 3},
            {"r": 10, "v": 228, "err": 5},
            {"r": 15, "v": 221, "err": 7},
            {"r": 20, "v": 213, "err": 10},
            {"r": 25, "v": 200, "err": 15},
        ],
        "references": [
            "Bland-Hawthorn & Gerhard 2016 ARAA 54 529 (bulge)",
            "McMillan 2017 MNRAS 465 76 (disk)",
            "Kalberla & Kerp 2009 ARAA 47 27 (gas)",
            "Eilers+2019 ApJ 871 120 (rotation curve)",
            "Jiao+2023 A&A 678 A208 (Gaia DR3)",
            "Sofue 2020 Galaxies 8 37 (inner region)",
        ],
    },
    {
        "id": "m31",
        "name": "Andromeda M31 (M = 1.1x10^11 M_sun)",
        "distance": 40,
        "mass": 11.04,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 3.0e10, "a": 1.0},
            "disk":  {"M": 7.0e10, "Rd": 5.5},
            "gas":   {"M": 1.0e10, "Rd": 12.0},
        },
        "observations": [
            {"r": 8,  "v": 250, "err": 12},
            {"r": 12, "v": 260, "err": 8},
            {"r": 15, "v": 260, "err": 8},
            {"r": 20, "v": 245, "err": 10},
            {"r": 25, "v": 230, "err": 12},
            {"r": 30, "v": 235, "err": 12},
            {"r": 35, "v": 250, "err": 15},
        ],
        "references": [
            "Tamm+2012 A&A 546 A4 (stellar mass)",
            "Kent 1989 (bulge)",
            "Barmby+2006 (3.6um photometry)",
            "Braun+2009 ApJ 695 937 (HI gas)",
            "Chemin+2009 ApJ 705 1395 (rotation curve)",
            "Corbelli+2010 A&A 511 A89 (tilted ring model)",
        ],
    },
    {
        "id": "ngc3198",
        "name": "NGC 3198 (M = 3.4x10^10 M_sun)",
        "distance": 30,
        "mass": 10.526,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 1.0e9,   "a": 0.3},
            "disk":  {"M": 18.14e9, "Rd": 3.0},
            "gas":   {"M": 14.46e9, "Rd": 6.0},
        },
        "observations": [
            {"r": 3,  "v": 110, "err": 8},
            {"r": 5,  "v": 145, "err": 5},
            {"r": 8,  "v": 150, "err": 3},
            {"r": 10, "v": 152, "err": 3},
            {"r": 15, "v": 150, "err": 3},
            {"r": 20, "v": 150, "err": 4},
            {"r": 25, "v": 150, "err": 5},
            {"r": 30, "v": 149, "err": 6},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016)",
            "Begeman 1989 A&A 223 47 (WSRT 21cm)",
            "de Blok+2008 AJ 136 2648 (THINGS)",
        ],
    },
    {
        "id": "ngc2403",
        "name": "NGC 2403 (M = 9.3x10^9 M_sun)",
        "distance": 22,
        "mass": 9.967,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 0.25e9, "a": 0.3},
            "disk":  {"M": 4.77e9, "Rd": 1.29},
            "gas":   {"M": 4.25e9, "Rd": 5.0},
        },
        "observations": [
            {"r": 1,  "v": 65,  "err": 10},
            {"r": 2,  "v": 100, "err": 8},
            {"r": 5,  "v": 130, "err": 5},
            {"r": 8,  "v": 134, "err": 3},
            {"r": 10, "v": 136, "err": 3},
            {"r": 15, "v": 135, "err": 4},
            {"r": 20, "v": 134, "err": 5},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016)",
            "Fraternali+2002 AJ 123 3124 (VLA HI)",
            "de Blok+2008 AJ 136 2648 (THINGS)",
        ],
    },
    # === GAS-DOMINATED DWARFS ===
    {
        "id": "ddo154",
        "name": "DDO 154 (M = 5x10^8 M_sun)",
        "distance": 10,
        "mass": 8.701,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 1.95e7, "Rd": 0.40},
            "gas":   {"M": 4.83e8, "Rd": 2.0},
        },
        "observations": [
            {"r": 0.5, "v": 15, "err": 3},
            {"r": 1.0, "v": 24, "err": 2},
            {"r": 1.5, "v": 31, "err": 2},
            {"r": 2.0, "v": 35, "err": 2},
            {"r": 2.5, "v": 38, "err": 2},
            {"r": 3.0, "v": 39, "err": 2},
            {"r": 4.0, "v": 42, "err": 2},
            {"r": 5.0, "v": 44, "err": 2},
            {"r": 6.0, "v": 45, "err": 3},
            {"r": 7.0, "v": 46, "err": 4},
            {"r": 7.5, "v": 46, "err": 5},
        ],
        "references": [
            "SPARC: Lelli+2016 AJ 152 157",
            "Carignan & Beaulieu 1989 ApJ 347 760",
            "Oh+2015 AJ 149 180 (LITTLE THINGS)",
        ],
    },
    {
        "id": "ic2574",
        "name": "IC 2574 (M = 1.1x10^9 M_sun)",
        "distance": 13,
        "mass": 9.051,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 9.31e7, "Rd": 1.70},
            "gas":   {"M": 1.032e9, "Rd": 4.0},
        },
        "observations": [
            {"r": 0.5, "v": 10, "err": 5},
            {"r": 1.0, "v": 17, "err": 4},
            {"r": 2.0, "v": 30, "err": 3},
            {"r": 3.0, "v": 40, "err": 3},
            {"r": 4.0, "v": 48, "err": 3},
            {"r": 5.0, "v": 54, "err": 3},
            {"r": 6.0, "v": 59, "err": 3},
            {"r": 7.0, "v": 62, "err": 3},
            {"r": 8.0, "v": 65, "err": 3},
            {"r": 9.0, "v": 66, "err": 4},
            {"r": 10.0, "v": 67, "err": 4},
            {"r": 11.0, "v": 66, "err": 5},
        ],
        "references": [
            "SPARC: Lelli+2016 AJ 152 157",
            "Oh+2008 AJ 136 2761 (THINGS)",
            "Walter+2008 AJ 136 2563 (THINGS)",
        ],
    },
    {
        "id": "ngc3109",
        "name": "NGC 3109 (M = 1.1x10^9 M_sun)",
        "distance": 8,
        "mass": 9.04,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 1.0e8, "Rd": 1.5},
            "gas":   {"M": 1.0e9, "Rd": 3.5},
        },
        "observations": [
            {"r": 0.26, "v": 6,  "err": 2},
            {"r": 0.52, "v": 11, "err": 1},
            {"r": 0.77, "v": 15, "err": 2},
            {"r": 1.03, "v": 19, "err": 2},
            {"r": 1.29, "v": 24, "err": 1},
            {"r": 1.55, "v": 30, "err": 2},
            {"r": 1.81, "v": 34, "err": 2},
            {"r": 2.06, "v": 38, "err": 4},
            {"r": 2.32, "v": 43, "err": 6},
            {"r": 2.58, "v": 42, "err": 3},
            {"r": 2.84, "v": 47, "err": 4},
            {"r": 3.10, "v": 49, "err": 4},
            {"r": 3.35, "v": 51, "err": 4},
            {"r": 3.61, "v": 53, "err": 4},
            {"r": 3.87, "v": 55, "err": 3},
            {"r": 4.13, "v": 55, "err": 3},
            {"r": 4.38, "v": 58, "err": 3},
            {"r": 4.64, "v": 59, "err": 2},
            {"r": 4.90, "v": 61, "err": 2},
            {"r": 5.16, "v": 63, "err": 2},
            {"r": 5.42, "v": 64, "err": 3},
            {"r": 5.67, "v": 67, "err": 3},
            {"r": 5.93, "v": 66, "err": 3},
            {"r": 6.19, "v": 66, "err": 3},
            {"r": 6.45, "v": 67, "err": 3},
        ],
        "references": [
            "SPARC: Lelli+2016 AJ 152 157",
            "Jobin & Carignan 1990 AJ 100 648 (HI rotation curve)",
            "Carignan 1985 ApJ 299 59 (WSRT HI)",
            "Valenzuela+2007 ApJ 657 773 (CDM cusp-core analysis)",
            "Carignan+2013 AJ 146 48 (KAT-7 HI)",
        ],
    },
    {
        "id": "ngc6503",
        "name": "NGC 6503 (M = 8.7x10^9 M_sun)",
        "distance": 18,
        "mass": 9.942,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 0.6e9,  "a": 0.3},
            "disk":  {"M": 5.82e9, "Rd": 1.7},
            "gas":   {"M": 2.32e9, "Rd": 4.0},
        },
        "observations": [
            {"r": 1,  "v": 50,  "err": 10},
            {"r": 2,  "v": 80,  "err": 6},
            {"r": 3,  "v": 95,  "err": 5},
            {"r": 5,  "v": 112, "err": 4},
            {"r": 7,  "v": 118, "err": 3},
            {"r": 10, "v": 121, "err": 3},
            {"r": 13, "v": 120, "err": 4},
            {"r": 16, "v": 118, "err": 5},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016)",
            "Bottema & Gerritsen 1997 MNRAS 290 585",
            "de Blok+2008 AJ 136 2648 (THINGS)",
            "Greisen+2009 AJ 137 4718",
        ],
    },
    # === GIANT SPIRALS ===
    {
        "id": "ugc2885",
        "name": "UGC 2885 Rubin's Galaxy (M = 2.5x10^11 M_sun)",
        "distance": 80,
        "mass": 11.40,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 5.0e10, "a": 1.5},
            "disk":  {"M": 1.0e11, "Rd": 6.0},
            "gas":   {"M": 1.0e11, "Rd": 22.0},
        },
        "observations": [
            {"r": 1.70,  "v": 305, "err": 10},
            {"r": 3.41,  "v": 257, "err": 10},
            {"r": 6.82,  "v": 256, "err": 10},
            {"r": 13.67, "v": 271, "err": 10},
            {"r": 19.49, "v": 282, "err": 10},
            {"r": 23.36, "v": 287, "err": 10},
            {"r": 27.24, "v": 283, "err": 10},
            {"r": 31.22, "v": 280, "err": 10},
            {"r": 35.10, "v": 280, "err": 10},
            {"r": 38.97, "v": 281, "err": 10},
            {"r": 42.85, "v": 282, "err": 10},
            {"r": 46.73, "v": 287, "err": 10},
            {"r": 50.71, "v": 292, "err": 10},
            {"r": 54.58, "v": 298, "err": 10},
            {"r": 58.46, "v": 298, "err": 10},
            {"r": 62.34, "v": 298, "err": 10},
            {"r": 66.21, "v": 298, "err": 10},
            {"r": 70.19, "v": 298, "err": 10},
            {"r": 74.07, "v": 298, "err": 10},
        ],
        "references": [
            "SPARC: Lelli+2016 AJ 152 157",
            "Rubin, Ford & Thonnard 1980 ApJ 238 471",
            "Roelfsema & Allen 1985 A&A 146 213 (WSRT HI)",
            "Canzian+1993 ApJ 406 457 (H-alpha kinematics)",
            "Carvalho+2024 A&A 692 A105 (multi-wavelength)",
        ],
    },
    {
        "id": "m33",
        "name": "M33 Triangulum (M = 4x10^9 M_sun)",
        "distance": 20,
        "mass": 9.598,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 0.1e9, "a": 0.2},
            "disk":  {"M": 2.0e9, "Rd": 1.4},
            "gas":   {"M": 1.86e9, "Rd": 7.0},
        },
        "observations": [
            {"r": 1,  "v": 45,  "err": 10},
            {"r": 2,  "v": 68,  "err": 8},
            {"r": 4,  "v": 100, "err": 5},
            {"r": 6,  "v": 108, "err": 5},
            {"r": 8,  "v": 112, "err": 5},
            {"r": 10, "v": 117, "err": 5},
            {"r": 12, "v": 122, "err": 6},
            {"r": 14, "v": 128, "err": 8},
            {"r": 16, "v": 130, "err": 10},
        ],
        "references": [
            "Corbelli & Salucci 2000 MNRAS 311 441",
            "Corbelli 2003 MNRAS 342 199",
            "Corbelli 2014 A&A (VLA+GBT)",
        ],
    },
]

# Simple prediction examples (no mass model / observations)
SIMPLE_PREDICTION_GALAXIES = []

INFERENCE_GALAXIES = [
    # Each inference example includes a mass_model "shape" from its prediction
    # counterpart. The absolute masses serve as initial proportions -- the
    # inference engine will scale them to match the observed velocity.
    {
        "id": "mw_inference",
        "name": "Milky Way (v = 230 km/s at 8 kpc)",
        "distance": 8,
        "velocity": 230,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 1.5e10, "a": 0.6},
            "disk":  {"M": 5.0e10, "Rd": 2.5},
            "gas":   {"M": 1.0e10, "Rd": 5.0},
        },
    },
    {
        "id": "m31_inference",
        "name": "Andromeda M31 (v = 260 km/s at 15 kpc)",
        "distance": 15,
        "velocity": 260,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 3.0e10, "a": 1.0},
            "disk":  {"M": 7.0e10, "Rd": 5.5},
            "gas":   {"M": 1.0e10, "Rd": 12.0},
        },
    },
    {
        "id": "ugc2885_inference",
        "name": "UGC 2885 Rubin's Galaxy (v = 298 km/s at 50 kpc)",
        "distance": 50,
        "velocity": 298,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 5.0e10, "a": 1.5},
            "disk":  {"M": 1.0e11, "Rd": 6.0},
            "gas":   {"M": 1.0e11, "Rd": 22.0},
        },
    },
    {
        "id": "ngc3198_inference",
        "name": "NGC 3198 (v = 150 km/s at 20 kpc)",
        "distance": 20,
        "velocity": 150,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 1.0e9,   "a": 0.3},
            "disk":  {"M": 18.14e9, "Rd": 3.0},
            "gas":   {"M": 14.46e9, "Rd": 6.0},
        },
    },
    {
        "id": "ngc6503_inference",
        "name": "NGC 6503 (v = 121 km/s at 10 kpc)",
        "distance": 10,
        "velocity": 121,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 0.6e9,  "a": 0.3},
            "disk":  {"M": 5.82e9, "Rd": 1.7},
            "gas":   {"M": 2.32e9, "Rd": 4.0},
        },
    },
    {
        "id": "ngc3109_inference",
        "name": "NGC 3109 (v = 63 km/s at 5 kpc)",
        "distance": 5,
        "velocity": 63,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 1.0e8, "Rd": 1.5},
            "gas":   {"M": 1.0e9, "Rd": 3.5},
        },
    },
    {
        "id": "m33_inference",
        "name": "M33 Triangulum (v = 112 km/s at 8 kpc)",
        "distance": 8,
        "velocity": 112,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 0.1e9, "a": 0.2},
            "disk":  {"M": 2.0e9, "Rd": 1.4},
            "gas":   {"M": 1.86e9, "Rd": 7.0},
        },
    },
    {
        "id": "ddo154_inference",
        "name": "DDO 154 (v = 44 km/s at 5 kpc)",
        "distance": 5,
        "velocity": 44,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 1.95e7, "Rd": 0.40},
            "gas":   {"M": 4.83e8, "Rd": 2.0},
        },
    },
    {
        "id": "ic2574_inference",
        "name": "IC 2574 (v = 66 km/s at 9 kpc)",
        "distance": 9,
        "velocity": 66,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 9.31e7, "Rd": 1.70},
            "gas":   {"M": 1.032e9, "Rd": 4.0},
        },
    },
]


def get_prediction_galaxies():
    """Return all prediction-mode galaxies (with mass models + observations)."""
    return PREDICTION_GALAXIES + SIMPLE_PREDICTION_GALAXIES


def get_inference_galaxies():
    """Return all inference-mode galaxies."""
    return INFERENCE_GALAXIES


def get_galaxy_by_id(galaxy_id):
    """Look up a galaxy by its unique id across all catalogs."""
    for g in PREDICTION_GALAXIES + SIMPLE_PREDICTION_GALAXIES + INFERENCE_GALAXIES:
        if g["id"] == galaxy_id:
            return g
    return None


def get_all_galaxies():
    """Return all galaxies grouped by mode."""
    return {
        "prediction": get_prediction_galaxies(),
        "inference": get_inference_galaxies(),
    }
