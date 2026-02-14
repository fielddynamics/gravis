"""
Galaxy catalog with enterprise-grade observational data.

DISTRIBUTED MASS MODEL (Hernquist bulge + exponential disks):
  Each galaxy decomposed into 3 independently measured components:
    1. Stellar bulge: Hernquist profile, M(<r) = M * r^2 / (r+a)^2
    2. Stellar disk: Exponential, M(<r) = M * [1 - (1+r/Rd)*exp(-r/Rd)]
    3. Gas disk (HI+H2+He): Same form, typically more extended

BARYONIC MASS PROVENANCE (independently measured, NOT fitted to rotation curves):
  Milky Way: Bland-Hawthorn+2016, McMillan 2017, Kalberla+2009
  SPARC galaxies (NGC 3198, NGC 2403, NGC 6503): VizieR J/AJ/152/157/table1
    M_baryon = L[3.6] * (M*/L) + 1.33 * M_HI
    M*/L = 0.5 M_sun/L_sun at 3.6um (stellar population models, NOT fitted)
    1.33 factor = primordial He abundance correction (Big Bang nucleosynthesis)
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
            "disk":  {"M": 4.77e9, "Rd": 1.7},
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
    {
        "id": "m33",
        "name": "M33 Triangulum (M = 5x10^9 M_sun)",
        "distance": 20,
        "mass": 9.7,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 0.1e9, "a": 0.2},
            "disk":  {"M": 2.0e9, "Rd": 1.4},
            "gas":   {"M": 2.9e9, "Rd": 7.0},
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
]

# Simple prediction examples (no mass model / observations)
SIMPLE_PREDICTION_GALAXIES = [
    {
        "id": "ic2574",
        "name": "IC 2574 Dwarf (M = 1x10^9 M_sun)",
        "distance": 12,
        "mass": 9.0,
        "accel": 1.0,
    },
    {
        "id": "generic_dwarf",
        "name": "Generic Dwarf (M = 1x10^9 M_sun)",
        "distance": 15,
        "mass": 9.0,
        "accel": 1.0,
    },
]

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
        "name": "UGC 2885 Giant (v = 300 km/s at 50 kpc)",
        "distance": 50,
        "velocity": 300,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 5.0e10, "a": 2.0},
            "disk":  {"M": 1.0e11, "Rd": 12.0},
            "gas":   {"M": 2.0e10, "Rd": 20.0},
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
        "id": "m33_inference",
        "name": "M33 Triangulum (v = 112 km/s at 8 kpc)",
        "distance": 8,
        "velocity": 112,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 0.1e9, "a": 0.2},
            "disk":  {"M": 2.0e9, "Rd": 1.4},
            "gas":   {"M": 2.9e9, "Rd": 7.0},
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
