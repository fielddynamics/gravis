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

import json
import math
import os

# Each galaxy entry contains:
#   id: unique identifier
#   name: display name
#   mode: "prediction" or "inference"
#   distance: max radius for plotting (kpc)
#   galactic_radius: gravitational horizon scale (kpc) for manifold computation
#   mass: log10(M_baryon / M_sun)
#   accel: acceleration ratio (a/a0)
#   mass_model: distributed mass model dict
#   observations: list of {r, v, err} dicts (1-sigma error bars)
#   references: list of reference strings

PREDICTION_GALAXIES = [
    # === MAJOR SPIRALS ===
    {
        "id": "milky_way",
        "name": "Milky Way (M = 7.6x10^10 M_sun, 20% gas)",
        "distance": 65,
        "galactic_radius": 60,
        "mass": 10.879,
        "accel": 1.0,
        "mass_model": {
            # Bulge/bar: 1.2-2.0e10 (Portail+2017 dynamical model of bar,
            # OGLE/MOA microlensing, COBE/DIRBE NIR star counts).
            # a = 0.6 kpc from bar half-length ~5 kpc projected.
            "bulge": {"M": 1.5e10, "a": 0.6},
            # Thin + thick disk combined: McMillan 2017 total stellar = 54.3e9,
            # minus bulge 0.91e9 -> disk ~4.52e10. Bovy & Rix 2013:
            # mass-weighted Rd = 2.15 +/- 0.14 kpc (SEGUE G dwarfs).
            # Licquia & Newman 2015: disk = 5.17 +/- 1.11e10.
            # Published range 3.5-5.5e10. Using 4.57e10 (McMillan 2017).
            "disk":  {"M": 4.57e10, "Rd": 2.2},
            # HI: 7.1e9 (Nakanishi & Sofue), H2: 0.9e9 (Dame+2001),
            # He correction: x1.33, warm/hot CGM: 5-10e9 (Bregman+2018).
            # Total ~1.0-2.0e10. HI extends to ~60 kpc (Kalberla+2008),
            # Rd_gas = 7.0 kpc.
            "gas":   {"M": 1.5e10, "Rd": 7.0},
        },
        "observations": [
            # Ou+2023 MNRAS (Table 1): Gaia DR3 + APOGEE DR17,
            # 120,309 RGB stars with spectrophotometric parallaxes.
            # Inner curve (< 5 kpc) excluded: bar contamination.
            # Errors are symmetrised from their +/- values.
            {"r": 6.3,  "v": 231, "err": 1, "src": "Ou+2023"},
            {"r": 7.9,  "v": 234, "err": 1, "src": "Ou+2023"},
            {"r": 9.2,  "v": 230, "err": 1, "src": "Ou+2023"},
            {"r": 10.2, "v": 229, "err": 1, "src": "Ou+2023"},
            {"r": 11.2, "v": 227, "err": 1, "src": "Ou+2023"},
            {"r": 12.2, "v": 227, "err": 1, "src": "Ou+2023"},
            {"r": 13.2, "v": 225, "err": 1, "src": "Ou+2023"},
            {"r": 14.2, "v": 222, "err": 1, "src": "Ou+2023"},
            {"r": 15.2, "v": 218, "err": 1, "src": "Ou+2023"},
            {"r": 16.2, "v": 218, "err": 2, "src": "Ou+2023"},
            {"r": 17.2, "v": 220, "err": 2, "src": "Ou+2023"},
            {"r": 18.2, "v": 215, "err": 2, "src": "Ou+2023"},
            {"r": 19.2, "v": 208, "err": 2, "src": "Ou+2023"},
            {"r": 20.2, "v": 203, "err": 2, "src": "Ou+2023"},
            {"r": 20.7, "v": 195, "err": 2, "src": "Ou+2023"},
            {"r": 21.2, "v": 200, "err": 2, "src": "Ou+2023"},
            {"r": 21.7, "v": 201, "err": 3, "src": "Ou+2023"},
            {"r": 22.3, "v": 197, "err": 6, "src": "Ou+2023"},
            {"r": 23.4, "v": 192, "err": 5, "src": "Ou+2023"},
            {"r": 25.0, "v": 191, "err": 8, "src": "Ou+2023"},
            # Huang+2016 MNRAS 463 2623: Halo K giants (SDSS/SEGUE),
            # Jeans equation with beta constraints. Different tracer
            # population from disk RGB stars; errors are much larger.
            # Values read from their Figure 12 combined RC.
            {"r": 30,  "v": 206, "err": 30, "src": "Huang+2016"},
            {"r": 40,  "v": 192, "err": 35, "src": "Huang+2016"},
            # Ablimit & Zhao 2017 ApJ 846: RR Lyrae halo tracers,
            # enclosed mass -> circular velocity at 50 kpc.
            {"r": 50,  "v": 180, "err": 32, "src": "Ablimit & Zhao 2017"},
        ],
        "references": [
            "Bland-Hawthorn & Gerhard 2016 ARAA 54 529 (bulge/bar review)",
            "Portail+2017 MNRAS 465 1621 (bar dynamical model)",
            "McMillan 2017 MNRAS 465 76 (total stellar 54.3e9)",
            "Licquia & Newman 2016 ApJ 831 71 (disk 4.8e10, Rd 2.64 kpc)",
            "Kalberla & Kerp 2009 ARAA 47 27 (HI distribution)",
            "Nakanishi & Sofue 2016 PASJ 68 5 (HI+H2 total gas)",
            "Ou+2023 MNRAS (Gaia DR3 + APOGEE DR17, Table 1)",
            "Eilers+2019 ApJ 871 120 (rotation curve 5-25 kpc)",
            "Jiao+2023 A&A 678 A208 (Gaia DR3, Keplerian decline)",
            "Huang+2016 MNRAS 463 2623 (HKG rotation curve to 100 kpc)",
            "Ablimit & Zhao 2017 ApJ 846 (RR Lyrae halo, v_c at 50 kpc)",
        ],
    },
    {
        "id": "m31",
        "name": "Andromeda M31 (M = 1.18x10^11 M_sun, 8% gas)",
        "distance": 40,
        "galactic_radius": 40,
        "mass": 11.07,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 3.0e10, "a": 1.0},
            "disk":  {"M": 7.76e10, "Rd": 5.5},
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
        "name": "NGC 3198 (M = 3.4x10^10 M_sun, 43% gas)",
        "distance": 30,
        "galactic_radius": 48,
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
        "name": "NGC 2403 (M = 1.3x10^10 M_sun, 42% gas)",
        "distance": 22,
        "galactic_radius": 20,
        "mass": 10.121,
        "accel": 1.0,
        "mass_model": {
            # Tiny pseudo-bulge. NIR: L_bulge ~ 1e8 L_sun, M/L ~ 1-2.
            "bulge": {"M": 0.2e9, "a": 0.15},
            # SPARC: L[3.6] ~ 6.9e9 L_sun, Rd = 1.73 kpc.
            # M/L range: 0.5 (SPARC default) -> 3.45e9, up to 1.5 -> 10.4e9.
            # de Blok+2008: dynamical fit supports higher M/L.
            # Sanders & McGaugh 2002 MOND fit: 5.2e9.
            # Using 7.5e9 (M/L ~ 1.1), Rd = 3.25 kpc (effective for
            # combined thin+thick disk at D=3.2 Mpc). CAVEAT: high end
            # of published range; independent verification needed.
            "disk":  {"M": 7.5e9, "Rd": 3.25},
            # M_HI = 3.2e9 (de Blok+2008, 21cm flux at D=3.2 Mpc).
            # x1.33 He = 4.3e9. CO: H2 ~ 0.5e9. Warm/ionized ~ 0.7e9.
            # Total 5.5e9. HI has central depression, Rd_gas = 3.0 kpc.
            "gas":   {"M": 5.5e9, "Rd": 3.0},
        },
        "observations": [
            # de Blok+2008 THINGS: VLA B+C+D arrays, ~6 arcsec = 90 pc
            # at D = 3.2 Mpc. Tilted-ring analysis with ROTCUR.
            {"r": 1,  "v": 50,  "err": 5},
            {"r": 2,  "v": 80,  "err": 5},
            {"r": 3,  "v": 100, "err": 5},
            {"r": 4,  "v": 110, "err": 5},
            {"r": 5,  "v": 120, "err": 5},
            {"r": 6,  "v": 125, "err": 5},
            {"r": 7,  "v": 128, "err": 5},
            {"r": 8,  "v": 130, "err": 5},
            {"r": 9,  "v": 131, "err": 5},
            {"r": 10, "v": 132, "err": 5},
            {"r": 12, "v": 133, "err": 5},
            {"r": 14, "v": 134, "err": 5},
            {"r": 16, "v": 135, "err": 5},
            {"r": 18, "v": 135, "err": 6},
            {"r": 20, "v": 134, "err": 8},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016, L[3.6], Rd)",
            "de Blok+2008 AJ 136 2648 (THINGS rotation curve + mass models)",
            "Fraternali+2002 AJ 123 3124 (VLA HI kinematics)",
            "Sanders & McGaugh 2002 ARAA 40 263 (M/L comparison)",
        ],
    },
    # === GAS-DOMINATED DWARFS ===
    {
        "id": "ddo154",
        "name": "DDO 154 (M = 4.3x10^8 M_sun, 93% gas)",
        "distance": 10,
        "galactic_radius": 8,
        "mass": 8.633,
        "accel": 1.0,
        "mass_model": {
            # No bulge in this irregular dwarf.
            "bulge": {"M": 0, "a": 0.1},
            # Tiny stellar disk: B/V photometry L_B ~ 4e7 L_sun,
            # M/L_B ~ 0.5-1.0. Stars are < 10% of mass at all radii.
            # Carignan 1989: disk scale ~0.5 kpc. Using 0.7 kpc
            # (effective including extended low-SB component).
            "disk":  {"M": 3.0e7, "Rd": 0.7},
            # M_HI = 2.7e8 (Carignan & Purton 1998, 21cm integrated flux).
            # x1.33 He = 3.6e8. Small molecular fraction ~4e7. Total ~4e8.
            # Rd_gas = 2.5 kpc from HI surface density profile.
            "gas":   {"M": 4.0e8, "Rd": 2.5},
        },
        "observations": [
            # Carignan & Purton 1998: HI 21cm VLA synthesis (C+D arrays).
            # Tilted-ring model. Regular kinematics, no strong non-circular
            # motions. Very clean system for gravity testing.
            {"r": 0.5, "v": 15, "err": 3},
            {"r": 1.0, "v": 25, "err": 3},
            {"r": 1.5, "v": 33, "err": 3},
            {"r": 2.0, "v": 38, "err": 3},
            {"r": 2.5, "v": 42, "err": 3},
            {"r": 3.0, "v": 45, "err": 3},
            {"r": 3.5, "v": 47, "err": 4},
            {"r": 4.0, "v": 48, "err": 4},
            {"r": 5.0, "v": 49, "err": 5},
            {"r": 6.0, "v": 50, "err": 5},
            {"r": 7.0, "v": 47, "err": 6},
        ],
        "references": [
            "SPARC: Lelli+2016 AJ 152 157 (3.6um photometry)",
            "Carignan & Purton 1998 ApJ 506 125 (HI rotation curve)",
            "Carignan & Beaulieu 1989 ApJ 347 760 (optical + HI)",
            "Oh+2015 AJ 149 180 (LITTLE THINGS)",
        ],
    },
    {
        "id": "ic2574",
        "name": "IC 2574 (M = 1.1x10^9 M_sun, 92% gas)",
        "distance": 13,
        "galactic_radius": 11,
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
        "name": "NGC 3109 (M = 1.1x10^9 M_sun, 91% gas)",
        "distance": 8,
        "galactic_radius": 7,
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
        "name": "NGC 6503 (M = 8.7x10^9 M_sun, 27% gas)",
        "distance": 18,
        "galactic_radius": 17,
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
        "name": "UGC 2885 Rubin's Galaxy (M = 2.5x10^11 M_sun, 40% gas)",
        "distance": 80,
        "galactic_radius": 75,
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
        "name": "M33 Triangulum (M = 7.1x10^9 M_sun, 45% gas)",
        "distance": 20,
        "galactic_radius": 17,
        "mass": 9.851,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 0.4e9, "a": 0.18},
            "disk":  {"M": 3.5e9, "Rd": 1.6},
            "gas":   {"M": 3.2e9, "Rd": 4.0},
        },
        "observations": [
            {"r": 0.5, "v": 30,  "err": 5},
            {"r": 1,   "v": 38,  "err": 8},
            {"r": 1.5, "v": 55,  "err": 7},
            {"r": 2,   "v": 68,  "err": 8},
            {"r": 3,   "v": 88,  "err": 6},
            {"r": 4,   "v": 100, "err": 7},
            {"r": 5,   "v": 105, "err": 5},
            {"r": 6,   "v": 108, "err": 6},
            {"r": 7,   "v": 108, "err": 5},
            {"r": 8,   "v": 108, "err": 6},
            {"r": 9,   "v": 110, "err": 6},
            {"r": 10,  "v": 115, "err": 7},
            {"r": 11,  "v": 117, "err": 6},
            {"r": 12,  "v": 120, "err": 7},
            {"r": 14,  "v": 125, "err": 7},
            {"r": 15,  "v": 128, "err": 8},
            {"r": 17,  "v": 130, "err": 8},
        ],
        "references": [
            "Corbelli & Salucci 2000 MNRAS 311 441",
            "Corbelli 2003 MNRAS 342 199",
            "Corbelli 2014 A&A (VLA+GBT)",
        ],
    },
    {
        "id": "m33_vortex",
        "name": "M33 Vortex (M_disk=1.3e11 M_sun, R_d=4.14 kpc)",
        "distance": 20,
        "galactic_radius": 45,
        "mass": 11.114,
        "accel": 1.0,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.2},
            "disk":  {"M": 1.30e11, "Rd": 4.14},
            "gas":   {"M": 0, "Rd": 4.0},
        },
        "observations": [
            {"r": 8.0, "v": 250.0, "err": 12},
            {"r": 12.0, "v": 260.0, "err": 8},
            {"r": 15.0, "v": 260.0, "err": 8},
            {"r": 20.0, "v": 245.0, "err": 10},
            {"r": 25.0, "v": 230.0, "err": 12},
            {"r": 30.0, "v": 235.0, "err": 12},
            {"r": 35.0, "v": 250.0, "err": 15},
        ],
        "references": [
            "M33 vortex plot: GFD forward (M_disk, R_d) single exponential disk",
        ],
    },
    # === SPARC SPIRALS with well-measured HI radii (Lelli+2016) ===
    # R_HI = HI radius at 1 M_sun/pc^2 column density (SPARC Table 1).
    # galactic_radius (R_env) set to ~1.15 * R_HI following existing catalog
    # convention. Stellar masses at M*/L = 0.5 M_sun/L_sun at 3.6um.
    # Gas masses = 1.33 * M_HI (He correction).
    # Quality flags from SPARC: Q=1 (high), Q=2 (medium).
    #
    # --- Massive Sb/Sbc spirals ---
    {
        "id": "ngc2841",
        "name": "NGC 2841 (M = 1.1x10^11 M_sun, 12% gas)",
        "distance": 68,
        "galactic_radius": 50,
        "mass": 11.029,
        "accel": 1.0,
        "mass_model": {
            # Sb galaxy with significant bulge. B/T ~ 0.20.
            # L[3.6] = 188.1e9 L_sun, M_star = 9.4e10 (M/L=0.5).
            # Bulge Reff ~ 0.9 kpc (compact), a = Reff/1.815 ~ 0.5.
            "bulge": {"M": 2.0e10, "a": 0.5},
            # Disk: Rdisk = 3.64 kpc (SPARC [3.6um]).
            # Stellar disk extends to ~70 kpc (Zhang+2018 Dragonfly).
            "disk":  {"M": 7.5e10, "Rd": 3.64},
            # M_HI = 9.775e9, x1.33 = 1.30e10. RHI = 45.12 kpc.
            # Gas Vcirc peaks beyond 60 kpc, so Rd_gas ~ 20 kpc.
            "gas":   {"M": 1.3e10, "Rd": 20.0},
        },
        "observations": [
            # SPARC: Dicaire+2008 (Fabry-Perot) + Begeman+1991 (WSRT 21cm).
            # D = 14.1 Mpc, Inc = 76 deg, Q = 1.
            # Vflat = 284.8 km/s, RHI = 45.12 kpc, R_last = 63.6 kpc.
            {"r": 3.44,  "v": 285, "err": 14},
            {"r": 4.47,  "v": 306, "err": 6},
            {"r": 5.48,  "v": 308, "err": 8},
            {"r": 6.86,  "v": 317, "err": 8},
            {"r": 8.57,  "v": 323, "err": 9},
            {"r": 10.60, "v": 323, "err": 9},
            {"r": 12.31, "v": 323, "err": 4},
            {"r": 14.35, "v": 319, "err": 4},
            {"r": 16.40, "v": 308, "err": 4},
            {"r": 18.48, "v": 299, "err": 6},
            {"r": 20.57, "v": 299, "err": 6},
            {"r": 24.59, "v": 296, "err": 6},
            {"r": 28.77, "v": 289, "err": 4},
            {"r": 32.79, "v": 288, "err": 4},
            {"r": 36.96, "v": 283, "err": 3},
            {"r": 40.99, "v": 272, "err": 3},
            {"r": 45.16, "v": 274, "err": 4},
            {"r": 49.19, "v": 281, "err": 4},
            {"r": 53.36, "v": 282, "err": 4},
            {"r": 57.38, "v": 288, "err": 6},
            {"r": 61.56, "v": 289, "err": 7},
            {"r": 63.64, "v": 294, "err": 7},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016, Q=1, RHI=45.12 kpc)",
            "Dicaire+2008 MNRAS 385 553 (Fabry-Perot Halpha)",
            "Begeman+1991 MNRAS 249 523 (WSRT 21cm)",
            "Zhang+2018 ApJ 855 78 (Dragonfly, stellar disk to 70 kpc)",
        ],
    },
    {
        "id": "ngc7331",
        "name": "NGC 7331 (M = 1.4x10^11 M_sun, 11% gas)",
        "distance": 40,
        "galactic_radius": 30,
        "mass": 11.146,
        "accel": 1.0,
        "mass_model": {
            # Sb galaxy with counterrotating bulge (Prada+1996).
            # L[3.6] = 250.6e9 L_sun, M_star = 1.25e11 (M/L=0.5).
            # SPARC decomposition has no SBbul, but bulge is known.
            # B/T ~ 0.15 from Prada+1996.
            "bulge": {"M": 2.0e10, "a": 0.8},
            # Rdisk = 5.02 kpc (SPARC [3.6um]).
            "disk":  {"M": 1.05e11, "Rd": 5.02},
            # M_HI = 11.067e9, x1.33 = 1.47e10. RHI = 27.01 kpc.
            # Gas Vcirc peaks ~65 at R=23.5 kpc, Rd_gas ~ 11 kpc.
            "gas":   {"M": 1.47e10, "Rd": 11.0},
        },
        "observations": [
            # SPARC: Begeman+1991 (WSRT) + Begeman+1987.
            # D = 14.7 Mpc, Inc = 75 deg, Q = 1.
            # Vflat = 239.0 km/s, RHI = 27.01 kpc, R_last = 36.3 kpc.
            {"r": 2.67,  "v": 221, "err": 20},
            {"r": 3.74,  "v": 249, "err": 8},
            {"r": 4.81,  "v": 253, "err": 5},
            {"r": 5.88,  "v": 257, "err": 5},
            {"r": 7.48,  "v": 257, "err": 5},
            {"r": 9.62,  "v": 248, "err": 6},
            {"r": 11.74, "v": 246, "err": 6},
            {"r": 13.91, "v": 242, "err": 4},
            {"r": 15.00, "v": 238, "err": 3},
            {"r": 17.07, "v": 233, "err": 3},
            {"r": 19.24, "v": 236, "err": 3},
            {"r": 21.41, "v": 238, "err": 3},
            {"r": 23.48, "v": 238, "err": 3},
            {"r": 25.65, "v": 236, "err": 3},
            {"r": 27.82, "v": 239, "err": 3},
            {"r": 30.98, "v": 236, "err": 4},
            {"r": 33.15, "v": 239, "err": 5},
            {"r": 36.31, "v": 238, "err": 9},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016, Q=1, RHI=27.01 kpc)",
            "Begeman+1991 MNRAS 249 523 (WSRT 21cm)",
            "Begeman 1987 PhD thesis (Groningen)",
            "Prada+1996 ApJ 463 L9 (counterrotating bulge)",
        ],
    },
    {
        "id": "ngc5055",
        "name": "NGC 5055 Sunflower (M = 9.2x10^10 M_sun, 17% gas)",
        "distance": 58,
        "galactic_radius": 40,
        "mass": 10.964,
        "accel": 1.0,
        "mass_model": {
            # Sbc spiral, no significant bulge in SPARC decomposition.
            # L[3.6] = 152.9e9 L_sun, M_star = 7.65e10 (M/L=0.5).
            "bulge": {"M": 1.0e9, "a": 0.3},
            # Rdisk = 3.20 kpc (SPARC [3.6um]).
            "disk":  {"M": 7.55e10, "Rd": 3.20},
            # M_HI = 11.722e9, x1.33 = 1.56e10. RHI = 35.06 kpc.
            # Gas extends to ~40 kpc (Battaglia+2006 warp study).
            # Gas Vcirc peaks ~47 at R=17 kpc, Rd_gas ~ 10 kpc.
            "gas":   {"M": 1.56e10, "Rd": 10.0},
        },
        "observations": [
            # SPARC: Battaglia+2006 (WSRT deep HI) + Blais-Ouellette+2004.
            # D = 9.9 Mpc, Inc = 55 deg, Q = 1.
            # Vflat = 179.0 km/s, RHI = 35.06 kpc, R_last = 54.6 kpc.
            {"r": 0.72,  "v": 125, "err": 18},
            {"r": 1.43,  "v": 162, "err": 9},
            {"r": 2.87,  "v": 193, "err": 2},
            {"r": 4.30,  "v": 190, "err": 1},
            {"r": 5.75,  "v": 201, "err": 1},
            {"r": 7.18,  "v": 204, "err": 1},
            {"r": 8.62,  "v": 206, "err": 2},
            {"r": 11.49, "v": 206, "err": 2},
            {"r": 14.30, "v": 206, "err": 1},
            {"r": 17.19, "v": 203, "err": 4},
            {"r": 20.07, "v": 200, "err": 1},
            {"r": 25.85, "v": 188, "err": 5},
            {"r": 28.74, "v": 184, "err": 1},
            {"r": 34.51, "v": 180, "err": 4},
            {"r": 40.29, "v": 180, "err": 5},
            {"r": 45.92, "v": 179, "err": 2},
            {"r": 51.70, "v": 174, "err": 3},
            {"r": 54.59, "v": 172, "err": 5},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016, Q=1, RHI=35.06 kpc)",
            "Battaglia+2006 A&A 447 49 (WSRT deep HI, warp to 40 kpc)",
            "Blais-Ouellette+2004 A&A 420 147 (Fabry-Perot)",
        ],
    },
    {
        "id": "ngc891",
        "name": "NGC 891 (M = 7.5x10^10 M_sun, 8% gas)",
        "distance": 20,
        "galactic_radius": 21,
        "mass": 10.875,
        "accel": 1.0,
        "mass_model": {
            # Edge-on Sb spiral (Inc = 90 deg). Terminal velocity method.
            # L[3.6] = 138.3e9 L_sun, M_star = 6.92e10 (M/L=0.5).
            # B/T ~ 0.20 for Sb. Prominent dust lane.
            "bulge": {"M": 1.4e10, "a": 0.5},
            # Rdisk = 2.55 kpc (SPARC [3.6um]).
            "disk":  {"M": 5.5e10, "Rd": 2.55},
            # M_HI = 4.462e9, x1.33 = 5.93e9. RHI = 18.16 kpc.
            # Gas has central depression. Rd_gas ~ 7 kpc.
            "gas":   {"M": 5.9e9, "Rd": 7.0},
        },
        "observations": [
            # SPARC: Fraternali+2011 A&A 531 A64.
            # D = 9.91 Mpc, Inc = 90 deg (edge-on), Q = 1.
            # Vflat = 216.1 km/s, RHI = 18.16 kpc, R_last = 17.1 kpc.
            # Note: edge-on, rotation derived from terminal velocities.
            {"r": 0.92,  "v": 234, "err": 12},
            {"r": 2.32,  "v": 192, "err": 12},
            {"r": 3.24,  "v": 212, "err": 6},
            {"r": 4.17,  "v": 223, "err": 4},
            {"r": 5.10,  "v": 222, "err": 4},
            {"r": 6.03,  "v": 224, "err": 4},
            {"r": 6.96,  "v": 224, "err": 3},
            {"r": 7.89,  "v": 226, "err": 3},
            {"r": 8.80,  "v": 226, "err": 3},
            {"r": 9.73,  "v": 227, "err": 3},
            {"r": 10.64, "v": 227, "err": 3},
            {"r": 11.58, "v": 224, "err": 2},
            {"r": 12.52, "v": 220, "err": 2},
            {"r": 13.46, "v": 218, "err": 2},
            {"r": 14.40, "v": 217, "err": 3},
            {"r": 15.33, "v": 216, "err": 4},
            {"r": 16.27, "v": 210, "err": 4},
            {"r": 17.11, "v": 208, "err": 4},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016, Q=1, RHI=18.16 kpc)",
            "Fraternali+2011 A&A 531 A64 (deep HI observations)",
        ],
    },
    # --- Intermediate Sc/Scd spirals ---
    {
        "id": "ngc6946",
        "name": "NGC 6946 Fireworks (M = 4.1x10^10 M_sun, 19% gas)",
        "distance": 24,
        "galactic_radius": 24,
        "mass": 10.608,
        "accel": 1.0,
        "mass_model": {
            # Scd spiral with compact bulge. B/T ~ 0.05.
            # L[3.6] = 66.2e9 L_sun, M_star = 3.31e10 (M/L=0.5).
            "bulge": {"M": 1.5e9, "a": 0.2},
            # Rdisk = 2.44 kpc (SPARC [3.6um]).
            "disk":  {"M": 3.15e10, "Rd": 2.44},
            # M_HI = 5.670e9, x1.33 = 7.54e9. RHI = 21.25 kpc.
            # Gas Vcirc peaks ~47 at R=14.3 kpc, Rd_gas ~ 7 kpc.
            "gas":   {"M": 7.5e9, "Rd": 7.0},
        },
        "observations": [
            # SPARC: Boomsma+2008 A&A 490 555 (deep WSRT, 192 hr).
            # D = 5.52 Mpc, Inc = 38 deg, Q = 1.
            # Vflat = 158.9 km/s, RHI = 21.25 kpc, R_last = 20.4 kpc.
            # Note: low inclination (38 deg) introduces some uncertainty.
            {"r": 0.19,  "v": 153, "err": 8},
            {"r": 0.56,  "v": 130, "err": 9},
            {"r": 0.94,  "v": 128, "err": 3},
            {"r": 1.59,  "v": 129, "err": 4},
            {"r": 2.71,  "v": 139, "err": 5},
            {"r": 3.74,  "v": 154, "err": 2},
            {"r": 4.77,  "v": 173, "err": 2},
            {"r": 5.89,  "v": 180, "err": 3},
            {"r": 6.92,  "v": 181, "err": 4},
            {"r": 7.95,  "v": 179, "err": 1},
            {"r": 9.08,  "v": 176, "err": 2},
            {"r": 10.48, "v": 170, "err": 2},
            {"r": 11.88, "v": 175, "err": 2},
            {"r": 13.29, "v": 173, "err": 8},
            {"r": 14.69, "v": 167, "err": 8},
            {"r": 16.09, "v": 161, "err": 9},
            {"r": 17.50, "v": 157, "err": 10},
            {"r": 18.90, "v": 160, "err": 14},
            {"r": 20.02, "v": 157, "err": 19},
            {"r": 20.40, "v": 154, "err": 21},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016, Q=1, RHI=21.25 kpc)",
            "Boomsma+2008 A&A 490 555 (deep WSRT 192hr HI)",
        ],
    },
    {
        "id": "ngc2903",
        "name": "NGC 2903 (M = 4.4x10^10 M_sun, 8% gas)",
        "distance": 28,
        "galactic_radius": 16,
        "mass": 10.647,
        "accel": 1.0,
        "mass_model": {
            # SBbc barred spiral. B/T ~ 0.06 (Catalan-Torrecilla+2020).
            # L[3.6] = 81.9e9 L_sun, M_star = 4.09e10 (M/L=0.5).
            "bulge": {"M": 2.5e9, "a": 0.3},
            # Rdisk = 2.33 kpc (SPARC [3.6um]).
            "disk":  {"M": 3.85e10, "Rd": 2.33},
            # M_HI = 2.552e9, x1.33 = 3.39e9. RHI = 13.76 kpc.
            # Gas Vcirc peaks ~38 at R=12.5 kpc, Rd_gas ~ 5.5 kpc.
            "gas":   {"M": 3.4e9, "Rd": 5.5},
        },
        "observations": [
            # SPARC: Begeman+1991 + Begeman+1987.
            # D = 6.6 Mpc, Inc = 66 deg, Q = 1.
            # Vflat = 184.6 km/s, RHI = 13.76 kpc, R_last = 25.0 kpc.
            {"r": 0.32,  "v": 44,  "err": 8},
            {"r": 0.96,  "v": 128, "err": 8},
            {"r": 1.60,  "v": 157, "err": 8},
            {"r": 2.24,  "v": 195, "err": 6},
            {"r": 2.88,  "v": 214, "err": 4},
            {"r": 3.52,  "v": 215, "err": 4},
            {"r": 4.80,  "v": 214, "err": 2},
            {"r": 6.72,  "v": 205, "err": 2},
            {"r": 8.64,  "v": 201, "err": 1},
            {"r": 10.56, "v": 199, "err": 1},
            {"r": 12.48, "v": 196, "err": 2},
            {"r": 14.40, "v": 190, "err": 3},
            {"r": 16.32, "v": 187, "err": 5},
            {"r": 18.24, "v": 186, "err": 3},
            {"r": 20.16, "v": 185, "err": 3},
            {"r": 22.08, "v": 178, "err": 3},
            {"r": 24.00, "v": 182, "err": 4},
            {"r": 24.96, "v": 180, "err": 8},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016, Q=1, RHI=13.76 kpc)",
            "Begeman+1991 MNRAS 249 523 (WSRT 21cm)",
            "Begeman 1987 PhD thesis (Groningen)",
            "Catalan-Torrecilla+2020 MNRAS 493 4094 (VENGA, B/T)",
        ],
    },
    {
        "id": "ngc3521",
        "name": "NGC 3521 (M = 4.8x10^10 M_sun, 12% gas)",
        "distance": 20,
        "galactic_radius": 22,
        "mass": 10.680,
        "accel": 1.0,
        "mass_model": {
            # SABbc spiral. No bulge in SPARC decomposition.
            # L[3.6] = 84.8e9 L_sun, M_star = 4.24e10 (M/L=0.5).
            "bulge": {"M": 2.0e9, "a": 0.3},
            # Rdisk = 2.40 kpc (SPARC [3.6um]).
            "disk":  {"M": 4.04e10, "Rd": 2.40},
            # M_HI = 4.154e9, x1.33 = 5.52e9. RHI = 18.85 kpc.
            # Anomalous slow-rotating HI component ~20% of M_HI (Elson+2014).
            # Gas Vcirc peaks ~46 at R=15.5 kpc, Rd_gas ~ 7 kpc.
            "gas":   {"M": 5.5e9, "Rd": 7.0},
        },
        "observations": [
            # SPARC: Daigle+2006 (Fabry-Perot) + Sanders 1996 (HI).
            # D = 7.7 Mpc, Inc = 75 deg, Q = 1.
            # Vflat = 213.7 km/s, RHI = 18.85 kpc, R_last = 17.7 kpc.
            # Inner region has large scatter (strong bar/spiral arms).
            {"r": 0.51,  "v": 143, "err": 33},
            {"r": 0.86,  "v": 178, "err": 8},
            {"r": 1.20,  "v": 187, "err": 8},
            {"r": 1.71,  "v": 203, "err": 7},
            {"r": 2.23,  "v": 219, "err": 13},
            {"r": 2.75,  "v": 213, "err": 8},
            {"r": 3.43,  "v": 212, "err": 11},
            {"r": 4.12,  "v": 220, "err": 8},
            {"r": 4.81,  "v": 216, "err": 24},
            {"r": 5.49,  "v": 215, "err": 21},
            {"r": 6.35,  "v": 216, "err": 20},
            {"r": 8.82,  "v": 210, "err": 11},
            {"r": 11.07, "v": 210, "err": 11},
            {"r": 13.32, "v": 206, "err": 6},
            {"r": 15.49, "v": 206, "err": 6},
            {"r": 17.74, "v": 206, "err": 11},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016, Q=1, RHI=18.85 kpc)",
            "Daigle+2006 MNRAS 367 469 (Fabry-Perot Halpha)",
            "Sanders 1996 ApJ 437 117 (HI rotation curve)",
            "Elson+2014 MNRAS 437 3736 (anomalous slow-rotating HI)",
        ],
    },
    {
        "id": "ngc2998",
        "name": "NGC 2998 (M = 1.1x10^11 M_sun, 29% gas)",
        "distance": 48,
        "galactic_radius": 50,
        "mass": 11.029,
        "accel": 1.0,
        "mass_model": {
            # Sc spiral, no bulge. Distant (68.1 Mpc) but well-measured.
            # L[3.6] = 150.9e9 L_sun, M_star = 7.55e10 (M/L=0.5).
            "bulge": {"M": 0, "a": 0.1},
            # Rdisk = 6.20 kpc (SPARC [3.6um]).
            "disk":  {"M": 7.55e10, "Rd": 6.20},
            # M_HI = 23.451e9, x1.33 = 3.12e10. RHI = 43.58 kpc.
            # Gas Vcirc peaks ~66 at R=37 kpc, Rd_gas ~ 17 kpc.
            "gas":   {"M": 3.12e10, "Rd": 17.0},
        },
        "observations": [
            # SPARC: Sanders 1996 + Broeils 1992.
            # D = 68.1 Mpc, Inc = 58 deg, Q = 1.
            # Vflat = 209.9 km/s, RHI = 43.58 kpc, R_last = 42.3 kpc.
            {"r": 0.33,  "v": 90,  "err": 20},
            {"r": 0.99,  "v": 125, "err": 15},
            {"r": 1.98,  "v": 148, "err": 15},
            {"r": 2.64,  "v": 180, "err": 17},
            {"r": 3.63,  "v": 201, "err": 15},
            {"r": 7.59,  "v": 206, "err": 10},
            {"r": 12.50, "v": 214, "err": 5},
            {"r": 17.48, "v": 212, "err": 5},
            {"r": 22.46, "v": 213, "err": 2},
            {"r": 27.44, "v": 214, "err": 2},
            {"r": 32.32, "v": 213, "err": 2},
            {"r": 37.30, "v": 213, "err": 3},
            {"r": 42.28, "v": 203, "err": 3},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016, Q=1, RHI=43.58 kpc)",
            "Sanders 1996 ApJ 437 117 (HI rotation curve)",
            "Broeils 1992 PhD thesis (Groningen, WSRT HI)",
        ],
    },
    # --- Gas-rich late-type and LSB galaxies ---
    {
        "id": "ngc1003",
        "name": "NGC 1003 (M = 1.1x10^10 M_sun, 70% gas)",
        "distance": 34,
        "galactic_radius": 38,
        "mass": 10.049,
        "accel": 1.0,
        "mass_model": {
            # Scd spiral, very gas-rich, no bulge.
            # L[3.6] = 6.82e9 L_sun, M_star = 3.41e9 (M/L=0.5).
            "bulge": {"M": 0, "a": 0.1},
            # Rdisk = 1.61 kpc (SPARC [3.6um]).
            "disk":  {"M": 3.4e9, "Rd": 1.61},
            # M_HI = 5.880e9, x1.33 = 7.82e9. RHI = 33.33 kpc.
            # Gas Vcirc still rising at 30 kpc, Rd_gas ~ 14 kpc.
            "gas":   {"M": 7.8e9, "Rd": 14.0},
        },
        "observations": [
            # SPARC: Sanders 1996 + Broeils 1992.
            # D = 11.4 Mpc, Inc = 67 deg, Q = 1.
            # Vflat = 109.8 km/s, RHI = 33.33 kpc, R_last = 30.2 kpc.
            {"r": 1.25,  "v": 47,  "err": 8},
            {"r": 2.08,  "v": 60,  "err": 5},
            {"r": 2.90,  "v": 72,  "err": 4},
            {"r": 3.73,  "v": 78,  "err": 3},
            {"r": 4.56,  "v": 83,  "err": 4},
            {"r": 5.39,  "v": 90,  "err": 4},
            {"r": 6.22,  "v": 94,  "err": 2},
            {"r": 7.87,  "v": 103, "err": 3},
            {"r": 9.54,  "v": 98,  "err": 2},
            {"r": 11.21, "v": 95,  "err": 2},
            {"r": 13.72, "v": 97,  "err": 2},
            {"r": 15.36, "v": 100, "err": 2},
            {"r": 17.00, "v": 104, "err": 3},
            {"r": 19.52, "v": 110, "err": 4},
            {"r": 21.93, "v": 106, "err": 4},
            {"r": 24.44, "v": 110, "err": 3},
            {"r": 27.73, "v": 114, "err": 2},
            {"r": 30.24, "v": 115, "err": 2},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016, Q=1, RHI=33.33 kpc)",
            "Sanders 1996 ApJ 437 117 (HI rotation curve)",
            "Broeils 1992 PhD thesis (Groningen, WSRT HI)",
        ],
    },
    {
        "id": "ngc5585",
        "name": "NGC 5585 (M = 3.7x10^9 M_sun, 60% gas)",
        "distance": 14,
        "galactic_radius": 13,
        "mass": 9.569,
        "accel": 1.0,
        "mass_model": {
            # Sd spiral, gas-dominated, no bulge.
            # L[3.6] = 2.94e9 L_sun, M_star = 1.47e9 (M/L=0.5).
            "bulge": {"M": 0, "a": 0.1},
            # Rdisk = 1.53 kpc (SPARC [3.6um]).
            "disk":  {"M": 1.47e9, "Rd": 1.53},
            # M_HI = 1.683e9, x1.33 = 2.24e9. RHI = 10.92 kpc.
            # Gas Vcirc peaks ~33 at R=10 kpc, Rd_gas ~ 5 kpc.
            "gas":   {"M": 2.24e9, "Rd": 5.0},
        },
        "observations": [
            # SPARC: Blais-Ouellette+1999 + Sanders 1996 + Cote+1991.
            # D = 7.06 Mpc, Inc = 51 deg, Q = 1.
            # Vflat = 90.3 km/s, RHI = 10.92 kpc, R_last = 11.0 kpc.
            {"r": 0.09,  "v": 11,  "err": 2},
            {"r": 0.26,  "v": 27,  "err": 1},
            {"r": 0.60,  "v": 33,  "err": 1},
            {"r": 0.94,  "v": 36,  "err": 2},
            {"r": 1.46,  "v": 44,  "err": 2},
            {"r": 2.05,  "v": 53,  "err": 2},
            {"r": 2.74,  "v": 64,  "err": 1},
            {"r": 3.42,  "v": 73,  "err": 2},
            {"r": 4.79,  "v": 78,  "err": 4},
            {"r": 5.48,  "v": 83,  "err": 3},
            {"r": 6.85,  "v": 90,  "err": 2},
            {"r": 7.53,  "v": 91,  "err": 3},
            {"r": 8.90,  "v": 92,  "err": 1},
            {"r": 9.57,  "v": 91,  "err": 1},
            {"r": 10.31, "v": 90,  "err": 3},
            {"r": 10.96, "v": 89,  "err": 3},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016, Q=1, RHI=10.92 kpc)",
            "Blais-Ouellette+1999 AJ 118 2123 (Fabry-Perot Halpha)",
            "Sanders 1996 ApJ 437 117 (HI rotation curve)",
            "Cote+1991 AJ 102 904 (HI observations)",
        ],
    },
    {
        "id": "ugc128",
        "name": "UGC 128 (M = 1.6x10^10 M_sun, 62% gas)",
        "distance": 58,
        "galactic_radius": 36,
        "mass": 10.201,
        "accel": 1.0,
        "mass_model": {
            # Low surface brightness (LSB) galaxy. No bulge.
            # L[3.6] = 12.02e9 L_sun, M_star = 6.01e9 (M/L=0.5).
            # One of the classic LSB rotation curve benchmarks.
            "bulge": {"M": 0, "a": 0.1},
            # Rdisk = 5.95 kpc (SPARC [3.6um]). Very extended disk.
            "disk":  {"M": 6.0e9, "Rd": 5.95},
            # M_HI = 7.431e9, x1.33 = 9.88e9. RHI = 31.27 kpc.
            # Gas Vcirc peaks ~45 at R=31 kpc, Rd_gas ~ 15 kpc.
            "gas":   {"M": 9.9e9, "Rd": 15.0},
        },
        "observations": [
            # SPARC: Verheijen & de Blok 1999 + van der Hulst+1993.
            # D = 64.5 Mpc, Inc = 57 deg, Q = 1.
            # Vflat = 129.3 km/s, RHI = 31.27 kpc, R_last = 53.8 kpc.
            {"r": 1.25,  "v": 34,  "err": 19},
            {"r": 3.75,  "v": 78,  "err": 7},
            {"r": 6.25,  "v": 97,  "err": 2},
            {"r": 8.76,  "v": 108, "err": 1},
            {"r": 11.25, "v": 113, "err": 1},
            {"r": 13.74, "v": 114, "err": 1},
            {"r": 16.22, "v": 121, "err": 1},
            {"r": 18.71, "v": 126, "err": 1},
            {"r": 21.30, "v": 128, "err": 1},
            {"r": 26.28, "v": 129, "err": 1},
            {"r": 31.25, "v": 131, "err": 1},
            {"r": 36.23, "v": 127, "err": 1},
            {"r": 41.31, "v": 130, "err": 2},
            {"r": 46.28, "v": 134, "err": 2},
            {"r": 51.26, "v": 132, "err": 5},
            {"r": 53.75, "v": 125, "err": 6},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016, Q=1, RHI=31.27 kpc)",
            "Verheijen & de Blok 1999 Ap&SS 269 673 (Ursa Major cluster)",
            "van der Hulst+1993 AJ 106 548 (WSRT HI)",
        ],
    },
    {
        "id": "ddo170",
        "name": "DDO 170 (M = 1.3x10^9 M_sun, 78% gas)",
        "distance": 14,
        "galactic_radius": 11,
        "mass": 9.097,
        "accel": 1.0,
        "mass_model": {
            # Irregular dwarf. No bulge. Gas-dominated.
            # L[3.6] = 0.543e9 L_sun, M_star = 2.72e8 (M/L=0.5).
            "bulge": {"M": 0, "a": 0.1},
            # Rdisk = 1.95 kpc (SPARC [3.6um]).
            "disk":  {"M": 2.7e8, "Rd": 1.95},
            # M_HI = 0.735e9, x1.33 = 9.78e8. RHI = 9.14 kpc.
            # Gas Vcirc peaks ~28 at R=9.3 kpc, Rd_gas ~ 4 kpc.
            "gas":   {"M": 9.8e8, "Rd": 4.0},
        },
        "observations": [
            # SPARC: Begeman+1991 + Lake+1990.
            # D = 15.4 Mpc, Inc = 66 deg, Q = 2.
            # Vflat = 60.0 km/s, RHI = 9.14 kpc, R_last = 12.3 kpc.
            {"r": 1.87,  "v": 28,  "err": 3},
            {"r": 3.36,  "v": 42,  "err": 2},
            {"r": 4.86,  "v": 53,  "err": 1},
            {"r": 6.35,  "v": 56,  "err": 1},
            {"r": 7.85,  "v": 59,  "err": 1},
            {"r": 9.34,  "v": 59,  "err": 1},
            {"r": 10.83, "v": 60,  "err": 1},
            {"r": 12.33, "v": 62,  "err": 1},
        ],
        "references": [
            "SPARC VizieR J/AJ/152/157 (Lelli+2016, Q=2, RHI=9.14 kpc)",
            "Begeman+1991 MNRAS 249 523 (WSRT 21cm)",
            "Lake+1990 AJ 99 547 (HI observations)",
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
        "name": "Milky Way (v = 230 km/s at 8 kpc, 20% gas)",
        "distance": 8,
        "velocity": 230,
        "accel": 1.0,
        "galactic_radius": 60,
        "mass_model": {
            "bulge": {"M": 1.5e10, "a": 0.6},
            "disk":  {"M": 4.57e10, "Rd": 2.2},
            "gas":   {"M": 1.5e10, "Rd": 7.0},
        },
    },
    {
        "id": "m31_inference",
        "name": "Andromeda M31 (v = 260 km/s at 15 kpc, 8% gas)",
        "distance": 15,
        "velocity": 260,
        "accel": 1.0,
        "galactic_radius": 40,
        "mass_model": {
            "bulge": {"M": 3.0e10, "a": 1.0},
            "disk":  {"M": 7.76e10, "Rd": 5.5},
            "gas":   {"M": 1.0e10, "Rd": 12.0},
        },
    },
    {
        "id": "ugc2885_inference",
        "name": "UGC 2885 Rubin's Galaxy (v = 298 km/s at 50 kpc, 40% gas)",
        "distance": 50,
        "velocity": 298,
        "accel": 1.0,
        "galactic_radius": 75,
        "mass_model": {
            "bulge": {"M": 5.0e10, "a": 1.5},
            "disk":  {"M": 1.0e11, "Rd": 6.0},
            "gas":   {"M": 1.0e11, "Rd": 22.0},
        },
    },
    {
        "id": "ngc3198_inference",
        "name": "NGC 3198 (v = 150 km/s at 20 kpc, 43% gas)",
        "distance": 20,
        "velocity": 150,
        "accel": 1.0,
        "galactic_radius": 48,
        "mass_model": {
            "bulge": {"M": 1.0e9,   "a": 0.3},
            "disk":  {"M": 18.14e9, "Rd": 3.0},
            "gas":   {"M": 14.46e9, "Rd": 6.0},
        },
    },
    {
        "id": "ngc6503_inference",
        "name": "NGC 6503 (v = 121 km/s at 10 kpc, 27% gas)",
        "distance": 10,
        "velocity": 121,
        "accel": 1.0,
        "galactic_radius": 17,
        "mass_model": {
            "bulge": {"M": 0.6e9,  "a": 0.3},
            "disk":  {"M": 5.82e9, "Rd": 1.7},
            "gas":   {"M": 2.32e9, "Rd": 4.0},
        },
    },
    {
        "id": "ngc3109_inference",
        "name": "NGC 3109 (v = 63 km/s at 5 kpc, 91% gas)",
        "distance": 5,
        "velocity": 63,
        "accel": 1.0,
        "galactic_radius": 7,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 1.0e8, "Rd": 1.5},
            "gas":   {"M": 1.0e9, "Rd": 3.5},
        },
    },
    {
        "id": "m33_inference",
        "name": "M33 Triangulum (v = 112 km/s at 8 kpc, 45% gas)",
        "distance": 8,
        "velocity": 112,
        "accel": 1.0,
        "galactic_radius": 17,
        "mass_model": {
            "bulge": {"M": 0.4e9, "a": 0.18},
            "disk":  {"M": 3.5e9, "Rd": 1.6},
            "gas":   {"M": 3.2e9, "Rd": 4.0},
        },
    },
    {
        "id": "ddo154_inference",
        "name": "DDO 154 (v = 44 km/s at 5 kpc, 93% gas)",
        "distance": 5,
        "velocity": 44,
        "accel": 1.0,
        "galactic_radius": 8,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 3.0e7, "Rd": 0.7},
            "gas":   {"M": 4.0e8, "Rd": 2.5},
        },
    },
    {
        "id": "ic2574_inference",
        "name": "IC 2574 (v = 66 km/s at 9 kpc, 92% gas)",
        "distance": 9,
        "velocity": 66,
        "accel": 1.0,
        "galactic_radius": 11,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 9.31e7, "Rd": 1.70},
            "gas":   {"M": 1.032e9, "Rd": 4.0},
        },
    },
    # === SPARC inference galaxies (matching prediction entries) ===
    {
        "id": "ngc2841_inference",
        "name": "NGC 2841 (v = 289 km/s at 30 kpc, 12% gas)",
        "distance": 30,
        "velocity": 289,
        "accel": 1.0,
        "galactic_radius": 50,
        "mass_model": {
            "bulge": {"M": 2.0e10, "a": 0.5},
            "disk":  {"M": 7.5e10, "Rd": 3.64},
            "gas":   {"M": 1.3e10, "Rd": 20.0},
        },
    },
    {
        "id": "ngc7331_inference",
        "name": "NGC 7331 (v = 237 km/s at 20 kpc, 11% gas)",
        "distance": 20,
        "velocity": 237,
        "accel": 1.0,
        "galactic_radius": 30,
        "mass_model": {
            "bulge": {"M": 2.0e10, "a": 0.8},
            "disk":  {"M": 1.05e11, "Rd": 5.02},
            "gas":   {"M": 1.47e10, "Rd": 11.0},
        },
    },
    {
        "id": "ngc5055_inference",
        "name": "NGC 5055 Sunflower (v = 180 km/s at 35 kpc, 17% gas)",
        "distance": 35,
        "velocity": 180,
        "accel": 1.0,
        "galactic_radius": 40,
        "mass_model": {
            "bulge": {"M": 1.0e9, "a": 0.3},
            "disk":  {"M": 7.55e10, "Rd": 3.20},
            "gas":   {"M": 1.56e10, "Rd": 10.0},
        },
    },
    {
        "id": "ngc891_inference",
        "name": "NGC 891 (v = 218 km/s at 13 kpc, 8% gas)",
        "distance": 13,
        "velocity": 218,
        "accel": 1.0,
        "galactic_radius": 21,
        "mass_model": {
            "bulge": {"M": 1.4e10, "a": 0.5},
            "disk":  {"M": 5.5e10, "Rd": 2.55},
            "gas":   {"M": 5.9e9, "Rd": 7.0},
        },
    },
    {
        "id": "ngc6946_inference",
        "name": "NGC 6946 Fireworks (v = 173 km/s at 12 kpc, 19% gas)",
        "distance": 12,
        "velocity": 173,
        "accel": 1.0,
        "galactic_radius": 24,
        "mass_model": {
            "bulge": {"M": 1.5e9, "a": 0.2},
            "disk":  {"M": 3.15e10, "Rd": 2.44},
            "gas":   {"M": 7.5e9, "Rd": 7.0},
        },
    },
    {
        "id": "ngc2903_inference",
        "name": "NGC 2903 (v = 190 km/s at 14 kpc, 8% gas)",
        "distance": 14,
        "velocity": 190,
        "accel": 1.0,
        "galactic_radius": 16,
        "mass_model": {
            "bulge": {"M": 2.5e9, "a": 0.3},
            "disk":  {"M": 3.85e10, "Rd": 2.33},
            "gas":   {"M": 3.4e9, "Rd": 5.5},
        },
    },
    {
        "id": "ngc3521_inference",
        "name": "NGC 3521 (v = 210 km/s at 10 kpc, 12% gas)",
        "distance": 10,
        "velocity": 210,
        "accel": 1.0,
        "galactic_radius": 22,
        "mass_model": {
            "bulge": {"M": 2.0e9, "a": 0.3},
            "disk":  {"M": 4.04e10, "Rd": 2.40},
            "gas":   {"M": 5.5e9, "Rd": 7.0},
        },
    },
    {
        "id": "ngc2998_inference",
        "name": "NGC 2998 (v = 213 km/s at 25 kpc, 29% gas)",
        "distance": 25,
        "velocity": 213,
        "accel": 1.0,
        "galactic_radius": 50,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 7.55e10, "Rd": 6.20},
            "gas":   {"M": 3.12e10, "Rd": 17.0},
        },
    },
    {
        "id": "ngc1003_inference",
        "name": "NGC 1003 (v = 110 km/s at 25 kpc, 70% gas)",
        "distance": 25,
        "velocity": 110,
        "accel": 1.0,
        "galactic_radius": 38,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 3.4e9, "Rd": 1.61},
            "gas":   {"M": 7.8e9, "Rd": 14.0},
        },
    },
    {
        "id": "ngc5585_inference",
        "name": "NGC 5585 (v = 92 km/s at 8 kpc, 60% gas)",
        "distance": 8,
        "velocity": 92,
        "accel": 1.0,
        "galactic_radius": 13,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 1.47e9, "Rd": 1.53},
            "gas":   {"M": 2.24e9, "Rd": 5.0},
        },
    },
    {
        "id": "ugc128_inference",
        "name": "UGC 128 (v = 131 km/s at 30 kpc, 62% gas)",
        "distance": 30,
        "velocity": 131,
        "accel": 1.0,
        "galactic_radius": 36,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 6.0e9, "Rd": 5.95},
            "gas":   {"M": 9.9e9, "Rd": 15.0},
        },
    },
    {
        "id": "ddo170_inference",
        "name": "DDO 170 (v = 59 km/s at 8 kpc, 78% gas)",
        "distance": 8,
        "velocity": 59,
        "accel": 1.0,
        "galactic_radius": 11,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 2.7e8, "Rd": 1.95},
            "gas":   {"M": 9.8e8, "Rd": 4.0},
        },
    },
]


def _sparc_dir():
    """Path to sparc/ folder at repo root."""
    return os.path.join(os.path.dirname(__file__), "..", "sparc")


def _load_sparc_galaxy_by_id(galaxy_id):
    """Load a single galaxy from sparc/<galaxy_id>.json if present. Returns dict or None."""
    if not galaxy_id or "/" in galaxy_id or "\\" in galaxy_id:
        return None
    path = os.path.join(_sparc_dir(), galaxy_id + ".json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            g = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(g, dict):
        return None
    required = ("id", "name", "distance", "galactic_radius", "mass", "accel", "mass_model", "observations", "references")
    if not all(g.get(k) is not None for k in required):
        return None
    return g


def _load_sparc_galaxies():
    """Load galaxy JSON files from sparc/ folder. Returns sorted list by id."""
    sparc_dir = _sparc_dir()
    if not os.path.isdir(sparc_dir):
        return []
    result = []
    for fname in os.listdir(sparc_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(sparc_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                g = json.load(f)
            if not isinstance(g, dict):
                continue
            required = ("id", "name", "distance", "galactic_radius", "mass", "accel", "mass_model", "observations", "references")
            if not all(g.get(k) is not None for k in required):
                continue
            result.append(g)
        except (json.JSONDecodeError, OSError):
            continue
    result.sort(key=lambda x: (x.get("id") or ""))
    return result


def get_prediction_galaxies():
    """Return all prediction-mode galaxies (with mass models + observations)."""
    base = PREDICTION_GALAXIES + SIMPLE_PREDICTION_GALAXIES
    sparc = _load_sparc_galaxies()
    return base + sparc


def get_inference_galaxies():
    """Return all inference-mode galaxies."""
    return INFERENCE_GALAXIES


def get_galaxy_by_id(galaxy_id):
    """Look up a galaxy by its unique id: built-in catalog first, then sparc/<id>.json."""
    for g in get_prediction_galaxies() + INFERENCE_GALAXIES:
        if g["id"] == galaxy_id:
            return g
    return _load_sparc_galaxy_by_id(galaxy_id)


def get_all_galaxies():
    """Return all galaxies grouped by mode."""
    return {
        "prediction": get_prediction_galaxies(),
        "inference": get_inference_galaxies(),
    }
