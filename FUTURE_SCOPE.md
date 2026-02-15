# GRAVIS Future Scope

Planned modules and extensions for the GRAVIS interactive platform. Each module demonstrates a different observational test of Dual Tetrad Gravity (DTG) and the field origin framework, all derived from the same tetrahedral topology with zero free parameters.

## Astrophysical Modules

### Radial Acceleration Relation (RAR)
Plot observed gravitational acceleration vs. baryonic (Newtonian) acceleration for all galaxies. DTG predicts the exact functional form:

$$g_{\text{obs}} = \frac{g_N}{1 - e^{-\sqrt{g_N / a_0}}}$$

with no free parameters and no intrinsic scatter from the theory line. SPARC provides data for 175 galaxies. An interactive version would let users click individual galaxies and see where their data points fall relative to the DTG prediction.

**Data source:** SPARC (Lelli, McGaugh, Schombert 2017), 2,693 individual data points across 153 galaxies.

### Baryonic Tully-Fisher Relation (BTFR)
Dedicated page plotting total baryonic mass vs. flat rotation velocity for all galaxies in the catalog. DTG predicts a strict power law:

$$M_{\text{baryon}} = \frac{v_f^4}{G \, a_0}$$

with slope exactly 4 and normalization fixed by fundamental constants. No scatter from the theory. Deviations in the data reflect measurement uncertainty, not model freedom.

**Status:** Data already exists in the app. Requires a new template and chart.

### Tully-Fisher Evolution with Redshift
Show how the Tully-Fisher relation evolves across cosmic time. The DTG field propagation framework predicts:

$$\frac{v(z)}{v(0)} = \left[\frac{H(z)}{H_0}\right]^{0.17}$$

where LCDM predicts no evolution (flat at 1.0). High-redshift observations (SINS, KMOS3D, KROSS) show a +20 to 30% velocity increase at z=2, consistent with the DTG prediction. Interactive controls for redshift range, cosmological parameters, and framework selection.

**Data sources:** Cresci+2009 (SINS), Wisnioski+2015 (KMOS3D), Harrison+2017 (KROSS).

### Wide Binary Stars
Binary star systems with separations exceeding 5,000 AU probe the low-acceleration regime where DTG diverges from Newtonian gravity. Gaia DR3 provides thousands of wide binaries with precise proper motions. DTG predicts a specific velocity excess as a function of separation that has no counterpart in Newtonian gravity (even with dark matter, which has negligible density at these scales).

**Data source:** Gaia DR3 wide binary catalogs (El-Badry+2021, Chae 2023).

## Cosmological Modules

### H(z) and the Distance Ladder
Compute DTG-calibrated luminosity distances and compare to standard LCDM calibration across redshift. If the DTG Baryonic Tully-Fisher normalization differs from the standard pipeline, this propagates as a systematic shift in H0. Interactive visualization showing the distance ratio (DTG / standard) as a function of redshift, with direct readout of the implied H0 correction.

**Connection to the Hubble tension:** A 3 to 4% systematic in Tully-Fisher distances corresponds to a ~3 to 4 km/s/Mpc shift in H0, which is the size of the current tension between local (SH0ES) and early-universe (Planck) measurements.

### CMB Acoustic Peak Ratios
Simplified treatment of how the CMB acoustic peak structure is constrained differently without cold dark matter. The baryon-to-photon ratio and expansion rate at recombination determine peak spacing and relative heights. Even a first-order interactive model showing peak positions under DTG vs. LCDM would highlight where the frameworks make distinct, testable predictions.

## Atomic and Nuclear Modules

### Valence Predictions
Predict electron shell structure and valence from the k=4 simplex topology. User selects an element (or enters atomic number Z), and the page displays the predicted configuration alongside the observed one. Derived from the same tetrahedral geometry that produces a0 at the galactic scale.

### Nuclear Decay Rates
Interactive nuclear chart where users click an isotope and see the DTG-predicted half-life or branching ratio compared to the measured value. The tetrahedral topology constrains the tunneling geometry for alpha decay, connecting nuclear lifetimes to the same structural constants that govern galaxy dynamics.

### The a0 Derivation
Interactive page showing how the characteristic acceleration scale connects atomic constants to galactic dynamics:

$$a_0 = k^2 \, \frac{G \, m_e}{r_e^2}$$

Slider for k demonstrates that only k=4 (the simplex number for d=3 spatial dimensions) reproduces the empirical value. Displays the chain of reasoning from tetrahedral topology to the electron mass to galaxy rotation curves.

## Solar System Constraints

### Newtonian Recovery
Address the primary objection to any modified gravity theory: solar system precision tests. Interactive page showing the DTG correction factor g_DTG / g_N as a function of distance from the Sun. At solar system accelerations (g >> a0), the correction is unmeasurably small (parts per billion inside Neptune's orbit). Users can zoom in to see exactly where the deviation becomes detectable and compare to current measurement precision from planetary ephemerides and Cassini tracking.

## Implementation Priority

| Priority | Module | Effort | Impact |
|----------|--------|--------|--------|
| 1 | Radial Acceleration Relation | Medium | Highest: single most cited plot in the field |
| 2 | Tully-Fisher evolution | Low | High: data and plot already exist |
| 3 | Baryonic Tully-Fisher | Low | High: data already in the app |
| 4 | Solar system constraints | Low | High: preempts the top referee objection |
| 5 | a0 derivation interactive | Low | Medium: simple but compelling |
| 6 | Wide binary stars | Medium | High: timely, Gaia data is public |
| 7 | H(z) distance ladder | Medium | High: directly addresses Hubble tension |
| 8 | Valence predictions | Medium | Medium: cross-scale demonstration |
| 9 | Nuclear decay rates | High | Medium: unique to this theory |
| 10 | CMB acoustic peaks | High | High: requires careful cosmological treatment |
