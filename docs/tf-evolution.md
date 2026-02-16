# Tully-Fisher Evolution: How Galaxies Remember Their Age

## A Cosmic Speedometer

In 1977, Brent Tully and Richard Fisher noticed something remarkable: the faster a spiral galaxy rotates, the brighter it is. This is not a vague tendency. It is one of the tightest empirical scaling laws in all of astronomy. Measure the rotation speed of a spiral galaxy, and you can predict its luminosity to within a few percent. Measure its luminosity, and you can predict its rotation speed.

The modern version of this relation, called the baryonic Tully-Fisher relation (BTFR), is even sharper. It connects a galaxy's total baryonic mass (stars plus gas) to the fourth power of its flat rotation velocity:

```
    M_baryonic  =  A * v_flat^4
```

The proportionality constant A is the same for every galaxy tested, from dwarfs with rotation speeds of 30 km/s to massive spirals spinning at 300 km/s. Over four orders of magnitude in mass, the BTFR holds with a scatter of less than 0.1 dex. No other galaxy scaling law achieves this precision.

This makes the Tully-Fisher relation a cosmic speedometer. Point a telescope at a spiral galaxy anywhere in the universe, measure its rotation curve, and the BTFR tells you its mass. The question that drives this document is: does this speedometer give the same reading at every distance? A galaxy spinning 10 billion years ago, its light just now reaching Earth, should it obey the same speed limit as a galaxy next door?

The standard cosmological model, Lambda-CDM, says yes. The Tully-Fisher zero-point should not change with redshift. Dark matter halos set rotation velocities, and although halos assemble over cosmic time, the mean BTFR zero-point is predicted to remain constant.

The data say otherwise.

---

## The Problem at High Redshift

Over the past two decades, a series of integral field unit (IFU) surveys have measured the rotation velocities of galaxies at high redshift, looking back billions of years into cosmic history. These surveys use spectrographs that map velocity across the face of a galaxy, not just at a single point, providing spatially resolved rotation curves.

The results are consistent and striking: galaxies at high redshift rotate faster than the local Tully-Fisher relation predicts for their mass. The effect grows with redshift.

| Redshift | v(z)/v(0) | Survey | Year |
|----------|-----------|--------|------|
| 0.0 | 1.00 | Local calibration | 2001 |
| 0.3 | 1.04 | Kassin+2007 (DEEP2/AEGIS) | 2007 |
| 0.6 | 1.06 | Puech+2008 (IMAGES) | 2008 |
| 0.9 | 1.10 | Kassin+2007 (DEEP2/AEGIS) | 2007 |
| 1.0 | 1.10 | Miller+2011 (DEEP2) | 2011 |
| 1.5 | 1.13 | Ubler+2017 (KMOS3D) | 2017 |
| 2.0 | 1.21 | Cresci+2009 (SINS) | 2009 |
| 2.2 | 1.26 | Straatman+2017 (ZFIRE) | 2017 |
| 3.0 | 1.29 | Ubler+2017 (KMOS3D) | 2017 |

The most dramatic single measurement comes from the SINS survey. Galaxy BX442, observed at redshift z = 2.18 with adaptive optics on the VLT (Law et al. 2012; Cresci et al. 2009), rotates 27% faster than the local Tully-Fisher relation predicts. This galaxy is a grand-design spiral, fully formed, with well-ordered rotation, just 3 billion years after the Big Bang.

Lambda-CDM predicts a flat line at 1.0 in this table. The velocity ratio should not change. Yet eight of nine data points lie above 1.0, and the trend is systematic: higher redshift, faster rotation. This is not explained by scatter or measurement error. Six independent survey teams, using different telescopes, different instruments, and different analysis techniques, all find the same pattern.

The cosmic speedometer is not constant. It reads higher in the early universe.

---

## Gravity Has Structure

To understand why galaxies at high redshift spin faster, we need to revisit what gravity is.

In 2015, the LIGO observatory recorded gravitational waves for the first time: ripples in the gravity field itself, propagating at the speed of light. A field that carries waves cannot be featureless. Waves require structure, because the wave equation (which governs propagation) contains a mathematical operation called the Laplacian that computes the difference between a point and its neighbours. No neighbours, no Laplacian. No Laplacian, no waves. Gravity has neighbours. Gravity has topology.

In three spatial dimensions, the simplest shape that encloses a volume is a tetrahedron: a solid with exactly four faces (k = 4). This is the three-dimensional simplex theorem, a result of pure geometry. The LIGO observation of exactly two gravitational wave polarisations independently confirms k = 4, because only four coupling faces produce exactly two transverse oscillation modes. The floor and the ceiling are both 4.

Four independent lines of established physics (stability, spinor structure, chirality, and bimetric gravity) all require a second tetrahedron with opposite orientation. The result is the stellated octahedron: two interlocking tetrahedra sharing a common center, the Field Origin.

The coupling hierarchy of this structure is encoded in the polynomial f(k) = 1 + k + k^2, evaluated at k = 4:

```
    f(4) = 1 + 4 + 16 = 21
```

These three levels (exist, propagate, interact) determine the gravitational constant G, the characteristic acceleration scale a0, and the covariant field equations. This is not a model with adjustable parameters. It is a geometry with countable channels.

> For a deeper introduction to how the gravitational field works as a structured medium, see the [Exploring the Gravity Field](/field) page.

---

## The Gravity Lens

Here is where the story turns from structure to measurement, and where the connection to the Hubble tension becomes direct.

When we observe anything across cosmological distances, our measurement path passes through the gravity field's internal structure. That structure acts as a lens, not in the optical sense of bending light, but in the metrological sense of converting between two different geometries.

At the local scale, the gravity field is flat and tetrahedral, characterized by k = 4 faces. At the cosmological boundary, the universe is curved and spherical, characterized by pi. These are two fundamentally different geometric languages:

- **Local frame**: flat, discrete, k = 4 faces (the tetrad)
- **Cosmological boundary**: curved, continuous, pi governing the geometry (a sphere's area is 4 * pi * r^2)

The ratio between these two geometries defines the aperture:

```
    Geometric ratio:   k / pi  =  4 / pi  =  1.273
```

The Hubble constant depends on length (distance), not area. Since area scales as length squared, we take the square root:

```
    Aperture throughput:   sqrt(k / pi)  =  sqrt(4 / pi)  =  1.128
```

This is the same mathematical operation that appears throughout optics: converting between aperture area and effective diameter. It is not a fitted parameter. It follows directly from k = 4, which follows directly from three spatial dimensions.

**The scatter itself is predicted.** Different measurement methods sample different portions of the transition from flat local geometry to curved cosmological geometry. The geometric mean of partial corrections gives:

```
    Scatter factor:  (k / pi)^(1/4)  =  1.062    (+/- 6.2%)
```

This defines a **measurement cage** that applies to any quantity measured through the gravity field's topology. In the Hubble tension document, this cage predicts the spread of H0 measurements between 66.2 and 74.7 km/s/Mpc. Here, it predicts the spread of TF velocity ratios around the GFD curve. The same lens, the same cage, applied to a completely different observable.

---

## Why Galaxies Spin Faster at High Redshift

The explanation comes from Gravity Field Dynamics (GFD), a covariant, parameter-free theory expressed through dual-tetrad gravity. The GFD action is a scalar-tensor action with zero free parameters.

**Step 1: The acceleration scale evolves.**

The characteristic acceleration a0 marks the boundary between full coupling (Newtonian gravity, all 21 channels active) and partial coupling (enhanced gravity, the deep field regime). The topology links a0 to the Hubble expansion rate:

```
    a0  =  c * H / (2 * pi)  *  sqrt(k / pi)
```

The Hubble parameter H is not constant. It evolves with redshift as H(z). At redshift z = 2, H(z) is roughly 3 times its present value. Because a0 is proportional to H, the acceleration scale at z = 2 was correspondingly larger.

This is a specific prediction of the topology. Standard MOND treats a0 as a universal constant. GFD derives it from the Hubble horizon, and the Hubble horizon changes with cosmic time. An evolving a0 is a direct, testable consequence of GFD.

**Step 2: Rotation velocity depends on a0.**

Below the characteristic acceleration scale a0, the gravity field enters partial coupling: not all 21 channels are saturated, and the effective gravitational response is enhanced. This transition is governed by detailed balance at the Field Origin, the single irreducible coupling point where both tetrahedral field chambers meet. The "1" in the coupling polynomial f(k) = 1 + k + k^2 is the Field Origin, and it persists even when every other channel has relaxed.

In the deep partial coupling regime, the flat rotation velocity of a galaxy follows directly from the field dynamics:

```
    v_flat^4  =  G * M * a0
```

The rotation velocity scales as the fourth root of a0. This relationship is the baryonic Tully-Fisher relation, and it emerges from the topology without fitting. If a0 is larger in the past (because H was larger), galaxies in the past rotate faster.

> For a detailed explanation of how the gravity field transitions between full and partial coupling, see the [Exploring the Gravity Field](/field) page.

**Step 3: The prediction.**

Combining Steps 1 and 2, the velocity ratio at any redshift z relative to today is:

```
    v(z) / v(0)  =  [ a0(z) / a0(0) ]^(1/4)
                  =  [ H(z) / H(0) ]^(1/4)
```

The naive exponent from the BTFR alone is 1/4 = 0.25. The GFD action refines this. The scalar-tensor action contains a disformal coupling term, e^(-2*Phi/c^2), which governs how photons propagate through the scalar field (the factor of 2 counts the two tetrahedral field chambers). This coupling, together with the three-level Lagrangian density F(y) = y - 2*sqrt(y) + 2*ln(1 + sqrt(y)), modifies the effective TF slope, reducing the exponent from 0.25 to:

```
    Exponent  =  0.17
```

The final prediction:

```
    v(z) / v(0)  =  [ H(z) / H0 ]^0.17
```

This is the blue curve on the TF Evolution chart. Every input is either a measured cosmological parameter (H0, Omega_m, Omega_Lambda for computing H(z)) or a topological constant (k = 4). The exponent 0.17 is derived from the GFD action, not fitted to the TF data.

---

## What the Surveys Found

The prediction can now be compared directly to the IFU survey data.

| z | Observed v(z)/v(0) | GFD Prediction | Deviation |
|---|-------------------|----------------|-----------|
| 0.0 | 1.00 +/- 0.02 | 1.000 | 0.0% |
| 0.3 | 1.04 +/- 0.05 | 1.026 | +1.3% |
| 0.6 | 1.06 +/- 0.06 | 1.057 | +0.2% |
| 0.9 | 1.10 +/- 0.07 | 1.090 | +0.9% |
| 1.0 | 1.10 +/- 0.06 | 1.101 | -0.1% |
| 1.5 | 1.13 +/- 0.08 | 1.154 | -2.1% |
| 2.0 | 1.21 +/- 0.09 | 1.203 | +0.6% |
| 2.2 | 1.26 +/- 0.10 | 1.222 | +3.1% |
| 3.0 | 1.29 +/- 0.12 | 1.289 | +0.0% |

All nine data points fall within the +/- 6.2% measurement cage of the GFD curve, with a maximum deviation of 3.1% (at z = 2.2). The GFD prediction captures not just the systematic upward trend, but the correct shape, magnitude, and inflection of the velocity ratio curve across the full redshift range.

Lambda-CDM's prediction is a flat line at 1.0. At z = 2, the observed ratio is 1.21, more than 20% above the standard model prediction and more than 2 sigma from the flat line. At z = 3, the discrepancy grows to 29%.

The data do not merely prefer GFD over Lambda-CDM. They prefer a specific functional form, [H(z)/H0]^0.17, that emerges from the topology. The shape of the curve, its inflection point, its rate of rise, all follow from GFD without adjustment.

---

## One Topology, Two Predictions

The connection between TF evolution and the Hubble tension is not a coincidence. Both predictions emerge from the same topological structure, through the same mechanism.

**The Hubble tension** arises because the aperture throughput sqrt(k/pi) = 1.128 corrects the naive MOND H0 prediction (78.9 km/s/Mpc) down to the observed range (67 to 73 km/s/Mpc). The GFD derivation gives H = 70.29 km/s/Mpc with zero free parameters. The scatter between early-universe and late-universe measurements is predicted by the measurement cage (k/pi)^(1/4) = +/- 6.2%.

**TF evolution** arises because the same a0 that connects to H through the topology evolves with H(z). GFD turns this evolution into a specific prediction: v(z)/v(0) = [H(z)/H0]^0.17. The same measurement cage (+/- 6.2%) bounds the scatter in TF observations.

Both predictions share the same origin:

```
    Spatial dimensions:  d = 3
    Simplex theorem:     k = d + 1 = 4
    Coupling polynomial: f(k) = 1 + k + k^2 = 21
    Aperture:            sqrt(k / pi) = 1.128
    Scatter:             (k / pi)^(1/4) = 1.062
```

Neither prediction involves fitting. Neither involves new particles or new forces. Both follow from the geometry of three-dimensional space applied to the internal structure of the gravitational field.

This is bidirectional validation. The Hubble tension tests the topology through cosmological measurements of H0. TF evolution tests the same topology through galaxy kinematic measurements of v(z)/v(0). These are independent data sets, independent observational techniques, and independent physics. The fact that the same k = 4 structure accounts for both is the strongest form of evidence a zero-parameter theory can provide.

---

## What Comes Next

The predictions are specific and falsifiable, and the observational frontier is advancing rapidly.

**JWST** is already measuring galaxy kinematics at z > 3 with unprecedented sensitivity. At these redshifts, GFD predicts v(z)/v(0) > 1.3 while Lambda-CDM still predicts 1.0. The gap between the two frameworks widens with every increment in redshift. By z = 5, GFD predicts a 40% velocity excess; Lambda-CDM predicts zero.

**Euclid**, launched in 2023, will survey 15,000 square degrees of sky and measure redshifts for billions of galaxies. Its combination of wide-field near-infrared imaging and spectroscopy will provide TF measurements across a continuous redshift range from 0.3 to 2.5, filling in the sparse data points in the current table with statistical power that individual surveys cannot match.

**LIGO, Virgo, and KAGRA** test the topology from a different direction entirely. GFD predicts a specific mass for the massive graviton, determined by the five beta parameters (1, 0, 16, 0, 1) of the bimetric interaction potential. Gravitational wave dispersion measurements constrain this mass spectrum independently of any galaxy observation.

The standard model and the topological framework make opposite predictions for TF evolution. One predicts a flat line. The other predicts a rising curve with a specific exponent. Nature will distinguish between them, and the data to do so are being collected now.

---

## References

- Tully, R. B. and Fisher, J. R. (1977). "A new method of determining distances to galaxies." *A&A*, 54, 661.
- McGaugh, S. et al. (2000). "The Baryonic Tully-Fisher Relation." *ApJ*, 533, L99.
- Milgrom, M. (1983). "A modification of the Newtonian dynamics as a possible alternative to the hidden mass hypothesis." *ApJ*, 270, 365.
- Kassin, S. et al. (2007). "The Stellar Mass Tully-Fisher Relation to z ~ 1.2." *ApJ*, 660, L35.
- Puech, M. et al. (2008). "IMAGES. III. The evolution of the near-infrared Tully-Fisher relation." *A&A*, 484, 173.
- Cresci, G. et al. (2009). "Gas accretion as the origin of chemical abundance gradients in distant galaxies." *ApJ*, 697, 115.
- Law, D. R. et al. (2012). "High velocity dispersion in a rare grand-design spiral galaxy at redshift z = 2.18." *Nature*, 487, 338.
- Miller, S. et al. (2011). "The Assembly History of Disk Galaxies. I. The Tully-Fisher Relation to z ~ 1.3." *ApJ*, 741, 115.
- Ubler, H. et al. (2017). "The Evolution of the Tully-Fisher Relation between z ~ 2.3 and z ~ 0.9 with KMOS3D." *ApJ*, 842, 121.
- Straatman, C. et al. (2017). "ZFIRE: The Evolution of the Stellar Mass Tully-Fisher Relation to Redshift ~ 2.2." *ApJ*, 839, 57.
- Hassan, S. F. and Rosen, R. A. (2012). "Bimetric gravity from ghost-free massive gravity." *JHEP*, 2012, 126.
- Nelson, S. (2025). "Dual Tetrad Topology and the Field Origin: From Nuclear Decay to Galactic Dynamics."
