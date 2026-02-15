# The Hubble Tension: A Century of Disagreement, Resolved

## A Number That Should Be Simple

In 1929, Edwin Hubble measured something that should have been straightforward: how fast is the universe expanding? He pointed his telescope at distant galaxies, measured how quickly they were receding, and divided by their distance. The result was a single number, now called the Hubble constant, H0.

Nearly a century later, we still cannot agree on what that number is.

Two of our most precise measurement techniques give answers that flatly contradict each other. The Planck satellite, studying the afterglow of the Big Bang (the cosmic microwave background), gives **H0 = 67.4 +/- 0.5 km/s/Mpc**. The SH0ES collaboration, using Cepheid variable stars and supernovae as cosmic rulers, gives **H0 = 73.0 +/- 1.0 km/s/Mpc**. These measurements disagree by more than 5 standard deviations: the probability that both are measuring the same value is less than 1 in 3 million.

This disagreement is called the **Hubble tension**, and it has consumed thousands of research papers, multiple space missions, and decades of effort. The tension is not a matter of sloppy measurement. Both teams have checked their work exhaustively. Something deeper is at play.

What if the problem is not with our telescopes, but with our assumptions about the medium through which the measurement travels?

---

## Gravity Has Structure

To understand the resolution, we need to revisit something that textbook physics already tells us but that cosmology has largely overlooked: **gravity has internal structure**.

General Relativity describes gravity through the metric tensor, a mathematical object that encodes how spacetime curves. In 1928, Einstein himself showed that this metric can be decomposed into a set of four basis vectors called a **tetrad** (or vierbein). These four vectors, one for time and three for space, define a local reference frame at every point in spacetime. This is not speculative physics; it is the standard tetrad formalism found in Misner, Thorne, and Wheeler's *Gravitation* and Wald's *General Relativity*.

The tetrad tells us something important: **the gravitational field has a local frame with definite structure.** At every point, four directions must be defined to fully specify the field. Fewer than four and the frame is incomplete; the metric becomes degenerate and the physics breaks down. This requirement, that the field must fully close at every point, is called **field closure**.

> For a deeper introduction to how the gravitational field works as a structured medium, see the [Exploring the Gravity Field](/field) page.

Having established that gravity has structure, the natural next question is: what shape does that structure take?

---

## Oranges, Tetrahedra, and the Shape of Gravity

Consider oranges stacked at a grocery store. Each orange in the interior of the pile touches 12 neighbours, and this is the most efficient way to fill three-dimensional space with spheres. Johannes Kepler conjectured this in 1611; Thomas Hales proved it rigorously in 1998.

But look more carefully at the gaps between those oranges. Where three adjacent oranges meet, they create a small pocket. Count the faces of that pocket and you find four: it is a **tetrahedron**, the simplest solid that encloses volume in three dimensions. You cannot build a closed container with fewer than four walls. This is not a choice or a model; it is a theorem of geometry called the **simplex theorem**: in *d* spatial dimensions, the minimum number of faces needed to enclose a volume is *d* + 1. For our three-dimensional universe, that gives **k = 4**.

Now apply this to gravity. Gravitational field regions are distributed, locally coupled, and packed in three dimensions, just like those oranges. Each field region couples to its neighbours through shared boundaries. The simplex theorem tells us the minimal coupling unit: a tetrahedral chamber with k = 4 faces.

But a single tetrahedron is not self-consistent. Four independent arguments from established physics (spinor theory, Ashtekar's chiral decomposition, ghost-free bimetric gravity, and stability analysis) all require a **second tetrahedron** with opposite orientation. The result is two interlocking tetrahedra: a structure called the **stellated octahedron**, also known as a Star of David in two-dimensional cross section.

This dual tetrad structure has a definite field coupling architecture. It couples at three levels:

| Level | Channels | Source |
|-------|----------|--------|
| Field existence (the origin) | 1 | The shared vertex where both tetrahedra meet |
| Propagation (vierbein faces) | k = 4 | The four faces of the tetrahedral chamber |
| Bimetric coupling | k^2 = 16 | The 16 Hassan-Rosen interaction channels |
| **Total** | **f(k) = 1 + k + k^2 = 21** | **The coupling polynomial** |

Of the 12 geometric neighbours touching each field region (just like those oranges), only **4 are open coupling channels** (the tetrahedral faces). The remaining **8 are geometrically closed**: they touch, but they do not transmit the field coupling. This is not a parameter; it is the geometry of three-dimensional packing.

---

## The Aperture Throughput

Here is where the story turns from geometry to measurement.

In optics, every lens has an **aperture throughput**: the fraction of incoming light that actually passes through the optical system. A perfect lens transmits 100%. A real lens, with its finite opening and curved surfaces, transmits less. The ratio of what gets through to what was sent is the throughput.

Gravity's tetrahedral field structure acts as a measurement lens. When we observe the Hubble constant, our measurement path goes from the **local frame** (flat, tetrahedral, characterized by k = 4) outward to the **cosmological horizon** (curved, spherical, characterized by pi). These are two fundamentally different geometries:

- **Local frame**: flat, discrete, with k = 4 faces. This is the tetrad.
- **Cosmological boundary**: curved, continuous, with pi governing its geometry. A sphere's area is 4 pi r^2.

The ratio between these two geometries defines the aperture:

```
    Geometric ratio:   k / pi  =  4 / pi  =  1.273
```

The Hubble constant depends on length (distance), not area. Since area scales as length squared, we take the square root to get the length conversion:

```
    Aperture throughput:   sqrt(k / pi)  =  sqrt(4 / pi)  =  1.128
```

This is the same mathematical operation that appears throughout optics: converting between aperture area and effective diameter. A lens with 4x the area has only 2x the diameter. Similarly, if flat and curved geometries differ by k/pi in their characterizations, lengths differ by sqrt(k/pi).

This is not a fitted parameter. It follows directly from k = 4, which follows directly from three spatial dimensions.

---

## Why the Measurements Disagree

The MOND acceleration scale a0, established empirically from galaxy rotation curves (Milgrom 1983), connects to the Hubble constant through horizon thermodynamics:

```
    a0  =  c * H / (2 * pi)
```

Inverting this for H in abstract flat space (no structure, no topology):

```
    H_flat  =  2 * pi * a0 / c  =  78.90 km/s/Mpc
```

Yet no one has ever measured H near 79. A century of observations places it between 67 and 73. The naive MOND prediction overshoots every measurement by roughly 12%.

The resolution: **we do not live in abstract flat space.** We measure through a structured field. Applying the aperture correction:

```
    H_measured  =  H_flat / sqrt(k / pi)
               =  78.90 / 1.128
               =  69.92 km/s/Mpc
```

This matches the GW170817 gravitational wave measurement (70.0 +12/-8 km/s/Mpc) to **0.11%**. The gravitational wave standard siren requires no distance ladder and no CMB model assumptions; it measures H directly from wave amplitude and host galaxy redshift. It is the purest spacetime measurement we have.

**The scatter itself is predicted.** Different measurement methods sample different portions of the frame transition from flat tetrad to curved horizon. The geometric mean of partial corrections through the aperture gives:

```
    Scatter factor  =  (k / pi)^(1/4)  =  1.062    (+/- 6.2%)
```

This defines a **measurement cage** spanning 65.8 to 74.3 km/s/Mpc. All six major measurement methods (Planck, BAO, GW170817, TRGB, SH0ES, Masers) fall within this range.

| Method | H0 (km/s/Mpc) | Deviation from 69.92 |
|--------|---------------|---------------------|
| Planck (CMB) | 67.4 +/- 0.5 | -3.6% |
| BAO | 68.0 +/- 1.5 | -2.7% |
| GW170817 | 70.0 +12/-8 | +0.1% |
| TRGB | 72.5 +/- 1.7 | +3.7% |
| SH0ES | 73.0 +/- 1.0 | +4.4% |
| Masers | 73.9 +/- 3.0 | +5.7% |

Early-universe methods (Planck, BAO) integrate from cosmological distances, accumulating more of the curved-boundary correction, and measure lower. Late-universe methods (SH0ES, TRGB, Masers) measure more locally, sample less of the correction, and measure higher. The "tension" between 67 and 73 is not a crisis. It is the predicted topological signature of measuring through k = 4 structure.

---

## The GFD Derivation: H from First Principles

The aperture correction applied to the empirical MOND value gives H = 69.92 km/s/Mpc. But Gravity Field Dynamics (GFD) goes further: it derives H entirely from the topology, with zero free parameters.

The derivation chain starts from the number of spatial dimensions and reaches the Hubble constant in six steps:

**Step 1.** Three spatial dimensions require k = d + 1 = 4 faces to enclose a volume (simplex theorem).

**Step 2.** The k = 4 tetrahedral topology creates k^2 = 16 bimetric coupling channels. These channels define a characteristic acceleration at the electron scale:

```
    a0  =  k^2 * G * m_e / r_e^2
        =  16 * (6.674e-11) * (9.109e-31) / (2.818e-15)^2
        =  1.225e-10 m/s^2
```

This matches the empirical MOND scale (1.22e-10 m/s^2) to **0.41%**.

**Step 3.** The acceleration a0 connects to the Hubble expansion rate through the Hubble horizon. At the Hubble radius R_H = c/H, the recession velocity equals light speed. The factor 2*pi enters from the Unruh relation, converting geometric acceleration to a physical threshold:

```
    a0  =  (c * H) / (2 * pi) * sqrt(k / pi)
```

**Step 4.** Invert for H:

```
    H  =  2 * pi * a0 / (c * sqrt(k / pi))
       =  2 * pi * (1.225e-10) / (3.0e8 * 1.128)
       =  70.21 km/s/Mpc
```

This is the **tree-level** GFD prediction. Zero parameters fitted to observation.

**Step 5.** The Schwinger one-loop QED correction alpha/(2*pi) = 0.116% propagates through the chain G -> a0 -> H. This is the same universal correction that shifts the electron's magnetic moment from 2 to 2(1 + alpha/(2*pi)):

```
    H_one-loop  =  70.29 km/s/Mpc
```

**Step 6.** The combined formula, substituting the expression for a0:

```
    H  =  (2 * pi * k^2) / (c * sqrt(k / pi))  *  G * m_e / r_e^2
```

Every input is either a measured constant (G, m_e, r_e, c) or a topological integer (k = 4). No parameters are adjusted. The predicted value, 70.29 km/s/Mpc, falls squarely between Planck (67.4) and SH0ES (73.0), consistent with the purest measurement available: GW170817 at 70.0 km/s/Mpc.

---

## What the Tension Was Telling Us

The Hubble tension is not a failure of measurement. It is gravity's internal structure making itself known.

For a century, cosmology has treated the gravitational field as a featureless continuum: spacetime with a single constant G and no internal architecture. This is equivalent to treating a crystal as a featureless solid and then being surprised when it diffracts light. The crystal diffracts because it has lattice structure. Gravity "diffracts" because it has tetrahedral structure.

The resolution requires no new particles, no new forces, and no new free parameters. It requires only that we take seriously what textbook physics already tells us: gravity has a field topology determined by three spatial dimensions, and measurements through that topology carry its geometric signature.

The disagreement between 67 and 73 km/s/Mpc was never a crisis requiring exotic new physics. It was a clue, hidden in plain sight, pointing to the internal structure of the gravitational field itself.

---

## References

- Hubble, E. (1929). "A relation between distance and radial velocity among extra-galactic nebulae." *Proceedings of the National Academy of Sciences*, 15(3), 168-173.
- Planck Collaboration (2020). "Planck 2018 results. VI. Cosmological parameters." *A&A*, 641, A6.
- Riess, A. et al. (2022). "A Comprehensive Measurement of the Local Value of the Hubble Constant." *ApJ*, 934, L7.
- Abbott, B. et al. (2017). "A gravitational-wave standard siren measurement of the Hubble constant." *Nature*, 551, 85.
- Milgrom, M. (1983). "A modification of the Newtonian dynamics as a possible alternative to the hidden mass hypothesis." *ApJ*, 270, 365.
- Hassan, S. F. and Rosen, R. A. (2012). "Bimetric gravity from ghost-free massive gravity." *JHEP*, 2012, 126.
- Hales, T. (2005). "A proof of the Kepler conjecture." *Annals of Mathematics*, 162, 1065-1185.
- Nelson, S. (2025). "Hubble Tension Resolution via Tetrahedral Field Topology."
- Nelson, S. (2025). "Dual Tetrad Topology and the Field Origin."
