# GRAVIS: Galactic Rotation from Field Dynamics

A zero-parameter rotation curve prediction tool using **Gravity Field Dynamics**, implementing the **Dual Tetrad Gravity** (DTG) covariant completion.

Author: Stephen Nelson | Source: [github.com/fielddynamics/gravis](https://github.com/fielddynamics/gravis)

Supplemental tool for: Nelson, S. (2026). *Dual Tetrad Topology and the Field Origin: From Nuclear Decay to Galactic Dynamics*.

Given only the independently measured baryonic mass distribution of a galaxy (stellar bulge, stellar disk, gas disk), GRAVIS computes the full rotation curve with no dark matter and no free parameters. The characteristic acceleration scale $a_0$ is derived from first principles:

$$a_0 = k^2 \, \frac{G \, m_e}{r_e^2} \approx 1.22 \times 10^{-10} \; \text{m/s}^2$$

where $k = 4$ is the simplex number for $d = 3$, $m_e$ is the electron mass, and $r_e$ is the classical electron radius. This matches the empirical MOND acceleration scale to within measurement uncertainty.

## What It Does

**Forward prediction (M -> v):** Supply a three-component baryonic mass model (Hernquist bulge + exponential stellar disk + exponential gas disk) and GRAVIS returns three rotation curves:

- **Newtonian** -- standard $1/r^2$ gravity, baryons only
- **Dual Tetrad Gravity** -- AQUAL field equation with $\mu(x) = x/(1+x)$, derived from topological closure
- **Classical MOND** -- AQUAL field equation with $\mu(x) = x/\sqrt{1+x^2}$ (Bekenstein & Milgrom 1984)

**Inverse inference (v -> M):** Given an observed circular velocity at a radius, infer the enclosed baryonic mass required by DTG.

## Quick Start

```bash
# Clone
git clone https://github.com/fielddynamics/gravis.git
cd gravis

# Install
pip install -r requirements.txt

# Run
python app.py
```

Open http://localhost:5000 in your browser.

## Project Structure

```
gravis/
  app.py                  Flask application entry point
  requirements.txt        Python dependencies
  physics/
    constants.py          CODATA 2022 / IAU 2015 physical constants
    mass_model.py         Distributed mass model (Hernquist + exponential)
    newtonian.py          Newtonian gravity: v_c = sqrt(G M(<r) / r)
    aqual.py              DTG solver: mu(x) = x/(1+x), AQUAL field equation
    mond.py               Classical MOND solver: mu(x) = x/sqrt(1+x^2)
    inference.py          Inverse problem: observed v -> inferred M
  api/
    routes.py             REST API endpoints
  data/
    galaxies.py           Galaxy catalog with observational data
  templates/
    index.html            Web frontend
  static/
    css/gravis.css        Stylesheet
    js/gravis.js          Frontend logic (Chart.js)
  tests/
    test_constants.py     Physical constant validation
    test_mass_model.py    Mass distribution tests
    test_aqual.py         DTG solver tests
    test_mond.py          MOND solver tests
    test_inference.py     Round-trip consistency (M -> v -> M)
    test_milky_way.py     30-test Milky Way validation suite
    test_api.py           API integration tests
    test_galaxies.py      Galaxy catalog integrity tests
```

## Physics

### AQUAL Field Equation

Both DTG and classical MOND solve the same class of field equation:

$$\mu\!\left(\frac{|\nabla\Phi|}{a_0}\right) \nabla\Phi = \nabla\Phi_N$$

where $\Phi_N$ is the Newtonian potential. The two theories differ only in the constitutive law $\mu(x)$:

| Theory | Constitutive Law | Origin |
|--------|-----------------|--------|
| DTG | $\mu(x) = x/(1+x)$ | Topological closure of dual tetrahedron metric |
| MOND | $\mu(x) = x/\sqrt{1+x^2}$ | Empirical fit (Bekenstein & Milgrom 1984) |

Both admit closed-form solutions. DTG's function was independently identified as the empirically preferred form by Famaey & Binney (2005).

### Galaxy Mass Model

Each galaxy is decomposed into three independently measured baryonic components:

| Component | Profile | Parameters |
|-----------|---------|------------|
| Stellar bulge | Hernquist (1990) | Mass $M_b$, scale radius $a$ |
| Stellar disk | Exponential (Freeman 1970) | Mass $M_d$, scale length $R_d$ |
| Gas disk (HI + H2 + He) | Exponential | Mass $M_g$, scale length $R_g$ |

No dark matter halos are used anywhere. All mass parameters come from photometry, stellar population models, and 21 cm HI surveys.

### Included Galaxies

| Galaxy | $M_{\text{baryon}}$ | Type | Sources |
|--------|---------------------|------|---------|
| Milky Way | $7.5 \times 10^{10}\;M_\odot$ | SBbc | Bland-Hawthorn+2016, McMillan 2017, Kalberla+2009 |
| Andromeda (M31) | $1.1 \times 10^{11}\;M_\odot$ | SA(s)b | Tamm+2012, Barmby+2006, Braun+2009 |
| NGC 3198 | $3.4 \times 10^{10}\;M_\odot$ | SBc | SPARC (Lelli+2016), Begeman 1989 |
| NGC 2403 | $9.3 \times 10^{9}\;M_\odot$ | SABcd | SPARC (Lelli+2016), Fraternali+2002 |
| M33 | $5.0 \times 10^{9}\;M_\odot$ | SA(s)cd | Corbelli+2003, Corbelli+2014 |
| NGC 6503 | $8.7 \times 10^{9}\;M_\odot$ | SA(s)cd | SPARC (Lelli+2016), de Blok+2008 |

Rotation curve data include 1-sigma error bars from published measurements.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/galaxies` | GET | List all galaxies |
| `/api/galaxies/<id>` | GET | Get galaxy details |
| `/api/rotation-curve` | POST | Compute rotation curves |
| `/api/infer-mass` | POST | Infer mass from velocity |
| `/api/constants` | GET | Physical constants |

## Tests

```bash
pip install pytest
pytest tests/ -v
```

The test suite validates physical constants against CODATA/IAU values, checks solver accuracy, verifies round-trip consistency (M -> v -> M), and runs 30 cross-theory tests against the Milky Way.

## Citation

If you use GRAVIS in your research, please cite:

```bibtex
@software{nelson2026gravis,
  author  = {Nelson, Stephen},
  title   = {GRAVIS: Galactic Rotation from Field Dynamics},
  year    = {2026},
  url     = {https://github.com/fielddynamics/gravis}
}

@article{nelson2026dtg,
  author  = {Nelson, Stephen},
  title   = {Dual Tetrad Topology and the Field Origin:
             From Nuclear Decay to Galactic Dynamics},
  year    = {2026}
}
```

## License

[MIT](LICENSE)
