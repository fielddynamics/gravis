// =====================================================================
// FAQ
// =====================================================================

const FAQ_DATA = [
    {
        category: "General",
        items: [
            {
                q: "What is GRAVIS?",
                a: "<p>GRAVIS (<strong>G</strong>alactic <strong>R</strong>otation from Field Dynamics) is a zero-parameter galactic rotation curve prediction tool. It computes rotation curves from baryonic mass alone using the dual tetrad covariant completion, comparing the result against Newtonian gravity, classical MOND, and the standard cosmological dark matter model on a single plot.</p>"
            },
            {
                q: "What does 'zero-parameter' mean?",
                a: "<p>Unlike dark matter models that require fitting halo parameters (mass, concentration, scale radius) to each galaxy individually, Gravity Field Dynamics derives its acceleration scale and Lagrangian from the topology of the gravitational field. There are no free parameters tuned to match observations. The same action applies to every galaxy.</p><div class='faq-advanced'><div class='faq-advanced-label'>What this means concretely</div><p>A standard CDM rotation curve fit requires choosing at least a halo mass M<sub>200</sub> and concentration c for each galaxy -- that is 2 free parameters per galaxy. For 175 SPARC galaxies, that is 350 fitted numbers. GFD uses the same Lagrangian F(y) and the same topologically derived a<sub>0</sub> for all of them, with zero per-galaxy fitting. The prediction is fully determined once you know the baryonic mass distribution.</p></div>"
            },
            {
                q: "Who developed this?",
                a: "<p>GRAVIS implements the theory developed by <strong>Stephen Nelson</strong> in \"Dual Tetrad Topology and the Field Origin: From Nuclear Decay to Galactic Dynamics\" (2026). The software and theory are independent research.</p>"
            },
            {
                q: "Is the source code available?",
                a: "<p>Yes. The full source code, including the physics engine, API, and test suite, is available at <strong>github.com/fielddynamics/gravis</strong>.</p>"
            }
        ]
    },
    {
        category: "Physics",
        items: [
            {
                q: "What is Gravity Field Dynamics (GFD)?",
                a: "<p>Gravity Field Dynamics is the covariant completion of dual tetrad gravity. It describes the gravitational field as two coupled tetrahedral field chambers that form a stellated octahedron, with a coupling hierarchy governed by the simplex number k = 4. The topology uniquely determines the scalar Lagrangian, the acceleration scale, and the complete gravitational action, with no free parameters.</p><div class='faq-advanced'><div class='faq-advanced-label'>The covariant action</div><p>The full scalar-tensor action is:</p><span class='faq-eq'>S = &int; d<sup>4</sup>x &radic;(&minus;g) [ R/16&pi;G &minus; (a<sub>0</sub><sup>2</sup>/8&pi;G) F(|&nabla;&Phi;|<sup>2</sup>/a<sub>0</sub><sup>2</sup>) &minus; &frac14; e<sup>&minus;2&Phi;/c<sup>2</sup></sup> F<sub>&mu;&nu;</sub>F<sup>&mu;&nu;</sup> + L<sub>matter</sub> ]</span><p>where the scalar Lagrangian density F(y) is uniquely determined by the coupling polynomial f(k) = 1 + k + k<sup>2</sup>:</p><span class='faq-eq'>F(y) = y/2 &minus; &radic;y + ln(1 + &radic;y)</span><span class='faq-eq-label'>y = |&nabla;&Phi;|<sup>2</sup> / a<sub>0</sub><sup>2</sup></span><p>The three terms of F(y) correspond directly to the three structural levels of the stellated octahedron: the quadratic term (k<sup>2</sup> = 16 coupled face channels), the square-root term (k = 4 face couplings), and the logarithmic term (k<sup>0</sup> = 1, the Field Origin). GRAVIS evaluates the field equation that follows from this Lagrangian.</p></div>"
            },
            {
                q: "What is the acceleration scale a0?",
                a: "<p>The acceleration scale is the characteristic threshold below which the gravitational field cannot sustain full coupling across all k<sup>2</sup> = 16 coupling channels. It is derived from the topology:</p><span class='faq-eq'>a<sub>0</sub> = k<sup>2</sup> G m<sub>e</sub> / r<sub>e</sub><sup>2</sup> &asymp; 1.22 &times; 10<sup>&minus;10</sup> m/s<sup>2</sup></span><p>where m<sub>e</sub> is the electron mass and r<sub>e</sub> is the classical electron radius. Above a<sub>0</sub>, gravity is Newtonian. Below it, the field enters partial coupling and the effective gravitational response is enhanced.</p><div class='faq-advanced'><div class='faq-advanced-label'>Computing a<sub>0</sub> from constants</div><p>With k = 4 (the simplex number in d = 3 spatial dimensions):</p><p>a<sub>0</sub> = 16 &times; (6.674 &times; 10<sup>&minus;11</sup>) &times; (9.109 &times; 10<sup>&minus;31</sup>) / (2.818 &times; 10<sup>&minus;15</sup>)<sup>2</sup></p><p>= 16 &times; 6.674 &times; 9.109 / 7.940 &times; 10<sup>&minus;11&minus;31+30</sup></p><p>= <strong>1.224 &times; 10<sup>&minus;10</sup> m/s<sup>2</sup></strong></p><p>This uses only CODATA 2022 fundamental constants and the integer k = 4. No fitting. The result matches the empirically determined MOND acceleration scale to within a few percent.</p></div>"
            },
            {
                q: "How does GFD differ from MOND?",
                a: "<p>MOND (Milgrom 1983) is an empirical modification of Newtonian dynamics. It chooses an interpolating function and fits an acceleration scale to galactic data. GFD derives both from the topology of the gravitational field: the Lagrangian F(y) is uniquely determined by the coupling polynomial, and a<sub>0</sub> is computed from fundamental constants. The predictions are similar at large radii because both field equations share the same deep-field limit, but they differ in the transition regime and -- crucially -- in origin.</p><div class='faq-advanced'><div class='faq-advanced-label'>Two different field equations</div><p>Both theories produce a field equation relating the true gravitational field strength g to the Newtonian value g<sub>N</sub>. Writing x = g/a<sub>0</sub> and y<sub>N</sub> = g<sub>N</sub>/a<sub>0</sub>:</p><span class='faq-eq'><strong>GFD:</strong>&ensp; x<sup>2</sup> / (1 + x) = y<sub>N</sub>&emsp;&emsp;<strong>MOND:</strong>&ensp; x<sup>2</sup> / &radic;(1 + x<sup>2</sup>) = y<sub>N</sub></span><p>Both are analytically solvable. Both reduce to x<sup>2</sup> = y<sub>N</sub> when x &Lt; 1 (deep field limit), which gives v<sup>4</sup> = GMa<sub>0</sub> (the Baryonic Tully-Fisher Relation). Both reduce to x = y<sub>N</sub> when x &Gt; 1 (Newtonian limit). They differ in the transition between these regimes.</p><p>At x = 1 (the transition point), GFD gives y<sub>N</sub> = 0.5 while MOND gives y<sub>N</sub> = 1/&radic;2 &asymp; 0.707. This means GFD enhances gravity more strongly in the transition regime, which is why the blue GFD curve in GRAVIS always sits above the green MOND curve.</p><p>The key difference is not numerical but structural: MOND's field equation is an empirical choice. GFD's field equation is derived from the Lagrangian F(y), which is itself determined by the dual tetrad topology.</p></div>"
            },
            {
                q: "How does GFD differ from dark matter (CDM)?",
                a: "<p>The standard model posits invisible dark matter halos around galaxies. Fitting these halos to observed rotation curves requires at minimum 2 free parameters per galaxy (halo mass and concentration), and often more. GFD achieves the same empirical agreement using only the measured baryonic mass, with zero fitted parameters. GRAVIS displays both predictions side by side for direct comparison.</p><div class='faq-advanced'><div class='faq-advanced-label'>Parameter count comparison</div><p>For a single galaxy, a CDM NFW halo fit requires:</p><ul><li><strong>M<sub>200</sub></strong>: Total halo mass (fitted or from abundance matching)</li><li><strong>c</strong>: Concentration parameter (fitted or from a c-M relation)</li><li>Optionally: stellar mass-to-light ratio, distance, inclination</li></ul><p>GFD requires:</p><ul><li>The baryonic mass distribution M(r) (independently measured, not fitted)</li><li>Nothing else. The Lagrangian F(y) and a<sub>0</sub> are fixed by the topology.</li></ul><p>When GRAVIS shows both curves matching the observations, it illustrates that the same empirical success can be achieved with zero free parameters instead of two or more.</p></div>"
            },
            {
                q: "What is the dual tetrad?",
                a: "<p>The tetrad formulation of general relativity decomposes the metric into four basis vectors at each spacetime point, forming a tetrahedron -- the minimal gravitational field chamber. Four independent arguments (stability, spinor structure, chirality, and the bimetric framework) establish that two such tetrahedral field chambers are required, coupled with opposite orientations to form a stellated octahedron. This dual tetrad structure is the foundation of GFD.</p>"
            },
            {
                q: "What is the Field Origin?",
                a: "<p>The Field Origin is the single interaction vertex at the center of the stellated octahedron where both field chambers meet. It corresponds to the constant term (k<sup>0</sup> = 1) of the coupling polynomial f(k) = 1 + k + k<sup>2</sup>, and to the logarithmic term ln(1 + &radic;y) in the Lagrangian F(y). The Field Origin persists at all acceleration scales and ensures the smooth, continuous transition from Newtonian gravity to the enhanced regime.</p>"
            },
            {
                q: "What is the coupling polynomial f(k)?",
                a: "<p>The coupling polynomial <strong>f(k) = 1 + k + k<sup>2</sup></strong> encodes the three structural levels of the stellated octahedron:</p><ul><li><strong>k<sup>0</sup> = 1</strong>: The Field Origin (single interaction vertex)</li><li><strong>k<sup>1</sup> = 4</strong>: Face coupling channels through the vierbein</li><li><strong>k<sup>2</sup> = 16</strong>: Full field interaction (4 faces &times; 4 faces)</li></ul><p>These three levels determine the complete coupling architecture of the gravitational field and map directly onto the three terms of the Lagrangian density F(y) = y/2 &minus; &radic;y + ln(1 + &radic;y).</p>"
            },
            {
                q: "What is the Lagrangian F(y)?",
                a: "<p>The scalar Lagrangian density F(y) is the central object of the theory. It governs how the scalar field &Phi; in the covariant action responds to matter. It is uniquely determined by the dual tetrad topology:</p><span class='faq-eq'>F(y) = y/2 &minus; &radic;y + ln(1 + &radic;y)</span><span class='faq-eq-label'>y = |&nabla;&Phi;|<sup>2</sup> / a<sub>0</sub><sup>2</sup></span><p>The three terms map to the three structural levels of the coupling polynomial. GRAVIS evaluates the field equation that follows from varying this Lagrangian.</p><div class='faq-advanced'><div class='faq-advanced-label'>From Lagrangian to field equation</div><p>The Euler-Lagrange equation from the action gives, in spherical symmetry, an algebraic relation between the true gravitational field g and the Newtonian field g<sub>N</sub> = GM/r<sup>2</sup>. Writing x = g/a<sub>0</sub>:</p><span class='faq-eq'>x<sup>2</sup> / (1 + x) = g<sub>N</sub> / a<sub>0</sub></span><p>This is a quadratic in x with the analytic solution:</p><span class='faq-eq'>x = (y<sub>N</sub> + &radic;(y<sub>N</sub><sup>2</sup> + 4 y<sub>N</sub>)) / 2&emsp; where y<sub>N</sub> = g<sub>N</sub>/a<sub>0</sub></span><p>This is all GRAVIS computes. Given the enclosed baryonic mass at radius r, it evaluates g<sub>N</sub>, solves this quadratic, and returns v = &radic;(g &middot; r). No numerical integration, no iteration, no fitting -- just a closed-form solution from the Lagrangian.</p></div>"
            }
        ]
    },
    {
        category: "Using GRAVIS",
        items: [
            {
                q: "What is Prediction mode?",
                a: "<p><strong>Prediction mode (M &rarr; v)</strong> takes a known baryonic mass distribution and computes the expected rotation curve. This is the forward problem: given the mass, what velocities do Newtonian gravity, GFD, and MOND each predict? Observational data points with error bars are overlaid for direct comparison.</p><div class='faq-advanced'><div class='faq-advanced-label'>Forward computation chain</div><p>For each radius r on the chart, GRAVIS performs these steps:</p><div class='faq-step'><span class='faq-step-num'>1</span><span class='faq-step-text'>Compute enclosed baryonic mass M(&lt;r) from the three-component mass model (bulge + disk + gas)</span></div><div class='faq-step'><span class='faq-step-num'>2</span><span class='faq-step-text'>Compute Newtonian acceleration: <strong>g<sub>N</sub> = GM(r) / r<sup>2</sup></strong></span></div><div class='faq-step'><span class='faq-step-num'>3</span><span class='faq-step-text'>Solve the field equation from the Lagrangian: <strong>x = (y<sub>N</sub> + &radic;(y<sub>N</sub><sup>2</sup> + 4y<sub>N</sub>)) / 2</strong> where y<sub>N</sub> = g<sub>N</sub>/a<sub>0</sub></span></div><div class='faq-step'><span class='faq-step-num'>4</span><span class='faq-step-text'>True gravitational acceleration: <strong>g = a<sub>0</sub> &middot; x</strong></span></div><div class='faq-step'><span class='faq-step-num'>5</span><span class='faq-step-text'>Circular velocity: <strong>v = &radic;(g &middot; r)</strong></span></div><p>Steps 2-5 are repeated for Newton (skip step 3, use g = g<sub>N</sub> directly), MOND (step 3 uses the MOND field equation instead), and CDM (step 2 adds the NFW halo mass).</p></div>"
            },
            {
                q: "What is Inference mode?",
                a: "<p><strong>Inference mode (v &rarr; M)</strong> solves the inverse problem: given an observed velocity at a particular radius, what enclosed baryonic mass does the Lagrangian require? The multi-point consistency test then checks whether every observation point independently infers the same total mass. Agreement across the full radial range, without parameter tuning, is the non-trivial result.</p><div class='faq-advanced'><div class='faq-advanced-label'>Inverse computation chain</div><p>The field equation x<sup>2</sup>/(1+x) = g<sub>N</sub>/a<sub>0</sub> is bidirectional. The forward direction (prediction) knows g<sub>N</sub> and must solve a quadratic for x. The inverse direction (inference) knows x directly from the observation and just evaluates the left side -- no solver needed:</p><div class='faq-step'><span class='faq-step-num'>1</span><span class='faq-step-text'>From the observed circular velocity: <strong>g = v<sup>2</sup> / r</strong></span></div><div class='faq-step'><span class='faq-step-num'>2</span><span class='faq-step-text'>Dimensionless field strength: <strong>x = g / a<sub>0</sub></strong></span></div><div class='faq-step'><span class='faq-step-num'>3</span><span class='faq-step-text'>Evaluate the field equation: <strong>g<sub>N</sub> = a<sub>0</sub> &middot; x<sup>2</sup> / (1 + x)</strong></span></div><div class='faq-step'><span class='faq-step-num'>4</span><span class='faq-step-text'>Enclosed mass: <strong>M = g<sub>N</sub> &middot; r<sup>2</sup> / G</strong></span></div><p>This is algebraic evaluation, not curve fitting. The forward direction requires solving a quadratic. The inverse direction requires only evaluating x<sup>2</sup>/(1+x) -- one line of arithmetic.</p><p><strong>Multi-point consistency:</strong> For each observation (r<sub>i</sub>, v<sub>i</sub>), GRAVIS infers M<sub>enc,i</sub> via the steps above. It then computes what fraction of the model's total mass is enclosed at r<sub>i</sub>, and scales up: M<sub>total,i</sub> = M<sub>enc,i</sub> / f<sub>i</sub>. If the theory and mass model shape are both correct, all M<sub>total,i</sub> should agree.</p></div>"
            },
            {
                q: "What are the Mass Distribution sliders?",
                a: "<p>These control the three-component baryonic mass model:</p><ul><li><strong>Stellar Bulge</strong>: Hernquist profile with mass M and scale radius a</li><li><strong>Stellar Disk</strong>: Exponential disk with mass M and scale length R<sub>d</sub></li><li><strong>Gas Disk</strong>: Exponential disk (HI + H<sub>2</sub> + He) with mass M and scale length R<sub>d</sub></li></ul><p>These parameters come from independent photometric and HI measurements. They are not fitted to rotation curves.</p>"
            },
            {
                q: "What do the example galaxies show?",
                a: "<p>Each example loads a published baryonic mass model and observed rotation curve data from the literature. The mass model is derived from photometric observations (stellar mass from 3.6 micron luminosity) and radio observations (gas mass from 21 cm HI emission). Click <strong>Data Sources</strong> on the chart to see the full provenance, including references.</p>"
            },
            {
                q: "Can I adjust sliders after loading an example?",
                a: "<p>Yes. The observational data points remain pinned to the chart when you move sliders, allowing you to fine-tune the mass model and see how GFD, Newton, MOND, and CDM all respond. This is useful for exploring parameter sensitivity and understanding which component (bulge, disk, gas) most affects the fit in different radial regions.</p>"
            },
            {
                q: "What is the Acceleration Regime slider?",
                a: "<p>This multiplies the topologically derived acceleration scale a<sub>0</sub>. At the default value of 1.0, GFD uses the theory's prediction. Adjusting it lets you explore what happens if the transition scale were different -- it is a diagnostic tool, not a free parameter in the theory.</p><div class='faq-advanced'><div class='faq-advanced-label'>What changing a<sub>0</sub> does to the field equation</div><p>The field equation x<sup>2</sup>/(1+x) = g<sub>N</sub>/a<sub>0</sub> has a<sub>0</sub> in the denominator on the right side. Increasing a<sub>0</sub> makes y<sub>N</sub> = g<sub>N</sub>/a<sub>0</sub> smaller, which pushes more of the galaxy into the enhanced-gravity regime. This raises the predicted velocity curve. Decreasing a<sub>0</sub> pushes the galaxy toward Newtonian behavior, lowering the curve. At a<sub>0</sub> &rarr; &infin;, the entire galaxy is in the deep-field regime. At a<sub>0</sub> &rarr; 0, GFD reduces to Newton everywhere.</p></div>"
            },
            {
                q: "What is the CDM + NFW curve?",
                a: "<p>This is the standard cosmological prediction using a Navarro-Frenk-White (NFW) dark matter halo. When observational data is available, GRAVIS fits the halo mass to minimize chi-squared against the data. Otherwise, it uses abundance matching (Moster+ 2013). The parameter count (typically 2: halo mass and concentration) is shown alongside GFD's zero parameters.</p>"
            }
        ]
    },
    {
        category: "Multi-Point Consistency",
        items: [
            {
                q: "What is the Multi-Point Consistency panel?",
                a: "<p>In inference mode, each observed data point independently implies a total baryonic mass. If the Lagrangian is correct and the mass model shape is accurate, all points should infer the same total mass. The panel shows the per-point results and how well they agree, including velocity residuals, sigma deviations, and aggregate statistics.</p><div class='faq-advanced'><div class='faq-advanced-label'>How per-point mass inference works</div><p>For a single observation (r<sub>i</sub>, v<sub>i</sub>):</p><div class='faq-step'><span class='faq-step-num'>1</span><span class='faq-step-text'>Infer the enclosed mass M<sub>enc,i</sub> by evaluating the field equation inversely (see Inference mode above)</span></div><div class='faq-step'><span class='faq-step-num'>2</span><span class='faq-step-text'>Compute how much of the model mass lies within r<sub>i</sub>: <strong>f<sub>i</sub> = M<sub>model</sub>(&lt;r<sub>i</sub>) / M<sub>model,total</sub></strong></span></div><div class='faq-step'><span class='faq-step-num'>3</span><span class='faq-step-text'>Scale up to infer total mass: <strong>M<sub>total,i</sub> = M<sub>enc,i</sub> / f<sub>i</sub></strong></span></div><p>This assumes the model shape (how mass is distributed between bulge/disk/gas) is correct, but the overall normalization may differ. If M<sub>total,i</sub> is consistent across all radii, the mass model and the Lagrangian together explain the full rotation curve. The enclosed fraction f<sub>i</sub> is shown as the \"M enc.\" column and also determines the weighting: outer points (high f<sub>i</sub>) constrain the total mass more reliably than inner points (low f<sub>i</sub>).</p></div>"
            },
            {
                q: "What do the table columns mean?",
                a: "<p>The inference table columns are:</p><ul><li><strong>r (kpc)</strong>: Observation radius in kiloparsecs</li><li><strong>v (km/s)</strong>: Observed velocity with 1-sigma error bar</li><li><strong>M enc.</strong>: Fraction of total model mass enclosed within this radius -- higher means more constraining</li><li><strong>&Delta;v</strong>: Velocity residual (v<sub>obs</sub> &minus; v<sub>GFD</sub>), the difference between observed and predicted velocity in km/s</li><li><strong>&sigma;</strong>: Sigma deviation |&Delta;v| / error -- how many measurement standard deviations the residual represents</li></ul><div class='faq-advanced'><div class='faq-advanced-label'>Reading the table</div><p><strong>&Delta;v</strong> tells you direction and magnitude: positive means the galaxy rotates faster than the Lagrangian predicts at that radius; negative means slower. <strong>&sigma;</strong> tells you significance: &sigma; &lt; 1 is within measurement noise, 1-2 is marginal, &gt; 2 is a statistically significant discrepancy that may indicate the mass model shape needs adjustment.</p><p>The table is color-coded: green for &sigma; &lt; 1, orange for 1-2, red for &gt; 2. A well-fitting galaxy will be mostly green.</p></div>"
            },
            {
                q: "What is the Mass Offset?",
                a: "<p>The mass offset is the percentage difference between the weighted mean of the per-point inferred masses and the GFD prediction from the mass model. A small offset (a few percent) indicates excellent agreement. A large offset suggests the total baryonic mass may be over- or under-estimated.</p>"
            },
            {
                q: "What are the confidence band methods?",
                a: "<p>The green confidence band on the chart shows the uncertainty range. Five methods are available, each capturing a different aspect of the scatter:</p><ul><li><strong>Weighted RMS</strong>: Root-mean-square deviation of per-point inferred masses from the GFD prediction, weighted by enclosed fraction</li><li><strong>1-sigma Scatter</strong>: Weighted standard deviation of per-point inferences around their mean</li><li><strong>Obs. Error</strong>: Propagated velocity measurement uncertainties through the field equation</li><li><strong>Min-Max</strong>: Full range of per-point inferred masses (most conservative)</li><li><strong>IQR (Robust)</strong>: Interquartile range, resistant to outlier inner points with low enclosed mass</li></ul><div class='faq-advanced'><div class='faq-advanced-label'>Choosing a band method</div><p><strong>Weighted RMS</strong> (default) captures both systematic offset and random scatter -- it shows how far the per-point inferences deviate from the anchor prediction, down-weighting unreliable inner points. <strong>Obs. Error</strong> shows only the measurement noise contribution, ignoring model mismatch. <strong>Min-Max</strong> is the most conservative: the band always contains every data point. <strong>IQR</strong> is robust to outliers. Try switching between them to separate measurement error from model discrepancy.</p></div>"
            },
            {
                q: "What is the Shape Diagnostic?",
                a: "<p>The shape diagnostic compares GFD residuals in the inner half versus the outer half of the galaxy. If both regions show low sigma (&lt; 1.5), the shape fit is excellent. If there is a significant sigma gradient, it suggests specific adjustments to the mass model shape.</p><div class='faq-advanced'><div class='faq-advanced-label'>Interpreting the diagnostic</div><p>The diagnostic splits the observation points into inner and outer halves and computes the average &sigma; in each region:</p><ul><li><strong>Both low (&lt; 1.5)</strong>: \"Excellent shape fit\" -- the mass model's radial distribution matches the data well</li><li><strong>High inner, low outer</strong>: Too much mass concentrated at center -- try increasing the disk or gas Scale length, or decreasing the Bulge Scale radius</li><li><strong>Low inner, high outer</strong>: Mass deficit at outer radii -- try decreasing the disk or gas Scale length to extend mass outward</li><li><strong>Both high, similar &sigma;</strong>: Systematic offset -- the overall mass may need adjustment rather than the shape</li></ul><p>The diagnostic only triggers recommendations when &sigma; values are statistically significant (&ge; 2 in at least one region).</p></div>"
            }
        ]
    },
    {
        category: "Data & Methodology",
        items: [
            {
                q: "Where does the observational data come from?",
                a: "<p>Rotation curve data comes from published kinematic surveys, primarily the <strong>SPARC</strong> database (Lelli, McGaugh &amp; Schombert 2016), <strong>THINGS</strong> (de Blok+ 2008), and <strong>Gaia DR3</strong> (Jiao+ 2023). All data points include 1-sigma error bars from the original publications. Click Data Sources on any example to see the full reference list.</p>"
            },
            {
                q: "How are baryonic masses measured?",
                a: "<p>Stellar masses are derived from Spitzer 3.6 micron luminosity using a fixed mass-to-light ratio (M*/L = 0.5 M<sub>sun</sub>/L<sub>sun</sub>) from stellar population synthesis models. Gas masses come from 21 cm HI observations with a 1.33x correction for primordial helium. Neither component is fitted to rotation curve data.</p><div class='faq-advanced'><div class='faq-advanced-label'>Why 3.6 micron?</div><p>Near-infrared light at 3.6 microns traces the old stellar population that dominates the mass budget, with minimal dust extinction. The mass-to-light ratio at this wavelength is nearly constant across galaxy types (Meidt+ 2014), making M*/L = 0.5 a well-motivated choice from stellar population synthesis -- not a fitted parameter. The 1.33x helium correction on gas mass follows from Big Bang nucleosynthesis: the primordial mass fraction is approximately 24% helium by mass, so M<sub>gas</sub> = 1.33 &times; M<sub>HI</sub>.</p></div>"
            },
            {
                q: "What mass profiles does GRAVIS use?",
                a: "<p>Three independently measured components:</p><ul><li><strong>Hernquist profile</strong> for the stellar bulge: M(&lt;r) = M r<sup>2</sup> / (r + a)<sup>2</sup></li><li><strong>Exponential disk</strong> for the stellar disk: M(&lt;r) = M [1 &minus; (1 + r/R<sub>d</sub>) exp(&minus;r/R<sub>d</sub>)]</li><li><strong>Exponential disk</strong> for the gas: Same form, typically more extended</li></ul><div class='faq-advanced'><div class='faq-advanced-label'>Profile behavior</div><p>The <strong>Hernquist bulge</strong> concentrates mass near the center: half the mass is enclosed within r = a (the scale radius). At large r, it approaches the total mass as r<sup>2</sup>/(r+a)<sup>2</sup> &rarr; 1.</p><p>The <strong>exponential disk</strong> distributes mass more broadly: at r = R<sub>d</sub> (one scale length), only about 26% of the mass is enclosed. At r = 3R<sub>d</sub>, about 80%. The gas disk uses the same functional form but typically has R<sub>d</sub> 2-3 times larger than the stellar disk, reflecting the extended HI distribution observed in 21 cm surveys.</p><p>The total enclosed mass M(&lt;r) is the sum of all three components. This is the input to the field equation at each radius.</p></div>"
            },
            {
                q: "How many galaxies are included?",
                a: "<p>GRAVIS includes examples spanning four orders of magnitude in baryonic mass: the Milky Way, Andromeda (M31), M33, NGC 3198, NGC 2403, NGC 6503, UGC 2885, IC 2574, DDO 154, and NGC 3109. These cover massive spirals, intermediate disks, and gas-dominated dwarfs.</p>"
            },
            {
                q: "Does GRAVIS have an API?",
                a: "<p>Yes. All computations are available through a REST API:</p><ul><li><code>POST /api/rotation-curve</code> -- Compute Newtonian, GFD, MOND, and CDM curves</li><li><code>POST /api/infer-mass</code> -- Infer enclosed mass from a single (r, v) point</li><li><code>POST /api/infer-mass-model</code> -- Infer a scaled mass model from observation + shape</li><li><code>POST /api/infer-mass-multi</code> -- Multi-point inference with consistency analysis</li><li><code>GET /api/galaxies</code> -- List the galaxy catalog</li><li><code>GET /api/constants</code> -- Physical constants used by the engine</li></ul>"
            }
        ]
    }
];

function buildFaqItems() {
    const container = document.getElementById('faq-content');
    if (!container) return;
    container.innerHTML = '';

    FAQ_DATA.forEach(function(section) {
        var catEl = document.createElement('div');
        catEl.className = 'faq-category';
        catEl.textContent = section.category;
        catEl.dataset.category = section.category;
        container.appendChild(catEl);

        section.items.forEach(function(item, idx) {
            var div = document.createElement('div');
            div.className = 'faq-item';
            div.dataset.category = section.category;
            div.dataset.question = item.q.toLowerCase();
            div.dataset.answer = item.a.replace(/<[^>]*>/g, '').toLowerCase();

            div.innerHTML =
                '<div class="faq-question" onclick="toggleFaqItem(this)">' +
                    '<span class="faq-question-text">' + item.q + '</span>' +
                    '<span class="faq-chevron">&#x25B6;</span>' +
                '</div>' +
                '<div class="faq-answer">' + item.a + '</div>';

            container.appendChild(div);
        });
    });
}

function toggleFaqItem(el) {
    var item = el.closest('.faq-item');
    if (item) item.classList.toggle('open');
}

function toggleAllFaq(expand) {
    var items = document.querySelectorAll('.faq-item');
    items.forEach(function(item) {
        if (!item.classList.contains('hidden')) {
            if (expand) {
                item.classList.add('open');
            } else {
                item.classList.remove('open');
            }
        }
    });
}

function filterFaq(query) {
    query = (query || '').trim().toLowerCase();
    var items = document.querySelectorAll('.faq-item');
    var categories = document.querySelectorAll('.faq-category');
    var visibleCount = 0;
    var visibleCategories = {};

    items.forEach(function(item) {
        if (!query) {
            item.classList.remove('hidden');
            // Restore original question text (remove highlights)
            var qEl = item.querySelector('.faq-question-text');
            var origQ = FAQ_DATA.reduce(function(found, sec) {
                if (found) return found;
                var match = sec.items.find(function(it) { return it.q.toLowerCase() === item.dataset.question; });
                return match ? match.q : null;
            }, null);
            if (origQ) qEl.innerHTML = origQ;
            visibleCount++;
            visibleCategories[item.dataset.category] = true;
            return;
        }

        var inQ = item.dataset.question.indexOf(query) !== -1;
        var inA = item.dataset.answer.indexOf(query) !== -1;

        if (inQ || inA) {
            item.classList.remove('hidden');
            item.classList.add('open'); // auto-expand matches
            visibleCount++;
            visibleCategories[item.dataset.category] = true;

            // Highlight matching text in question
            var qEl = item.querySelector('.faq-question-text');
            var origQ = FAQ_DATA.reduce(function(found, sec) {
                if (found) return found;
                var match = sec.items.find(function(it) { return it.q.toLowerCase() === item.dataset.question; });
                return match ? match.q : null;
            }, null);
            if (origQ) {
                var regex = new RegExp('(' + escapeRegex(query) + ')', 'gi');
                qEl.innerHTML = origQ.replace(regex, '<span class="faq-highlight">$1</span>');
            }
        } else {
            item.classList.add('hidden');
            item.classList.remove('open');
        }
    });

    // Show/hide category headers based on whether they have visible items
    categories.forEach(function(cat) {
        cat.style.display = visibleCategories[cat.dataset.category] ? '' : 'none';
    });

    // Update result count
    var countEl = document.getElementById('faq-result-count');
    if (countEl) {
        if (query) {
            countEl.textContent = visibleCount + ' result' + (visibleCount !== 1 ? 's' : '');
        } else {
            countEl.textContent = '';
        }
    }

    // Show no-results message
    var container = document.getElementById('faq-content');
    var noRes = container.querySelector('.faq-no-results');
    if (visibleCount === 0 && query) {
        if (!noRes) {
            noRes = document.createElement('div');
            noRes.className = 'faq-no-results';
            container.appendChild(noRes);
        }
        noRes.textContent = 'No questions match "' + query + '"';
        noRes.style.display = '';
    } else if (noRes) {
        noRes.style.display = 'none';
    }
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Wire up search input
(function() {
    var searchInput = document.getElementById('faq-search');
    var clearBtn = document.getElementById('faq-search-clear');
    if (!searchInput) return;

    var debounceTimer = null;
    searchInput.addEventListener('input', function() {
        clearTimeout(debounceTimer);
        var val = searchInput.value;
        if (clearBtn) clearBtn.style.display = val ? '' : 'none';
        debounceTimer = setTimeout(function() { filterFaq(val); }, 150);
    });

    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            searchInput.value = '';
            clearBtn.style.display = 'none';
            filterFaq('');
            searchInput.focus();
        });
    }
})();

// Build FAQ on load
buildFaqItems();
