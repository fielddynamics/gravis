/**
 * GRAVIS - GRAvity VISion
 * Frontend JavaScript: fetches rotation curve data from Flask API,
 * renders charts with Chart.js.
 *
 * All physics computations happen server-side via /api/rotation/curve
 * and /api/rotation/infer-mass-model. The frontend handles only:
 *   - UI state management (sliders, dropdowns, mode toggle)
 *   - Chart rendering (Chart.js with error bar plugin)
 *   - API communication
 */

// =====================================================================
// PHYSICAL CONSTANTS (mirrors physics/constants.py)
// =====================================================================
const PHYS = {
    G: 6.67430e-11,        // m^3 kg^-1 s^-2  (CODATA 2022)
    KPC_TO_M: 3.0857e19,   // meters per kpc
    A0: 16 * 6.67430e-11 * 9.1093837139e-31 / (2.8179403205e-15 * 2.8179403205e-15)
    // A0 = k^2 * G * m_e / r_e^2 ~ 1.22e-10 m/s^2
};
const THROAT_FRAC = 0.30;  // R_t / R_env emergent ratio (mirrors physics.constants.THROAT_FRAC)

// =====================================================================
// STATE
// =====================================================================
let currentMode = 'prediction';
let currentExample = null;
let pinnedObservations = null;   // observations stay visible when sliders are tweaked
let pinnedGalaxyLabel = null;    // galaxy name stays in chart title when pinned
let pinnedGalaxyExample = null;  // full example ref for Data Sources card
let isLoadingExample = false;
let galaxyCatalog = { prediction: [], inference: [] };
let isAutoFitted = false;        // true when in Observation mode (GFD derived from data)
let lastCdmHalo = null;
let lastApiResponse = null;      // last API response from updateChart()
let inferredFieldGeometry = null; // field geometry from topological prediction (after inference)
let sandboxResult = null;        // cached response from sandbox API (photometric + sigma curves)
let photometricResult = null;    // cached response from fast photometric API (includes spline)
let massModelManuallyModified = false;  // true when user changed mass sliders from photometric (enables Reset, manual GFD)

/**
 * Get the galactic radius (gravitational horizon) from the slider.
 * Returns null only when no galaxy is loaded and slider is at 0.
 */
function getGalacticRadius() {
    var val = parseFloat(galacticRadiusSlider.value);
    return (val && val > 0) ? val : null;
}
let lastMultiResult = null;     // cached multi-point inference result
let lastModelTotal = 0;         // cached anchor model total mass
let lastAccelRatio = 1.0;       // cached accel ratio for band recompute
let lastMassModel = null;       // cached mass model for band recompute

// Debounce timer for API calls
let updateTimer = null;
const DEBOUNCE_MS = 80;
let manualGfdTimer = null;
const MANUAL_GFD_DEBOUNCE_MS = 150;

// =====================================================================
// DOM ELEMENTS
// =====================================================================
const distanceSlider = document.getElementById('distance-slider');
const distanceLabel = document.getElementById('distance-label');
const anchorRadiusInput = document.getElementById('anchor-radius');
const massSlider = document.getElementById('mass-slider');
const velocitySlider = document.getElementById('velocity-slider');
const accelSlider = document.getElementById('accel-slider');
const galacticRadiusSlider = document.getElementById('galactic-radius-slider');
const galacticRadiusValue = document.getElementById('galactic-radius-value');
const lensSlider = document.getElementById('lens-slider');
const lensValue = document.getElementById('lens-value');
const vortexStrengthSlider = document.getElementById('vortex-strength-slider');
const vortexStrengthValue = document.getElementById('vortex-strength-value');
const vortexAutoBtn = document.getElementById('vortex-auto-btn');
let autoVortexStrength = null;  // last auto-calculated value from API
let isAutoThroughput = true;    // when true, backend computes the value
let inferenceNeeded = false;    // true only on initial galaxy load in inference mode
const distanceValue = document.getElementById('distance-value');
const massValue = document.getElementById('mass-value');
const velocityValue = document.getElementById('velocity-value');
const accelValue = document.getElementById('accel-value');
const velocityControl = document.getElementById('velocity-control');
const inferenceResult = document.getElementById('inference-result');
const inferredMassValue = document.getElementById('inferred-mass-value');
const observedLegend = document.getElementById('observed-legend');
const massLabel = document.getElementById('mass-label');

// Mass model panel
const massModelContent = document.getElementById('mass-model-content');
const bulgeMassSlider = document.getElementById('bulge-mass-slider');
const bulgeScaleSlider = document.getElementById('bulge-scale-slider');
const diskMassSlider = document.getElementById('disk-mass-slider');
const diskScaleSlider = document.getElementById('disk-scale-slider');
const gasMassSlider = document.getElementById('gas-mass-slider');
const gasScaleSlider = document.getElementById('gas-scale-slider');

const massModelActive = true;

// =====================================================================
// HELPERS
// =====================================================================

function formatMassCompact(logMass) {
    const val = Math.pow(10, logMass);
    const exp = Math.floor(Math.log10(val));
    const coeff = val / Math.pow(10, exp);
    return coeff.toFixed(3) + 'e' + exp;
}

function superscript(num) {
    const superscripts = {
        '0': '\u2070', '1': '\u00B9', '2': '\u00B2', '3': '\u00B3', '4': '\u2074',
        '5': '\u2075', '6': '\u2076', '7': '\u2077', '8': '\u2078', '9': '\u2079'
    };
    return String(num).split('').map(d => superscripts[d] || d).join('');
}

// =====================================================================
// MASS MODEL SLIDERS
// =====================================================================

function getMassModelFromSliders() {
    return {
        bulge: {
            M: Math.pow(10, parseFloat(bulgeMassSlider.value)),
            a: parseFloat(bulgeScaleSlider.value)
        },
        disk: {
            M: Math.pow(10, parseFloat(diskMassSlider.value)),
            Rd: parseFloat(diskScaleSlider.value)
        },
        gas: {
            M: Math.pow(10, parseFloat(gasMassSlider.value)),
            Rd: parseFloat(gasScaleSlider.value)
        }
    };
}

function setMassModelSliders(mm) {
    if (mm.bulge) {
        bulgeMassSlider.value = Math.log10(Math.max(mm.bulge.M, 1e7));
        bulgeScaleSlider.value = mm.bulge.a;
    }
    if (mm.disk) {
        diskMassSlider.value = Math.log10(Math.max(mm.disk.M, 1e7));
        diskScaleSlider.value = mm.disk.Rd;
    }
    if (mm.gas) {
        gasMassSlider.value = Math.log10(Math.max(mm.gas.M, 1e7));
        gasScaleSlider.value = mm.gas.Rd;
    }
    updateMassModelDisplays();
}

function updateMassModelDisplays() {
    const bm = Math.pow(10, parseFloat(bulgeMassSlider.value));
    const dm = Math.pow(10, parseFloat(diskMassSlider.value));
    const gm = Math.pow(10, parseFloat(gasMassSlider.value));
    const total = bm + dm + gm;

    document.getElementById('bulge-mass-value').textContent = formatMassCompact(parseFloat(bulgeMassSlider.value)) + ' M_sun';
    document.getElementById('bulge-scale-value').textContent = parseFloat(bulgeScaleSlider.value).toFixed(2) + ' kpc';
    document.getElementById('disk-mass-value').textContent = formatMassCompact(parseFloat(diskMassSlider.value)) + ' M_sun';
    document.getElementById('disk-scale-value').textContent = parseFloat(diskScaleSlider.value).toFixed(2) + ' kpc';
    document.getElementById('gas-mass-value').textContent = formatMassCompact(parseFloat(gasMassSlider.value)) + ' M_sun';
    document.getElementById('gas-scale-value').textContent = parseFloat(gasScaleSlider.value).toFixed(2) + ' kpc';

    const totalExp = Math.floor(Math.log10(total));
    const totalCoeff = total / Math.pow(10, totalExp);
    document.getElementById('mass-model-total-value').textContent =
        totalCoeff.toFixed(2) + 'e' + totalExp + ' M_sun';

    const logTotal = Math.log10(total);
    massSlider.value = logTotal;
}

// Update mass model display labels only (no side effects, for inference auto-scaling)
function updateResetToPhotometricButton() {
    var btn = document.getElementById('reset-to-photometric-btn');
    if (!btn) return;
    var inMassModelMode = !isAutoFitted;
    btn.style.display = (massModelManuallyModified && inMassModelMode && currentExample && currentExample.id) ? '' : 'none';
}

function resetToPhotometric() {
    if (!currentExample || !currentExample.id || !sandboxResult || !sandboxResult.photometric_mass_model) return;
    setMassModelSliders(sandboxResult.photometric_mass_model);
    massModelManuallyModified = false;
    updateResetToPhotometricButton();
    fetchPhotometricData(currentExample.id).then(function() {
        var massModel = getMassModelFromSliders();
        var predObs = pinnedObservations || [];
        var predMaxR = parseFloat(distanceSlider.value);
        if (predObs.length > 0) {
            var maxObsR = Math.max.apply(null, predObs.map(function(o) { return o.r; }));
            predMaxR = Math.max(predMaxR, maxObsR * 1.15);
        }
        var gr = currentExample.galactic_radius ? parseFloat(currentExample.galactic_radius) : parseFloat(distanceSlider.value);
        fetchRotationCurve(predMaxR, parseFloat(accelSlider.value), massModel, predObs, gr).then(function(data) {
            var i, newtonianData = [], mondData = [], cdmData = [];
            for (i = 0; i < (data.radii || []).length; i++) {
                newtonianData.push({ x: data.radii[i], y: data.newtonian[i] });
                mondData.push({ x: data.radii[i], y: data.mond[i] });
                if (data.cdm) cdmData.push({ x: data.radii[i], y: data.cdm[i] });
            }
            chart.data.datasets[0].data = newtonianData;
            chart.data.datasets[2].data = mondData;
            chart.data.datasets[7].data = cdmData;
            chart.data.datasets[0].hidden = !isChipEnabled('newtonian');
            chart.data.datasets[2].hidden = !isChipEnabled('mond');
            chart.data.datasets[7].hidden = !isChipEnabled('cdm');
            if (data.cdm_halo) lastCdmHalo = data.cdm_halo;
            chart.update('none');
        }).catch(function() {});
    });
}

function updateMassModelDisplaysOnly() {
    const bm = Math.pow(10, parseFloat(bulgeMassSlider.value));
    const dm = Math.pow(10, parseFloat(diskMassSlider.value));
    const gm = Math.pow(10, parseFloat(gasMassSlider.value));
    const total = bm + dm + gm;

    document.getElementById('bulge-mass-value').textContent = formatMassCompact(parseFloat(bulgeMassSlider.value)) + ' M_sun';
    document.getElementById('bulge-scale-value').textContent = parseFloat(bulgeScaleSlider.value).toFixed(2) + ' kpc';
    document.getElementById('disk-mass-value').textContent = formatMassCompact(parseFloat(diskMassSlider.value)) + ' M_sun';
    document.getElementById('disk-scale-value').textContent = parseFloat(diskScaleSlider.value).toFixed(2) + ' kpc';
    document.getElementById('gas-mass-value').textContent = formatMassCompact(parseFloat(gasMassSlider.value)) + ' M_sun';
    document.getElementById('gas-scale-value').textContent = parseFloat(gasScaleSlider.value).toFixed(2) + ' kpc';

    const totalExp = Math.floor(Math.log10(total));
    const totalCoeff = total / Math.pow(10, totalExp);
    document.getElementById('mass-model-total-value').textContent =
        totalCoeff.toFixed(2) + 'e' + totalExp + ' M_sun';
}

// Mass model slider listeners (always active)
// In prediction mode: all sliders directly control the model
// In inference mode: scale length sliders control shape, masses are auto-computed
[bulgeMassSlider, bulgeScaleSlider, diskMassSlider, diskScaleSlider, gasMassSlider, gasScaleSlider].forEach(slider => {
    slider.addEventListener('input', () => {
        if (!isLoadingExample && currentExample && currentExample.id) {
            massModelManuallyModified = true;
            updateResetToPhotometricButton();
            clearTimeout(manualGfdTimer);
            manualGfdTimer = setTimeout(function() {
                var mm = getMassModelFromSliders();
                var maxR = parseFloat(distanceSlider.value);
                if (pinnedObservations && pinnedObservations.length > 0) {
                    var maxObsR = Math.max.apply(null, pinnedObservations.map(function(o) { return o.r; }));
                    maxR = Math.max(maxR, maxObsR * 1.15);
                }
                fetchGfdFromMassModel(mm, maxR, parseFloat(accelSlider.value));
            }, MANUAL_GFD_DEBOUNCE_MS);
        }
        if (isAutoFitted) {
            navigateTo('charts', 'chart');
            hideAutoMapDiagnostics();
        }
        updateMassModelDisplays();
        debouncedUpdateChart();
    });
});

// Galactic radius slider listener: updates the display and triggers
// chart recompute so the GFD-sigma structural term responds in real time.
galacticRadiusSlider.addEventListener('input', () => {
    galacticRadiusValue.textContent = galacticRadiusSlider.value + ' kpc';
    debouncedUpdateChart();
});

// Gravity Lens Throughput slider: controls the +/- band width around
// the GFD curve. Default 6.2% from (4/pi)^(1/4) topological constraint.
lensSlider.addEventListener('input', () => {
    var pct = parseFloat(lensSlider.value);
    lensValue.textContent = '+/- ' + pct.toFixed(1) + '%';
    updateLensBand();
});

// Origin Throughput slider listener: modulates the GFD-sigma structural
// correction via vortex reflection through the Field Origin.
vortexStrengthSlider.addEventListener('input', () => {
    vortexStrengthValue.textContent = parseFloat(vortexStrengthSlider.value).toFixed(2);
    // Mark as manual override
    isAutoThroughput = false;
    vortexAutoBtn.classList.remove('active');
    debouncedUpdateChart();
});

// Auto button: switch back to auto mode (triggers re-fetch which
// will run GA when observations are available)
vortexAutoBtn.addEventListener('click', () => {
    isAutoThroughput = true;
    inferenceNeeded = false;
    vortexAutoBtn.classList.add('active');
    if (autoVortexStrength !== null) {
        vortexStrengthSlider.value = autoVortexStrength;
        vortexStrengthValue.textContent = 'auto';
    } else {
        vortexStrengthValue.textContent = 'auto';
    }
    debouncedUpdateChart();
});

/**
 * Sync the Origin Throughput slider and mass displays from an API
 * response. When the two-stage GA ran, updates the throughput slider
 * with the fitted value and adjusts mass sliders to reflect the
 * GA-optimized mass scale.
 */
function syncThroughputFromResponse(data) {
    if (data.auto_origin_throughput !== undefined) {
        autoVortexStrength = data.auto_origin_throughput;
        if (isAutoThroughput) {
            vortexStrengthSlider.value = autoVortexStrength;
            var label = autoVortexStrength.toFixed(2);
            if (data.throughput_fit && data.throughput_fit.method) {
                var m = data.throughput_fit.method;
                if (m === 'inference') {
                    label += ' (origin inferred)';
                } else if (m === 'two_stage_ga') {
                    label += ' (fit)';
                }
            }
            vortexStrengthValue.textContent = label;
        }
    }
    // When inference optimized the mass model, sync all sliders
    if (data.optimized_mass_model && isAutoThroughput) {
        var om = data.optimized_mass_model;
        if (om.bulge && om.bulge.M > 0) {
            bulgeMassSlider.value = Math.log10(Math.max(om.bulge.M, 1e7));
            if (om.bulge.a > 0) bulgeScaleSlider.value = om.bulge.a;
        }
        if (om.disk && om.disk.M > 0) {
            diskMassSlider.value = Math.log10(Math.max(om.disk.M, 1e7));
            if (om.disk.Rd > 0) diskScaleSlider.value = om.disk.Rd;
        }
        if (om.gas && om.gas.M > 0) {
            gasMassSlider.value = Math.log10(Math.max(om.gas.M, 1e7));
            if (om.gas.Rd > 0) gasScaleSlider.value = om.gas.Rd;
        }
        updateMassModelDisplaysOnly();
    }
}

// =====================================================================
// ERROR BAR PLUGIN
// =====================================================================

const errorBarPlugin = {
    id: 'errorBars',
    afterDatasetsDraw(chart) {
        const dataset = chart.data.datasets[3];
        if (!dataset || dataset.hidden || !dataset.data || dataset.data.length === 0) return;
        const meta = chart.getDatasetMeta(3);
        if (!meta || meta.hidden) return;
        const ctx = chart.ctx;
        ctx.save();
        ctx.strokeStyle = '#FFC107';
        ctx.lineWidth = 1.5;
        dataset.data.forEach((point, index) => {
            const element = meta.data[index];
            if (!element) return;
            const x = element.x;
            const yScale = chart.scales.y;
            const yCenter = yScale.getPixelForValue(point.y);
            // Draw error whiskers if error is present
            if (point.err) {
                const yTop = yScale.getPixelForValue(point.y + point.err);
                const yBottom = yScale.getPixelForValue(point.y - point.err);
                ctx.beginPath(); ctx.moveTo(x, yTop); ctx.lineTo(x, yBottom); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(x - 4, yTop); ctx.lineTo(x + 4, yTop); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(x - 4, yBottom); ctx.lineTo(x + 4, yBottom); ctx.stroke();
            }
            // Always draw a filled center dot so it is visible on top of curves
            ctx.fillStyle = '#FFC107';
            ctx.beginPath();
            ctx.arc(x, yCenter, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
        ctx.restore();
    }
};
Chart.register(errorBarPlugin);

// =====================================================================
// FIELD ORIGIN BOUNDARY PLUGIN
// =====================================================================
// Draws a vertical dashed line at the throat radius R_t = 0.30 * R_env
// to mark where the stellated field origin boundary ends.

var fieldOriginBoundaryPlugin = {
    id: 'fieldOriginBoundary',
    afterDraw: function(chartInstance) {
        return;  // R_t line hidden for now (off observational band graph)

        var rThroat;
        var label;
        var fg = sandboxResult ? sandboxResult.field_geometry : null;
        if (fg && fg.throat_radius_kpc != null) {
            rThroat = fg.throat_radius_kpc;
            label = 'R_t = ' + rThroat.toFixed(1) + ' kpc';
        } else if (inferredFieldGeometry
            && inferredFieldGeometry.throat_radius_kpc != null) {
            rThroat = inferredFieldGeometry.throat_radius_kpc;
            label = 'R_t = ' + rThroat.toFixed(1) + ' kpc';
        } else {
            var rEnv = getGalacticRadius();
            if (!rEnv) return;
            rThroat = THROAT_FRAC * rEnv;
            label = 'R_t = ' + rThroat.toFixed(1) + ' kpc';
        }

        var xScale = chartInstance.scales.x;
        var yAxis = chartInstance.scales.y;
        if (!xScale || !yAxis) return;
        if (rThroat < xScale.min || rThroat > xScale.max) return;

        var xPixel = xScale.getPixelForValue(rThroat);
        var ctx = chartInstance.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([6, 4]);
        ctx.moveTo(xPixel, yAxis.top);
        ctx.lineTo(xPixel, yAxis.bottom);
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#00ff88';
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.font = '10px Inter, "Segoe UI", system-ui, sans-serif';
        ctx.fillStyle = '#00ff88';
        ctx.textAlign = 'center';
        ctx.fillText(label, xPixel, yAxis.top - 6);
        ctx.restore();
    }
};
Chart.register(fieldOriginBoundaryPlugin);

// =====================================================================
// FIELD HORIZON PLUGIN
// =====================================================================
// Draws a vertical dashed line at R_env (the galactic radius / baryonic
// horizon) in red to mark the full extent of the baryonic envelope.

var fieldHorizonPlugin = {
    id: 'fieldHorizon',
    afterDraw: function(chartInstance) {
        return;  // R_env line hidden for now (off chart in both mass model and observational)
        var rEnv;
        var label;
        var fg = sandboxResult ? sandboxResult.field_geometry : null;
        if (fg && fg.envelope_radius_kpc != null) {
            rEnv = fg.envelope_radius_kpc;
            label = 'R_env = ' + rEnv.toFixed(1) + ' kpc';
        } else if (isAutoFitted && inferredFieldGeometry
            && inferredFieldGeometry.envelope_radius_kpc != null) {
            rEnv = inferredFieldGeometry.envelope_radius_kpc;
            label = 'R_env = ' + rEnv.toFixed(1) + ' kpc';
        } else {
            rEnv = getGalacticRadius();
            if (!rEnv) return;
            label = 'R_env = ' + rEnv.toFixed(1) + ' kpc';
        }

        var xScale = chartInstance.scales.x;
        var yAxis = chartInstance.scales.y;
        if (!xScale || !yAxis) return;
        if (rEnv < xScale.min || rEnv > xScale.max) return;

        var xPixel = xScale.getPixelForValue(rEnv);
        var ctx = chartInstance.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([6, 4]);
        ctx.moveTo(xPixel, yAxis.top);
        ctx.lineTo(xPixel, yAxis.bottom);
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#ff6688';
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.font = '10px Inter, "Segoe UI", system-ui, sans-serif';
        ctx.fillStyle = '#ff6688';
        ctx.textAlign = 'center';
        ctx.fillText(label, xPixel, yAxis.top - 6);
        ctx.restore();
    }
};
Chart.register(fieldHorizonPlugin);

// =====================================================================
// SPARC R_HI PLUGIN
// =====================================================================
// Draws a vertical dotted line at the SPARC catalog R_HI (galactic_radius)
// to mark the observed HI extent from 21cm radio surveys.

var sparcRhiPlugin = {
    id: 'sparcRhi',
    afterDraw: function(chartInstance) {
        var rHi = sandboxResult ? sandboxResult.sparc_r_hi_kpc : null;
        if (!rHi || rHi <= 0) return;

        var xScale = chartInstance.scales.x;
        var yAxis = chartInstance.scales.y;
        if (!xScale || !yAxis) return;
        if (rHi < xScale.min || rHi > xScale.max) return;

        var xPixel = xScale.getPixelForValue(rHi);
        var ctx = chartInstance.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([3, 3]);
        ctx.moveTo(xPixel, yAxis.top);
        ctx.lineTo(xPixel, yAxis.bottom);
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = '#ffaa44';
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.font = '10px Inter, "Segoe UI", system-ui, sans-serif';
        ctx.fillStyle = '#ffaa44';
        ctx.textAlign = 'center';
        ctx.fillText('R_HI = ' + rHi.toFixed(1) + ' kpc', xPixel, yAxis.top - 6);
        ctx.restore();
    }
};
Chart.register(sparcRhiPlugin);

// =====================================================================
// R_VIS BAND PLUGIN
// =====================================================================
// Draws a shaded vertical band showing the baryonic extent (90% to
// 99.5% enclosed mass) from the sandbox field geometry.

var rVisBandPlugin = {
    id: 'rVisBand',
    beforeDraw: function(chartInstance) {
        return;  // Purple baryonic band hidden for now (off observational band graph)
        var fg = sandboxResult ? sandboxResult.field_geometry : null;
        if (!fg) return;
        var r90 = fg.visible_radius_90_kpc || 0;
        var r99 = fg.visible_radius_99_kpc || 0;
        if (r90 <= 0 || r99 <= 0) return;

        var xScale = chartInstance.scales.x;
        var yAxis = chartInstance.scales.y;
        if (!xScale || !yAxis) return;

        var x1 = xScale.getPixelForValue(r90);
        var x2 = xScale.getPixelForValue(r99);
        if (x2 < xScale.left || x1 > xScale.right) return;
        var left = Math.max(x1, xScale.left);
        var right = Math.min(x2, xScale.right);

        var ctx = chartInstance.ctx;
        ctx.save();
        ctx.fillStyle = 'rgba(170,136,255,0.12)';
        ctx.fillRect(left, yAxis.top, right - left, yAxis.bottom - yAxis.top);
        ctx.fillStyle = '#aa88ff';
        ctx.font = '10px Inter, "Segoe UI", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(
            'Baryonic extent (' + r90.toFixed(0) + ' to ' + r99.toFixed(0) + ' kpc)',
            (left + right) / 2,
            yAxis.top - 6);
        ctx.restore();
    }
};
Chart.register(rVisBandPlugin);

// =====================================================================

// ---- External HTML tooltip handler ----
function getOrCreateTooltipEl(chartInstance) {
    var el = document.getElementById('gravis-tooltip');
    if (!el) {
        el = document.createElement('div');
        el.id = 'gravis-tooltip';
        el.style.cssText = 'position:absolute;pointer-events:none;z-index:9999;'
            + 'opacity:0;transition:opacity 0.15s ease, left 0.1s ease, top 0.1s ease;'
            + 'font-family:Inter,"Segoe UI",system-ui,sans-serif;font-size:12px;'
            + 'background:rgba(18,18,22,0.96);color:#d0d0d0;'
            + 'border:1px solid #333;border-radius:8px;'
            + 'box-shadow:0 4px 20px rgba(0,0,0,0.5);'
            + 'min-width:200px;max-width:340px;';
        document.body.appendChild(el);
    }
    return el;
}

function ttRow(label, value, color) {
    var c = color || '#d0d0d0';
    return '<tr>'
        + '<td style="padding:2px 10px 2px 0;color:' + c + ';white-space:nowrap;">' + label + '</td>'
        + '<td style="padding:2px 0;color:#e8e8e8;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;">' + value + '</td>'
        + '</tr>';
}

function ttSection(title) {
    return '<div style="padding:7px 12px 4px;font-size:11px;font-weight:600;color:#808080;'
        + 'text-transform:uppercase;letter-spacing:0.6px;">' + title + '</div>'
        + '<hr style="margin:0 12px;border:none;border-top:1px solid #333;">';
}

function ttMass(val) {
    if (!val || val <= 0) return '---';
    var e = Math.floor(Math.log10(val));
    var c = val / Math.pow(10, e);
    return c.toFixed(2) + ' &times; 10<sup>' + e + '</sup> M<sub>\u2609</sub>';
}

function buildInferenceTooltip(dp) {
    var m = dp.raw.meta;
    var r = dp.parsed.x;
    var gfdV = dp.parsed.y;
    var html = '';

    // Header
    html += '<div style="padding:8px 12px 4px;font-size:13px;font-weight:700;color:#e0e0e0;">';
    html += 'r = ' + r.toFixed(1) + ' kpc';
    html += '</div>';
    html += '<hr style="margin:0 12px;border:none;border-top:1px solid #444;">';

    // Section: Velocity comparison
    html += ttSection('Velocity Comparison');
    html += '<table style="width:100%;padding:4px 12px 6px;border-spacing:0;">';
    html += ttRow('<span style="color:#4da6ff;">\u25CF</span> GFD', gfdV.toFixed(1) + ' km/s', '#4da6ff');
    if (m.obs_v !== undefined) {
        var errStr = (m.err && m.err > 0) ? ' \u00B1 ' + m.err : '';
        html += ttRow('<span style="color:#ffa726;">\u25CF</span> Observed', m.obs_v.toFixed(1) + errStr + ' km/s', '#ffa726');
    }
    var newtonV = interpolateCurve(0, r);
    var mondV = interpolateCurve(2, r);
    var cdmV = interpolateCurve(7, r);
    var gfdSymV2 = interpolateCurve(9, r);
    if (gfdSymV2 !== null && chart.data.datasets[9].data.length > 0) {
        html += ttRow('<span style="color:#00E5FF;">\u25CF</span> GFD (Observed)', gfdSymV2.toFixed(1) + ' km/s', '#00E5FF');
    }
    if (newtonV !== null) {
        html += ttRow('<span style="color:#ef5350;">\u25CF</span> Newton', newtonV.toFixed(1) + ' km/s', '#ef5350');
    }
    if (mondV !== null) {
        html += ttRow('<span style="color:#ab47bc;">\u25CF</span> MOND', mondV.toFixed(1) + ' km/s', '#ab47bc');
    }
    if (cdmV !== null && chart.data.datasets[7].data.length > 0) {
        html += ttRow('<span style="color:#ffffff;">\u25CF</span> CDM+NFW', cdmV.toFixed(1) + ' km/s', '#ffffff');
    }
    html += '</table>';

    // Section: Residual
    if (m.obs_v !== undefined) {
        html += ttSection('Residual');
        var delta_v = gfdV - m.obs_v;
        var resStr = (delta_v >= 0 ? '+' : '') + delta_v.toFixed(1) + ' km/s';
        html += '<div style="padding:4px 12px 6px;">';
        html += '<span style="color:#e0e0e0;">GFD &minus; Obs: </span>';
        html += '<span style="font-weight:600;color:#e0e0e0;">' + resStr + '</span>';
        if (m.sigma_away !== null) {
            var nSig = m.sigma_away.toFixed(1);
            var sigColor = m.sigma_away < 1.0 ? '#4caf50'
                         : m.sigma_away < 2.0 ? '#8bc34a'
                         : m.sigma_away < 3.0 ? '#ffa726'
                         : '#ef5350';
            var sigLabel = m.sigma_away < 1.0 ? 'within 1\u03C3'
                         : m.sigma_away < 2.0 ? 'within 2\u03C3'
                         : m.sigma_away < 3.0 ? '2-3\u03C3'
                         : '> 3\u03C3';
            html += '<br><span style="font-size:11px;color:' + sigColor + ';font-weight:600;">'
                + nSig + '\u03C3 &mdash; ' + sigLabel + '</span>';
        }
        html += '</div>';
    }

    // Section: Inferred Mass
    html += ttSection('Inferred Mass');
    html += '<table style="width:100%;padding:4px 12px 8px;border-spacing:0;">';
    html += ttRow('From obs', ttMass(m.inferred_total), '#e0e0e0');
    if (m.gfd_total && m.gfd_total > 0) {
        html += ttRow('GFD predicts', ttMass(m.gfd_total), '#4da6ff');
    }
    var devColor = Math.abs(m.deviation) < 10 ? '#4caf50' : Math.abs(m.deviation) < 25 ? '#ffa726' : '#ef5350';
    html += ttRow('Deviation', '<span style="color:' + devColor + ';">' + (m.deviation >= 0 ? '+' : '') + m.deviation.toFixed(1) + '%</span>', '#808080');
    html += ttRow('Mass enclosed', (m.enclosed_frac * 100).toFixed(0) + '% of model', '#808080');
    html += '</table>';

    return html;
}

function buildCdmTooltip(dp) {
    if (!lastCdmHalo) return '';
    var html = '';
    html += '<div style="padding:8px 12px 4px;font-size:13px;font-weight:700;color:#e0e0e0;">';
    html += '<span style="color:#ffffff;">\u25CF</span> CDM + NFW Halo';
    html += '</div>';
    html += '<hr style="margin:0 12px;border:none;border-top:1px solid #444;">';
    html += '<table style="width:100%;padding:6px 12px 8px;border-spacing:0;">';
    html += ttRow('Velocity', dp.parsed.y.toFixed(1) + ' km/s', '#e0e0e0');
    var m200 = lastCdmHalo.m200;
    html += ttRow('Halo M<sub>200</sub>', ttMass(m200), '#e0e0e0');
    html += ttRow('Concentration', 'c = ' + lastCdmHalo.c, '#e0e0e0');
    if (lastCdmHalo.chi2_reduced !== undefined) {
        html += ttRow('Reduced \u03C7\u00B2', '' + lastCdmHalo.chi2_reduced, '#e0e0e0');
    }
    html += ttRow('Fitted params', '' + (lastCdmHalo.n_params_fitted || 0), '#e0e0e0');
    html += '</table>';
    return html;
}

function buildObservedTooltip(dp) {
    var r = dp.parsed.x;
    var vObs = dp.parsed.y;
    var err = (dp.raw && dp.raw.err) ? dp.raw.err : 0;
    var html = '';

    // Header
    html += '<div style="padding:8px 12px 4px;font-size:13px;font-weight:700;color:#e0e0e0;">';
    html += 'r = ' + r.toFixed(1) + ' kpc';
    html += '</div>';
    html += '<hr style="margin:0 12px;border:none;border-top:1px solid #444;">';

    // Section 1: Velocity comparison (with deltas from observed)
    html += ttSection('Velocity Comparison');
    var errStr = (err > 0) ? ' \u00B1 ' + err : '';

    var gfdV = interpolateCurve(1, r);
    var newtonV = interpolateCurve(0, r);
    var mondV = interpolateCurve(2, r);
    var cdmV = interpolateCurve(7, r);

    // Helper: format a delta value with color coding
    function fmtDelta(theoryV) {
        var d = theoryV - vObs;
        var pct = (d / vObs) * 100;
        var sign = d >= 0 ? '+' : '';
        var dColor = Math.abs(pct) < 5 ? '#4caf50'
                   : Math.abs(pct) < 15 ? '#ffa726'
                   : '#ef5350';
        return '<span style="color:' + dColor + ';font-size:11px;"> ' + sign + pct.toFixed(1) + '%</span>';
    }

    html += '<table style="width:100%;padding:4px 12px 6px;border-spacing:0;">';
    html += ttRow('<span style="color:#ffa726;">\u25CF</span> Observed', vObs.toFixed(1) + errStr + ' km/s', '#ffa726');
    if (gfdV !== null)    html += ttRow('<span style="color:#4da6ff;">\u25CF</span> GFD', gfdV.toFixed(1) + ' km/s' + fmtDelta(gfdV), '#4da6ff');
    var gfdSymV = interpolateCurve(9, r);
    if (gfdSymV !== null && chart.data.datasets[9].data.length > 0) {
        html += ttRow('<span style="color:#00E5FF;">\u25CF</span> GFD (Observed)', gfdSymV.toFixed(1) + ' km/s' + fmtDelta(gfdSymV), '#00E5FF');
    }
    if (newtonV !== null) html += ttRow('<span style="color:#ef5350;">\u25CF</span> Newton', newtonV.toFixed(1) + ' km/s' + fmtDelta(newtonV), '#ef5350');
    if (mondV !== null)   html += ttRow('<span style="color:#ab47bc;">\u25CF</span> MOND', mondV.toFixed(1) + ' km/s' + fmtDelta(mondV), '#ab47bc');
    if (cdmV !== null && chart.data.datasets[7].data.length > 0) {
        html += ttRow('<span style="color:#ffffff;">\u25CF</span> CDM+NFW', cdmV.toFixed(1) + ' km/s' + fmtDelta(cdmV), '#ffffff');
    }
    html += '</table>';

    // Section 2: GFD Agreement (sigma detail)
    if (gfdV !== null && err > 0) {
        var delta = gfdV - vObs;
        var sigAway = Math.abs(delta) / err;
        var sigColor = sigAway < 1.0 ? '#4caf50'
                     : sigAway < 2.0 ? '#8bc34a'
                     : sigAway < 3.0 ? '#ffa726'
                     : '#ef5350';
        var sigLabel = sigAway < 1.0 ? 'within 1\u03C3'
                     : sigAway < 2.0 ? 'within 2\u03C3'
                     : sigAway < 3.0 ? '2-3\u03C3'
                     : '> 3\u03C3';
        html += ttSection('GFD Agreement');
        html += '<div style="padding:4px 12px 6px;">';
        html += '<span style="color:#e0e0e0;">GFD &minus; Obs: </span>';
        html += '<span style="font-weight:600;color:#e0e0e0;">' + (delta >= 0 ? '+' : '') + delta.toFixed(1) + ' km/s</span>';
        html += '<br><span style="font-size:11px;color:' + sigColor + ';font-weight:600;">'
            + sigAway.toFixed(1) + '\u03C3 &mdash; ' + sigLabel + '</span>';
        html += '</div>';
    }

    // Section 4: Acceleration regime
    // g_eff = v^2 / r, x = g_eff / a0
    var v_ms = vObs * 1000;
    var r_m = r * PHYS.KPC_TO_M;
    if (r_m > 0) {
        var g_eff = (v_ms * v_ms) / r_m;
        var x = g_eff / PHYS.A0;
        var regimeColor, regimeLabel;
        if (x > 10) {
            regimeColor = '#ef5350'; regimeLabel = 'Newtonian regime';
        } else if (x > 1) {
            regimeColor = '#ffa726'; regimeLabel = 'Transition zone';
        } else if (x > 0.1) {
            regimeColor = '#8bc34a'; regimeLabel = 'Field dynamics regime';
        } else {
            regimeColor = '#4caf50'; regimeLabel = 'Deep field regime';
        }
        html += ttSection('Field Regime');
        html += '<div style="padding:4px 12px 6px;">';
        html += '<span style="color:#e0e0e0;">g / a</span><sub style="color:#e0e0e0;">0</sub>';
        html += '<span style="color:#e0e0e0;"> = </span>';
        html += '<span style="font-weight:600;color:#e0e0e0;">' + x.toFixed(2) + '</span>';
        html += '<br><span style="font-size:11px;color:' + regimeColor + ';font-weight:600;">' + regimeLabel + '</span>';
        html += '</div>';
    }

    // Section 5: Mass discrepancy (how much mass Newton needs vs baryonic model)
    if (newtonV !== null && newtonV > 0) {
        var massRatio = (vObs * vObs) / (newtonV * newtonV);
        html += ttSection('Mass Discrepancy');
        html += '<div style="padding:4px 12px 8px;">';
        html += '<span style="color:#e0e0e0;">M</span><sub style="color:#e0e0e0;">dynamic</sub>';
        html += '<span style="color:#e0e0e0;"> / M</span><sub style="color:#e0e0e0;">baryon</sub>';
        html += '<span style="color:#e0e0e0;"> = </span>';
        var mrColor = massRatio < 1.5 ? '#4caf50' : massRatio < 3 ? '#ffa726' : '#ef5350';
        html += '<span style="font-weight:600;color:' + mrColor + ';">' + massRatio.toFixed(1) + 'x</span>';
        if (massRatio > 1.5) {
            html += '<br><span style="font-size:11px;color:#808080;">Newton requires '
                + massRatio.toFixed(1) + 'x the visible mass</span>';
        } else {
            html += '<br><span style="font-size:11px;color:#808080;">Consistent with baryonic mass</span>';
        }
        html += '</div>';
    }

    return html;
}

function buildDefaultTooltip(dp) {
    var label = dp.dataset.label || '';
    var color = dp.dataset.borderColor || '#e0e0e0';
    var vStr = dp.parsed.y !== null ? dp.parsed.y.toFixed(1) + ' km/s' : '';
    return '<div style="padding:8px 12px;white-space:nowrap;">'
        + '<span style="color:' + color + ';font-weight:600;">' + label + '</span>'
        + '<span style="margin-left:10px;color:#e0e0e0;">' + vStr + '</span>'
        + '</div>';
}

function externalTooltipHandler(tooltipContext) {
    var tooltip = tooltipContext.tooltip;
    var el = getOrCreateTooltipEl(tooltipContext.chart);

    // Hide if no tooltip
    if (tooltip.opacity === 0) {
        el.style.opacity = '0';
        return;
    }

    var dp = tooltip.dataPoints && tooltip.dataPoints[0];
    if (!dp) { el.style.opacity = '0'; return; }

    // Suppress for confidence band datasets
    if (dp.datasetIndex === 4 || dp.datasetIndex === 5) {
        el.style.opacity = '0';
        return;
    }

    // Build HTML based on dataset
    var html = '';
    if (dp.datasetIndex === 6 && dp.raw && dp.raw.meta) {
        html = buildInferenceTooltip(dp);
    } else if (dp.datasetIndex === 3) {
        html = buildObservedTooltip(dp);
    } else if (dp.datasetIndex === 7 && lastCdmHalo) {
        html = buildCdmTooltip(dp);
    } else {
        html = buildDefaultTooltip(dp);
    }

    el.innerHTML = html;

    // Position: offset from caret, clamped to viewport
    var chartRect = tooltipContext.chart.canvas.getBoundingClientRect();
    var left = chartRect.left + window.scrollX + tooltip.caretX + 14;
    var top = chartRect.top + window.scrollY + tooltip.caretY - 20;

    // Clamp right edge
    var elWidth = el.offsetWidth || 280;
    if (left + elWidth > window.innerWidth - 10) {
        left = chartRect.left + window.scrollX + tooltip.caretX - elWidth - 14;
    }
    // Clamp bottom edge
    var elHeight = el.offsetHeight || 200;
    if (top + elHeight > window.innerHeight + window.scrollY - 10) {
        top = window.innerHeight + window.scrollY - elHeight - 10;
    }
    if (top < window.scrollY + 5) top = window.scrollY + 5;

    el.style.left = left + 'px';
    el.style.top = top + 'px';
    el.style.opacity = '1';
}

// Vertical crosshair line plugin: draws a thin line at the mouse x position
var crosshairLinePlugin = {
    id: 'crosshairLine',
    afterDraw: function(chartInstance) {
        if (!chartInstance._crosshairX) return;
        var ctx2 = chartInstance.ctx;
        var yAxis = chartInstance.scales.y;
        var x = chartInstance._crosshairX;
        ctx2.save();
        ctx2.beginPath();
        ctx2.moveTo(x, yAxis.top);
        ctx2.lineTo(x, yAxis.bottom);
        ctx2.lineWidth = 1;
        ctx2.strokeStyle = 'rgba(77, 166, 255, 0.3)';
        ctx2.stroke();
        ctx2.restore();
    }
};

const ctx = document.getElementById('gravityChart').getContext('2d');
const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Newtonian Gravity',
                data: [],
                borderColor: '#ff6b6b',
                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                borderWidth: 2.5,
                tension: 0.4,
                pointRadius: 0,
                hidden: true
            },
            {
                label: 'Gravity Field Dynamics (Dual Tetrad)',
                data: [],
                borderColor: '#4da6ff',
                backgroundColor: 'rgba(77, 166, 255, 0.1)',
                borderWidth: 2.5,
                tension: 0.4,
                pointRadius: 0
            },
            {
                label: 'Classical MOND (Empirical)',
                data: [],
                borderColor: '#9966ff',
                backgroundColor: 'rgba(153, 102, 255, 0.1)',
                borderWidth: 2.5,
                tension: 0.4,
                pointRadius: 0,
                hidden: true
            },
            {
                label: 'Observed Data',
                data: [],
                borderColor: '#FFC107',
                backgroundColor: '#FFC107',
                borderWidth: 2.5,
                pointRadius: 7,
                pointStyle: 'line',
                showLine: false
            },
            // Dataset 4: GFD/GFD-sigma envelope upper edge (hidden from legend)
            {
                label: 'GFD envelope upper',
                data: [],
                borderColor: 'rgba(77, 166, 255, 0.25)',
                backgroundColor: 'rgba(77, 166, 255, 0.08)',
                borderWidth: 1,
                borderDash: [4, 4],
                tension: 0.4,
                pointRadius: 0,
                fill: {target: 5, above: 'rgba(77, 166, 255, 0.08)', below: 'rgba(77, 166, 255, 0.08)'}
            },
            // Dataset 5: GFD/GFD-sigma envelope lower edge (hidden from legend)
            {
                label: 'GFD envelope lower',
                data: [],
                borderColor: 'rgba(77, 166, 255, 0.25)',
                backgroundColor: 'transparent',
                borderWidth: 1,
                borderDash: [4, 4],
                tension: 0.4,
                pointRadius: 0,
                fill: false
            },
            // Dataset 6: Inference markers (green diamonds)
            {
                label: 'GFD Auto Fit',
                data: [],
                borderColor: '#4caf50',
                backgroundColor: 'rgba(76, 175, 80, 0.7)',
                borderWidth: 2,
                pointRadius: 6,
                pointStyle: 'rectRot',
                showLine: false
            },
            // Dataset 7: CDM (baryonic + best-fit NFW halo)
            {
                label: '\u039BCDM + NFW Halo (Best Fit)',
                data: [],
                borderColor: '#ffffff',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                borderWidth: 2,
                borderDash: [8, 4],
                tension: 0.4,
                pointRadius: 0,
                hidden: true
            },
            // Dataset 8: GFD\u03C6 (covariant + gas-fraction-scaled structural release)
            {
                label: 'GFD\u03C6',
                data: [],
                borderColor: '#76FF03',
                backgroundColor: 'rgba(118, 255, 3, 0.08)',
                borderWidth: 2.5,
                tension: 0.4,
                pointRadius: 0
            },
            // Dataset 9: GFD (Observed) -- GFD mapped to observations via Origin Throughput.
            // Hidden by default; revealed only after auto-map completes.
            {
                label: 'GFD (Observed)',
                data: [],
                borderColor: '#00E5FF',
                backgroundColor: 'rgba(0, 229, 255, 0.08)',
                borderWidth: 2.5,
                tension: 0.4,
                pointRadius: 0,
                hidden: true
            },
            // Dataset 10: GFD (Observed) curve from observation fit
            {
                label: 'GFD (Observed)',
                data: [],
                borderColor: '#ff44dd',
                backgroundColor: 'transparent',
                borderWidth: 2.5,
                cubicInterpolationMode: 'monotone',
                tension: 0.4,
                pointRadius: 0,
                hidden: true
            },
            // Dataset 11: GFD (Observed)
            {
                label: 'GFD (Observed)',
                data: [],
                borderColor: '#aa44ff',
                backgroundColor: 'transparent',
                borderWidth: 2.5,
                cubicInterpolationMode: 'monotone',
                tension: 0.4,
                pointRadius: 0
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        layout: {
            padding: { top: 16 }
        },
        plugins: {
            legend: {
                // Legend replaced by the theory toggle bar above the chart.
                // We keep it hidden but retain the filter so internal
                // Chart.js operations that reference legend items still work.
                display: false
            },
            title: {
                display: true,
                text: 'Rotation Curve: Gravitational Theory Comparison',
                color: '#e0e0e0',
                font: { size: 16, weight: '500' },
                padding: 20
            },
            tooltip: {
                enabled: false,
                external: externalTooltipHandler
            },
            crosshairLine: {},
            zoom: {
                pan: {
                    enabled: true,
                    mode: 'x',
                    modifierKey: 'shift',
                    onPanComplete: ({chart: ch}) => {
                        const resetBtn = document.getElementById('reset-zoom-btn');
                        if (resetBtn) resetBtn.style.display = 'block';
                        syncVpsXAxis();
                    }
                },
                zoom: {
                    wheel: { enabled: true, speed: 0.1 },
                    pinch: { enabled: true },
                    drag: {
                        enabled: true,
                        backgroundColor: 'rgba(77, 166, 255, 0.2)',
                        borderColor: 'rgba(77, 166, 255, 0.8)',
                        borderWidth: 2
                    },
                    mode: 'x',
                    onZoomComplete: ({chart: ch}) => {
                        const resetBtn = document.getElementById('reset-zoom-btn');
                        if (resetBtn) resetBtn.style.display = 'block';
                        syncVpsXAxis();
                    }
                },
                limits: { x: {min: 0, max: 100} }
            }
        },
        scales: {
            x: {
                type: 'linear',
                title: {
                    display: true,
                    text: 'Galactocentric Radius (kpc)',
                    color: '#b0b0b0',
                    font: { size: 13, weight: '500' }
                },
                grid: { color: 'rgba(64, 64, 64, 0.3)' },
                ticks: { color: '#b0b0b0', maxTicksLimit: 10 }
            },
            y: {
                afterFit: function(scale) { scale.width = CHART_Y_AXIS_LEFT_WIDTH; },
                title: {
                    display: true,
                    text: 'Circular Velocity v(r) [km/s]',
                    color: '#b0b0b0',
                    font: { size: 13, weight: '500' }
                },
                grid: { color: 'rgba(64, 64, 64, 0.3)' },
                ticks: { color: '#b0b0b0' },
                beginAtZero: true
            }
        }
    }
});

// Register crosshair line plugin
Chart.register(crosshairLinePlugin);

// Track mouse position for the crosshair vertical line
var gravityCanvas = document.getElementById('gravityChart');
gravityCanvas.addEventListener('mousemove', function(e) {
    var rect = gravityCanvas.getBoundingClientRect();
    var pixelX = e.clientX - rect.left;
    var xScale = chart.scales.x;
    if (xScale && pixelX >= xScale.left && pixelX <= xScale.right) {
        chart._crosshairX = pixelX;
    } else {
        chart._crosshairX = null;
    }
    chart.draw();
});
gravityCanvas.addEventListener('mouseleave', function() {
    chart._crosshairX = null;
    chart.draw();
});

// =====================================================================
// ZOOM CONTROLS
// =====================================================================

const canvas = document.getElementById('gravityChart');
const resetZoomBtn = document.getElementById('reset-zoom-btn');

function hideResetButton() { resetZoomBtn.style.display = 'none'; }
function resetChartZoom() { chart.resetZoom(); hideResetButton(); }

resetZoomBtn.addEventListener('click', resetChartZoom);

// Fixed y-axis width for consistent layout.
var CHART_Y_AXIS_LEFT_WIDTH = 80;

// VPS panel: mirror of main chart + histogram drawn as line vertical segments (like observation error bars).
var vpsChart = null;
var vortexChartAInstance = null;
var vortexChartBInstance = null;
var vortexCurrentVariance = 1.5;
var lastVortexRotationData = null;
var lastVortexPhotometricData = null;

function getVpsDeltaVData() {
    var src = (photometricResult && photometricResult.chart) || (sandboxResult && sandboxResult.chart);
    if (!src || !src.radii) return [];
    var radii = src.radii;
    var accel = src.gfd_accel || [];
    var photo = src.gfd_photometric || [];
    var out = [];
    for (var i = 0; i < radii.length; i++) {
        var a = accel[i] != null ? accel[i] : 0;
        var p = photo[i] != null ? photo[i] : 0;
        out.push({ x: radii[i], y: a - p });
    }
    return out;
}

function getVpsHistogramLineData() {
    var points = getVpsDeltaVData();
    var boost = [];
    var suppress = [];
    for (var i = 0; i < points.length; i++) {
        if (i === 0) continue;
        var r = points[i].x;
        var y = points[i].y;
        var seg = [{ x: r, y: 0 }, { x: r, y: y }, { x: r, y: 0 }];
        if (y >= 0) {
            boost.push(seg[0], seg[1], seg[2]);
        } else {
            suppress.push(seg[0], seg[1], seg[2]);
        }
    }
    return { boost: boost, suppress: suppress };
}

function ensureVpsChart() {
    if (vpsChart) return vpsChart;
    var canvas = document.getElementById('vpsChart');
    if (!canvas) return null;
    var ctx = canvas.getContext('2d');
    var datasets = [];
    for (var i = 0; i < chart.data.datasets.length; i++) {
        var d = chart.data.datasets[i];
        datasets.push({
            label: d.label,
            data: [],
            borderColor: d.borderColor,
            backgroundColor: d.backgroundColor,
            borderWidth: d.borderWidth,
            tension: d.tension != null ? d.tension : 0.4,
            pointRadius: d.pointRadius != null ? d.pointRadius : 0,
            hidden: d.hidden,
            borderDash: d.borderDash,
            cubicInterpolationMode: d.cubicInterpolationMode,
            showLine: d.showLine,
            pointStyle: d.pointStyle,
            fill: d.fill
        });
    }
    datasets.push({
        label: 'delta v (boost)',
        data: [],
        borderColor: 'rgba(0, 229, 160, 0.9)',
        backgroundColor: 'transparent',
        borderWidth: 2,
        tension: 0,
        pointRadius: 0,
        fill: false
    });
    datasets.push({
        label: 'delta v (suppress)',
        data: [],
        borderColor: 'rgba(255, 68, 221, 0.9)',
        backgroundColor: 'transparent',
        borderWidth: 2,
        tension: 0,
        pointRadius: 0,
        fill: false
    });
    vpsChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            layout: { padding: { top: 16 } },
            plugins: {
                legend: { display: false },
                title: { display: false },
                tooltip: { enabled: false },
                zoom: { zoom: { enabled: false }, pan: { enabled: false } }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: false },
                    grid: { color: 'rgba(64, 64, 64, 0.3)' },
                    ticks: { color: '#b0b0b0', maxTicksLimit: 10 }
                },
                y: {
                    afterFit: function(scale) { scale.width = CHART_Y_AXIS_LEFT_WIDTH; },
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Circular Velocity v(r) [km/s]',
                        color: '#b0b0b0',
                        font: { size: 13, weight: '500' }
                    },
                    grid: { color: 'rgba(64, 64, 64, 0.3)' },
                    ticks: { color: '#b0b0b0' },
                    beginAtZero: true
                }
            }
        }
    });
    return vpsChart;
}

function copyMainChartToVps() {
    if (!vpsChart || !chart) return;
    for (var i = 0; i < chart.data.datasets.length && i < vpsChart.data.datasets.length; i++) {
        vpsChart.data.datasets[i].data = chart.data.datasets[i].data;
        vpsChart.data.datasets[i].hidden = true;
        vpsChart.data.datasets[i].label = chart.data.datasets[i].label;
    }
    var hist = getVpsHistogramLineData();
    if (vpsChart.data.datasets[12]) vpsChart.data.datasets[12].data = hist.boost;
    if (vpsChart.data.datasets[13]) vpsChart.data.datasets[13].data = hist.suppress;
    if (chart.options.scales && chart.options.scales.x) {
        vpsChart.options.scales.x.min = chart.options.scales.x.min;
        vpsChart.options.scales.x.max = chart.options.scales.x.max;
    }
    var points = getVpsDeltaVData();
    var maxAbs = 1;
    for (var j = 0; j < points.length; j++) {
        var ay = Math.abs(points[j].y);
        if (ay > maxAbs) maxAbs = ay;
    }
    vpsChart.options.scales.y.min = -maxAbs;
    vpsChart.options.scales.y.max = maxAbs;
    vpsChart.options.scales.y.title.display = true;
    vpsChart.options.scales.y.title.text = 'delta v (km/s)';
    vpsChart.update('none');
}

function updateVpsChart() {
    if (!isAutoFitted) return;
    var panel = document.getElementById('vps-panel');
    if (!panel || panel.style.display === 'none') return;
    ensureVpsChart();
    if (!vpsChart) return;
    copyMainChartToVps();
}

function syncVpsXAxis() {
    if (!vpsChart || !chart) return;
    copyMainChartToVps();
}
canvas.addEventListener('dblclick', resetChartZoom);

// =====================================================================
// RIGHT PANEL: SCIENTIFIC METRICS (server-computed)
// =====================================================================

var lastApiMetrics = null;  // cached metrics from last API response

/**
 * Update all right panel sections from server-provided metrics.
 * Falls back to '--' when no metrics are available.
 */
function updateRightPanel(apiData) {
    var m = (apiData && apiData.metrics) ? apiData.metrics : lastApiMetrics;
    if (apiData && apiData.metrics) {
        lastApiMetrics = apiData.metrics;
    }

    // --- Fit Quality ---
    var fit = m ? m.fit_quality : null;
    if (fit) {
        document.getElementById('metric-rms').textContent = fit.rms_km_s + ' km/s';
        var chi2El = document.getElementById('metric-chi2');
        chi2El.textContent = fit.chi2_reduced.toFixed(2);
        chi2El.className = 'metrics-value' + (fit.chi2_reduced <= 1.5 ? ' good' : fit.chi2_reduced <= 3 ? ' warn' : ' bad');
        document.getElementById('metric-hits-1s').textContent = fit.within_1sigma + '/' + fit.n_obs;
        document.getElementById('metric-hits-2s').textContent = fit.within_2sigma + '/' + fit.n_obs;
    } else {
        document.getElementById('metric-rms').textContent = '--';
        document.getElementById('metric-chi2').textContent = '--';
        document.getElementById('metric-chi2').className = 'metrics-value';
        document.getElementById('metric-hits-1s').textContent = '--';
        document.getElementById('metric-hits-2s').textContent = '--';
    }

    // --- Observation Summary ---
    var obs = m ? m.observation_summary : null;
    if (obs) {
        document.getElementById('metric-npoints').textContent = obs.n_points;
        document.getElementById('metric-radial-range').textContent = obs.r_min_kpc.toFixed(1) + ' - ' + obs.r_max_kpc.toFixed(1) + ' kpc';
        document.getElementById('metric-mean-err').textContent = obs.mean_error_km_s > 0 ? ('+/-' + obs.mean_error_km_s + ' km/s') : '--';
    } else {
        document.getElementById('metric-npoints').textContent = '--';
        document.getElementById('metric-radial-range').textContent = '--';
        document.getElementById('metric-mean-err').textContent = '--';
    }

    // --- Mass Model ---
    var mm = m ? m.mass_model : null;
    if (mm) {
        var totalExp = Math.floor(Math.log10(mm.total_baryonic_M_sun));
        var totalCoeff = mm.total_baryonic_M_sun / Math.pow(10, totalExp);
        document.getElementById('metric-total-mass').textContent = totalCoeff.toFixed(2) + 'e' + totalExp + ' M_sun';
        document.getElementById('metric-gas-frac').textContent = mm.gas_fraction_pct + '%';

        // Use inference-predicted field geometry when available (observation mode)
        var fg = inferredFieldGeometry;
        if (isAutoFitted && fg && fg.throat_radius_kpc !== null) {
            var rt = fg.throat_radius_kpc;
            var re = fg.envelope_radius_kpc;
            var rtCat = mm.field_origin_kpc;
            var reCat = mm.field_horizon_kpc;
            var rtDelta = rtCat > 0 ? ((rt - rtCat) / rtCat * 100).toFixed(1) : '0.0';
            var reDelta = reCat > 0 ? ((re - reCat) / reCat * 100).toFixed(1) : '0.0';
            document.getElementById('metric-field-origin').textContent =
                rt.toFixed(1) + ' kpc (' + (rtDelta >= 0 ? '+' : '') + rtDelta + '%)';
            document.getElementById('metric-field-horizon').textContent =
                re.toFixed(1) + ' kpc (' + (reDelta >= 0 ? '+' : '') + reDelta + '%)';
        } else {
            document.getElementById('metric-field-origin').textContent = mm.field_origin_kpc + ' kpc';
            document.getElementById('metric-field-horizon').textContent = mm.field_horizon_kpc + ' kpc';
        }
    } else {
        document.getElementById('metric-total-mass').textContent = '--';
        document.getElementById('metric-gas-frac').textContent = '--';
        document.getElementById('metric-field-origin').textContent = '--';
        document.getElementById('metric-field-horizon').textContent = '--';
    }

    // --- CDM Comparison ---
    if (lastCdmHalo) {
        var h = lastCdmHalo;
        var m200Exp = Math.floor(Math.log10(h.m200));
        var m200Coeff = h.m200 / Math.pow(10, m200Exp);
        document.getElementById('metric-cdm-m200').textContent = m200Coeff.toFixed(2) + 'e' + m200Exp + ' M_sun';
        document.getElementById('metric-cdm-c').textContent = h.c.toFixed(1);
        document.getElementById('metric-cdm-r200').textContent = h.r200_kpc ? (h.r200_kpc.toFixed(1) + ' kpc') : '--';
        document.getElementById('metric-cdm-method').textContent = h.method === 'chi-squared fit to observations'
            ? 'Chi-sq fit' : 'Abundance matching';
        document.getElementById('metric-cdm-params').textContent = (h.n_params_total || 2) + ' (NFW)';
    } else {
        document.getElementById('metric-cdm-m200').textContent = '--';
        document.getElementById('metric-cdm-c').textContent = '--';
        document.getElementById('metric-cdm-r200').textContent = '--';
        document.getElementById('metric-cdm-method').textContent = '--';
        document.getElementById('metric-cdm-params').textContent = '--';
    }

    // --- Residuals Table ---
    var residuals = m ? m.residuals : [];
    var tbody = document.getElementById('metrics-residuals-tbody');
    if (tbody) {
        var html = '';
        for (var i = 0; i < residuals.length; i++) {
            var row = residuals[i];
            var cls = '';
            if (row.sigma !== null) {
                cls = row.sigma <= 1.0 ? 'sigma-good' : row.sigma <= 2.0 ? 'sigma-warn' : 'sigma-bad';
            }
            html += '<tr class="' + cls + '">'
                + '<td>' + row.r_kpc.toFixed(1) + '</td>'
                + '<td>' + row.v_obs.toFixed(1) + '</td>'
                + '<td>' + row.v_gfd.toFixed(1) + '</td>'
                + '<td>' + row.delta_v.toFixed(1) + '</td>'
                + '<td>' + (row.sigma !== null ? row.sigma.toFixed(1) : '--') + '</td>'
                + '</tr>';
        }
        tbody.innerHTML = html;
    }
}

/**
 * Toggle a collapsible metrics section open/closed.
 */
function toggleMetricsSection(headerEl) {
    var section = headerEl.parentElement;
    var body = headerEl.nextElementSibling;
    var chevron = headerEl.querySelector('.metrics-chevron');
    if (body.style.display === 'none') {
        body.style.display = '';
        chevron.innerHTML = '&#9660;';
        section.classList.remove('collapsed');
    } else {
        body.style.display = 'none';
        chevron.innerHTML = '&#9654;';
        section.classList.add('collapsed');
    }
}

// =====================================================================
// API COMMUNICATION
// =====================================================================

function _buildCurveBody(maxRadius, accelRatio, massModel, observations, galacticRadius, mode) {
    var body = {
        max_radius: maxRadius,
        num_points: 100,
        accel_ratio: accelRatio,
        mass_model: massModel
    };
    if (observations) {
        body.observations = observations;
    }
    if (galacticRadius) {
        body.galactic_radius = galacticRadius;
    }
    if (mode === 'vortex' || mode === 'default') {
        body.mode = mode;
    }
    // Origin Throughput: send explicit value when user has overridden.
    if (!isAutoThroughput && vortexStrengthSlider) {
        body.vortex_strength = parseFloat(vortexStrengthSlider.value);
    }
    return body;
}

async function fetchRotationCurve(maxRadius, accelRatio, massModel, observations, galacticRadius, mode) {
    var body = _buildCurveBody(maxRadius, accelRatio, massModel, observations, galacticRadius, mode);
    const resp = await fetch('/api/rotation/curve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    if (!resp.ok) throw new Error('API error: ' + resp.status);
    return resp.json();
}

async function fetchInferRotationCurve(maxRadius, accelRatio, massModel, observations, galacticRadius) {
    var body = _buildCurveBody(maxRadius, accelRatio, massModel, observations, galacticRadius);
    const resp = await fetch('/api/rotation/infer-curve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    if (!resp.ok) throw new Error('API error: ' + resp.status);
    return resp.json();
}

async function fetchInferredMassModel(rKpc, vKmS, accelRatio, massModel) {
    const resp = await fetch('/api/rotation/infer-mass-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            r_kpc: rKpc,
            v_km_s: vKmS,
            accel_ratio: accelRatio,
            mass_model: massModel
        })
    });
    if (!resp.ok) throw new Error('API error: ' + resp.status);
    return resp.json();
}

async function fetchGalaxies() {
    const resp = await fetch('/api/rotation/galaxies');
    if (!resp.ok) throw new Error('API error: ' + resp.status);
    return resp.json();
}

async function fetchMultiPointInference(observations, accelRatio, massModel) {
    const resp = await fetch('/api/rotation/infer-mass-multi', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            observations: observations,
            accel_ratio: accelRatio,
            mass_model: massModel
        })
    });
    if (!resp.ok) return null;
    return resp.json();
}

function showChartLoading() {
    var overlay = document.getElementById('chart-loading-overlay');
    if (overlay) overlay.style.display = 'flex';
}

function hideChartLoading() {
    var overlay = document.getElementById('chart-loading-overlay');
    if (overlay) overlay.style.display = 'none';
}

let observationFetchPromise = null;

async function fetchPhotometricData(galaxyId, chartMode, maxRadius) {
    try {
        var payload = {
            galaxy_id: galaxyId,
            num_points: 500,
            mode: 'mass_model'
        };
        if (chartMode === 'vortex') {
            payload.chart_mode = 'vortex';
            if (maxRadius != null && maxRadius > 0) {
                payload.max_radius = maxRadius;
            }
        }
        if (chartMode === 'vortex_chart') {
            if (maxRadius != null && maxRadius > 0) {
                payload.max_radius = maxRadius;
            }
        }
        const resp = await fetch('/api/sandbox/photometric', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!resp.ok) {
            if (chartMode === 'vortex' || chartMode === 'vortex_chart') return null;
            sandboxResult = null;
            photometricResult = null;
            return;
        }
        var result = await resp.json();
        if (chartMode === 'vortex' || chartMode === 'vortex_chart') {
            return result;
        }
        sandboxResult = result;
        sandboxResult.gfd_source = 'photometric';
        photometricResult = sandboxResult;
        renderSandboxCurves();
        updatePhotometricPanel();
        updateDiagnosticsPanel();
        blankRightPaneMetrics();

        // Prefetch observation curves (sigma + accel) in background
        observationFetchPromise = fetchObservationData(galaxyId);
    } catch (e) {
        if (chartMode === 'vortex' || chartMode === 'vortex_chart') return null;
        sandboxResult = null;
        photometricResult = null;
    }
}

/**
 * Fetch GFD curve from manually set mass model (slider values).
 * Same API as photometric but mode 'manual' and mass_model in body.
 * Used when user has changed mass sliders after loading an example.
 */
async function fetchGfdFromMassModel(massModel, maxRadius, accelRatio) {
    try {
        const resp = await fetch('/api/sandbox/photometric', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: 'manual',
                mass_model: massModel,
                num_points: 500,
                max_radius: maxRadius,
                accel_ratio: accelRatio
            })
        });
        if (!resp.ok) return;
        var result = await resp.json();
        if (!sandboxResult) return;
        sandboxResult = {
            chart: result.chart,
            field_geometry: result.field_geometry,
            sparc_r_hi_kpc: sandboxResult.sparc_r_hi_kpc,
            photometric_mass_model: sandboxResult.photometric_mass_model,
            gfd_source: 'manual'
        };
        renderSandboxCurves();
    } catch (e) {
        console.error('Manual GFD fetch error:', e);
    }
}

async function fetchObservationData(galaxyId) {
    try {
        const resp = await fetch('/api/sandbox/photometric', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                galaxy_id: galaxyId,
                num_points: 500,
                mode: 'observations'
            })
        });
        if (!resp.ok) return;
        var result = await resp.json();
        if (result.chart && result.chart.gfd_spline) {
            photometricResult.chart.gfd_spline = result.chart.gfd_spline;
            photometricResult.chart.gfd_covariant_spline =
                result.chart.gfd_covariant_spline;
            photometricResult.chart.delta_v2_spline =
                result.chart.delta_v2_spline;
        }
        if (result.chart && result.chart.gfd_accel) {
            photometricResult.chart.gfd_accel = result.chart.gfd_accel;
        }
        if (result.vortex_signal_spline) {
            photometricResult.vortex_signal_spline =
                result.vortex_signal_spline;
        }
        if (result.accel_ratio_fitted != null) {
            photometricResult.accel_ratio_fitted =
                result.accel_ratio_fitted;
        }
        if (isAutoFitted) {
            sandboxResult = result;
            renderSandboxCurves();
        } else {
            updateObservationCurvesOnly();
            if (pinnedObservations && pinnedObservations.length > 0 && result.chart && result.chart.gfd_accel) {
                sandboxResult = result;
                navigateTo('charts', 'obs-chart');
            }
        }
        updateDiagnosticsPanel();
    } catch (e) {
        console.error('Observation data fetch error:', e);
    }
}

// fetchSandboxData: DISABLED (slow Bayesian endpoint, 9-12s).
// Replaced by fetchPhotometricData which includes fast GFD Sigma
// and GFD (with acceleration) from the photometric endpoint.
// Kept for reference; remove when no longer needed.
//
// async function fetchSandboxData(galaxyId) {
//     try {
//         const resp = await fetch('/api/sandbox/map_gfd_with_bayesian', {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify({
//                 galaxy_id: galaxyId,
//                 num_points: 500
//             })
//         });
//         if (!resp.ok) { return; }
//         sandboxResult = await resp.json();
//         renderSandboxCurves();
//         updatePhotometricPanel();
//         updateDiagnosticsPanel();
//     } catch (e) {
//         console.error('Sandbox fetch error:', e);
//     }
// }

// =====================================================================
// SANDBOX RENDERING: photometric + sigma curves from sandbox API
// =====================================================================

/**
 * Update only the observation-derived curve data (GFD Sigma, GFD Accel)
 * without changing scales or sandboxResult. Used when observation data
 * arrives in the background while the user is in mass model mode, so the
 * chart does not rezoom or rescale.
 */
function updateObservationCurvesOnly() {
    if (!photometricResult || !photometricResult.chart) return;
    var xMax = chart.options.scales && chart.options.scales.x
        ? chart.options.scales.x.max
        : 100;
    var splineSource = photometricResult.chart;
    var splineRadii = splineSource.radii || [];
    var splineVels = splineSource.gfd_spline || [];
    var splineData = [];
    for (var i = 0; i < splineRadii.length; i++) {
        if (splineRadii[i] > xMax) break;
        if (splineVels[i] !== undefined) {
            splineData.push({ x: splineRadii[i], y: splineVels[i] });
        }
    }
    chart.data.datasets[10].data = splineData;
    chart.data.datasets[10].hidden = true;  // Never show spline line; GFD (Observed) is dataset 11 only

    var accelVels = splineSource.gfd_accel || [];
    var accelRadii = splineSource.radii || [];
    var accelData = [];
    for (var j = 0; j < accelRadii.length; j++) {
        if (accelRadii[j] > xMax) break;
        if (accelVels[j] !== undefined) {
            accelData.push({ x: accelRadii[j], y: accelVels[j] });
        }
    }
    chart.data.datasets[11].data = accelData;
    chart.data.datasets[11].hidden = !isAutoFitted;

    chart.update('none');
}

function renderSandboxCurves() {
    if (!sandboxResult || !sandboxResult.chart) return;

    var radii = sandboxResult.chart.radii || [];
    var photoVels = sandboxResult.chart.gfd_photometric || [];
    var covariantVels = sandboxResult.chart.gfd_covariant || [];
    var fg = sandboxResult.field_geometry || {};

    // Compute axis limits from sandbox data
    var derivedRenv = fg.envelope_radius_kpc || 0;
    var derivedRvis99 = fg.visible_radius_99_kpc || fg.visible_radius_kpc || 0;
    var obsData = pinnedObservations || [];
    var maxObsR = obsData.length > 0
        ? Math.max(...obsData.map(function(o) { return o.r; })) : 0;
    var sparcRhi = sandboxResult.sparc_r_hi_kpc || 0;
    var xMax = Math.max(derivedRenv, derivedRvis99, maxObsR, sparcRhi) * 1.10;

    var obsMax = obsData.length > 0
        ? Math.max(...obsData.map(function(o) { return o.v; })) : 0;
    var yMax = obsMax > 0 ? Math.ceil(obsMax * 1.2 / 10) * 10 : undefined;

    // Build curve data, filtered to xMax
    var photoData = [];
    var sigmaData = [];
    for (var i = 0; i < radii.length; i++) {
        if (radii[i] > xMax) break;
        if (photoVels[i] !== undefined) {
            photoData.push({ x: radii[i], y: photoVels[i] });
        }
        if (covariantVels[i] !== undefined) {
            sigmaData.push({ x: radii[i], y: covariantVels[i] });
        }
    }

    // Spline data uses photometricResult's own radii to avoid index mismatch
    var splineData = [];
    var splineSource = (photometricResult && photometricResult.chart) || sandboxResult.chart;
    var splineRadii = splineSource.radii || [];
    var splineVels = splineSource.gfd_spline || [];
    for (var i = 0; i < splineRadii.length; i++) {
        if (splineRadii[i] > xMax) break;
        if (splineVels[i] !== undefined) {
            splineData.push({ x: splineRadii[i], y: splineVels[i] });
        }
    }

    // Dataset 1: GFD (Photometric) or GFD (Manual mass params), green dashed
    var gfdLabel = (sandboxResult.gfd_source === 'manual')
        ? 'GFD (Manual mass params)'
        : 'GFD (Photometric)';
    chart.data.datasets[1].data = photoData;
    chart.data.datasets[1].label = gfdLabel;
    var gfdChipLabel = document.getElementById('gfd-chip-label');
    if (gfdChipLabel) gfdChipLabel.textContent = gfdLabel;
    chart.data.datasets[1].borderColor = 'rgba(0, 229, 160, 0.7)';
    chart.data.datasets[1].backgroundColor = 'transparent';
    chart.data.datasets[1].borderDash = [8, 4];
    chart.data.datasets[1].borderWidth = 2;
    chart.data.datasets[1].cubicInterpolationMode = 'monotone';
    chart.data.datasets[1].tension = 0.4;

    // Dataset 8: old Bayesian GFD Sigma (disabled, kept for legacy)
    chart.data.datasets[8].data = [];
    chart.data.datasets[8].hidden = true;

    // Dataset 10: GFD (Observed), magenta solid
    chart.data.datasets[10].data = splineData;
    chart.data.datasets[10].label = 'GFD (Observed)';
    chart.data.datasets[10].borderColor = '#ff44dd';
    chart.data.datasets[10].backgroundColor = 'transparent';
    chart.data.datasets[10].borderDash = [];
    chart.data.datasets[10].borderWidth = 2.5;
    chart.data.datasets[10].cubicInterpolationMode = 'monotone';
    chart.data.datasets[10].tension = 0.4;
    chart.data.datasets[10].hidden = true;  // Spline curve not shown; only GFD (Observed) accel line (dataset 11)

    // Dataset 11: GFD (Observed), purple solid
    var accelSource = (photometricResult && photometricResult.chart) || sandboxResult.chart;
    var accelVels = accelSource.gfd_accel || [];
    var accelRadii = accelSource.radii || [];
    var accelData = [];
    for (var i = 0; i < accelRadii.length; i++) {
        if (accelRadii[i] > xMax) break;
        if (accelVels[i] !== undefined) {
            accelData.push({ x: accelRadii[i], y: accelVels[i] });
        }
    }
    chart.data.datasets[11].data = accelData;
    chart.data.datasets[11].label = 'GFD (Observed)';
    chart.data.datasets[11].borderColor = '#aa44ff';
    chart.data.datasets[11].backgroundColor = 'transparent';
    chart.data.datasets[11].borderDash = [];
    chart.data.datasets[11].borderWidth = 2.5;
    chart.data.datasets[11].cubicInterpolationMode = 'monotone';
    chart.data.datasets[11].tension = 0.4;
    chart.data.datasets[11].hidden = !isAutoFitted;

    // Clear envelope/band and legacy theory curves; keep Newton/MOND/CDM (0,2,7) for chip toggles
    chart.data.datasets[4].data = [];   // Envelope upper
    chart.data.datasets[5].data = [];   // Envelope lower
    chart.data.datasets[6].data = [];   // Auto fit markers
    chart.data.datasets[9].data = [];   // GFD Observed

    // Apply axis rules
    chart.options.scales.x.min = 0;
    chart.options.scales.x.max = xMax;
    chart.options.scales.y.min = 0;
    if (yMax) chart.options.scales.y.max = yMax;

    // Store field geometry for plugins
    chart.options.plugins.sandboxGeometry = fg;

    chart.update('none');

    if (isAutoFitted) updateVpsChart();
}

function updatePhotometricPanel() {
    if (!sandboxResult) return;
    var pm = sandboxResult.photometric_mass_model;
    if (!pm) return;

    var gasMass = pm.gas ? pm.gas.M : 0;
    var gasScale = pm.gas ? pm.gas.Rd : 0;
    var diskMass = pm.disk ? pm.disk.M : 0;
    var diskScale = pm.disk ? pm.disk.Rd : 0;
    var bulgeMass = pm.bulge ? pm.bulge.M : 0;
    var bulgeScale = pm.bulge ? pm.bulge.a : 0;
    var totalMass = gasMass + diskMass + bulgeMass;

    // Update display values (Mass Model panel)
    document.getElementById('gas-mass-value').textContent = gasMass.toExponential(1) + ' M_sun';
    document.getElementById('gas-scale-value').textContent = gasScale.toFixed(1) + ' kpc';
    document.getElementById('disk-mass-value').textContent = diskMass.toExponential(1) + ' M_sun';
    document.getElementById('disk-scale-value').textContent = diskScale.toFixed(1) + ' kpc';
    document.getElementById('bulge-mass-value').textContent = bulgeMass.toExponential(1) + ' M_sun';
    document.getElementById('bulge-scale-value').textContent = bulgeScale.toFixed(1) + ' kpc';
    document.getElementById('mass-model-total-value').textContent = totalMass.toExponential(1) + ' M_sun';
}

/**
 * Populate the read-only Observation mode mass panel from photometric_mass_model.
 * Called when entering Observation mode so the panel shows current photometric values.
 */
function populateObservationMassPanel() {
    var el = function(id) { return document.getElementById(id); };
    if (!el('obs-gas-mass-value')) return;
    var pm = sandboxResult && sandboxResult.photometric_mass_model;
    if (!pm) {
        el('obs-gas-mass-value').textContent = '--';
        el('obs-gas-scale-value').textContent = '--';
        el('obs-disk-mass-value').textContent = '--';
        el('obs-disk-scale-value').textContent = '--';
        el('obs-bulge-mass-value').textContent = '--';
        el('obs-bulge-scale-value').textContent = '--';
        el('obs-mass-model-total-value').textContent = '--';
        return;
    }
    var gasMass = pm.gas ? pm.gas.M : 0;
    var gasScale = pm.gas ? pm.gas.Rd : 0;
    var diskMass = pm.disk ? pm.disk.M : 0;
    var diskScale = pm.disk ? pm.disk.Rd : 0;
    var bulgeMass = pm.bulge ? pm.bulge.M : 0;
    var bulgeScale = pm.bulge ? pm.bulge.a : 0;
    var totalMass = gasMass + diskMass + bulgeMass;
    el('obs-gas-mass-value').textContent = gasMass.toExponential(1) + ' M_sun';
    el('obs-gas-scale-value').textContent = gasScale.toFixed(1) + ' kpc';
    el('obs-disk-mass-value').textContent = diskMass.toExponential(1) + ' M_sun';
    el('obs-disk-scale-value').textContent = diskScale.toFixed(1) + ' kpc';
    el('obs-bulge-mass-value').textContent = bulgeMass.toExponential(1) + ' M_sun';
    el('obs-bulge-scale-value').textContent = bulgeScale.toFixed(1) + ' kpc';
    el('obs-mass-model-total-value').textContent = totalMass.toExponential(1) + ' M_sun';
}

function updateDiagnosticsPanel() {
    var section = document.getElementById('gfd-diagnostics-section');
    var body = document.getElementById('gfd-diagnostics-body');
    if (!section || !body) return;
    if (!sandboxResult) { section.style.display = 'none'; return; }

    section.style.display = '';
    var fg = sandboxResult.field_geometry || {};
    var vs = sandboxResult.vortex_signal || {};
    var hasBayesian = sandboxResult.rms != null;
    var rms = sandboxResult.rms || 0;
    var chi2 = sandboxResult.chi2_dof || 0;
    var nObs = sandboxResult.n_obs || 0;

    var rvis90 = fg.visible_radius_90_kpc || 0;
    var rvis99 = fg.visible_radius_99_kpc || 0;
    var rvisStr = (rvis90 && rvis99)
        ? rvis90.toFixed(1) + ' to ' + rvis99.toFixed(1) + ' kpc (90% to 99.5%)'
        : (fg.visible_radius_kpc ? fg.visible_radius_kpc.toFixed(2) + ' kpc' : 'N/A');

    var pendingSpan = '<span style="color:#666;font-style:italic">switch to Observations</span>';

    var html = '';
    html += '<div style="color:#999;font-weight:600;margin-bottom:4px;">Bayesian Fit</div>';
    if (hasBayesian) {
        html += 'RMS: <span style="color:#e0e0e0">' + rms.toFixed(2) + ' km/s</span><br>';
        html += 'chi2/dof: <span style="color:#e0e0e0">' + chi2.toFixed(4) + '</span><br>';
        html += 'Observations: <span style="color:#e0e0e0">' + nObs + '</span><br>';
    } else {
        html += 'RMS: ' + pendingSpan + '<br>';
        html += 'chi2/dof: ' + pendingSpan + '<br>';
        html += 'Observations: <span style="color:#e0e0e0">' + nObs + '</span><br>';
    }

    html += '<div style="color:#999;font-weight:600;margin-top:8px;margin-bottom:4px;">Vortex Signal (MACD)</div>';
    if (hasBayesian) {
        html += 'sigma (net): <span style="color:#e0e0e0">' + (vs.sigma_net || 0).toFixed(4) + '</span><br>';
        html += 'Boost energy: <span style="color:#4caf50">+' + Math.round(vs.energy_boost || 0).toLocaleString() + ' km2/s2</span><br>';
        html += 'Suppress energy: <span style="color:#ff6b6b">' + Math.round(vs.energy_suppress || 0).toLocaleString() + ' km2/s2</span><br>';
        html += 'Boost/Suppress: <span style="color:#e0e0e0">' + (vs.energy_ratio || 0).toFixed(2) + '</span><br>';
    } else {
        html += 'sigma (net): ' + pendingSpan + '<br>';
        html += 'Boost energy: ' + pendingSpan + '<br>';
        html += 'Suppress energy: ' + pendingSpan + '<br>';
        html += 'Boost/Suppress: ' + pendingSpan + '<br>';
    }

    html += '<div style="color:#999;font-weight:600;margin-top:8px;margin-bottom:4px;">Field Geometry</div>';
    html += 'R_t (throat): <span style="color:#00ff88">' + (fg.throat_radius_kpc ? fg.throat_radius_kpc.toFixed(2) + ' kpc' : 'N/A') + '</span><br>';
    html += 'R_env (field horizon): <span style="color:#ff6688">' + (fg.envelope_radius_kpc ? fg.envelope_radius_kpc.toFixed(2) + ' kpc' : 'N/A') + '</span><br>';
    var rHiVal = sandboxResult.sparc_r_hi_kpc;
    html += 'R_HI (SPARC extent): <span style="color:#ffaa44">' + (rHiVal ? rHiVal.toFixed(1) + ' kpc' : 'N/A') + '</span><br>';
    html += 'R_vis (baryonic extent): <span style="color:#aa88ff">' + rvisStr + '</span><br>';
    html += 'R_t / R_env: <span style="color:#e0e0e0">' + (fg.throat_fraction ? fg.throat_fraction.toFixed(4) : 'N/A') + '</span><br>';
    html += 'Cycle: <span style="color:#e0e0e0">' + (fg.cycle || '?') + '</span>';

    var accelRatio = sandboxResult.accel_ratio_fitted;
    var accelRms = sandboxResult.accel_rms;
    if (accelRatio != null) {
        html += '<div style="color:#999;font-weight:600;margin-top:8px;margin-bottom:4px;">Acceleration Fit</div>';
        html += 'a0 ratio: <span style="color:#aa44ff">' + accelRatio.toFixed(4) + 'x</span><br>';
        html += 'RMS: <span style="color:#e0e0e0">' + (accelRms || 0).toFixed(2) + ' km/s</span><br>';
    }

    body.innerHTML = html;
}

/**
 * Find the prediction counterpart for an inference galaxy and return
 * its observations array, or null if none available.
 */
function getPredictionObservations(inferenceGalaxy) {
    if (!inferenceGalaxy) return null;
    var baseId = inferenceGalaxy.id.replace(/_inference$/, '');
    if (baseId === 'mw') baseId = 'milky_way';
    var predictions = galaxyCatalog.prediction || [];
    for (var i = 0; i < predictions.length; i++) {
        if (predictions[i].id === baseId && predictions[i].observations) {
            return predictions[i].observations;
        }
    }
    return null;
}

// =====================================================================
// MULTI-POINT INFERENCE ANALYSIS
// =====================================================================

/**
 * Scale a mass model by a factor (multiply all component masses).
 */
function scaleMassModel(model, factor) {
    var scaled = {};
    var comps = ['bulge', 'disk', 'gas'];
    for (var c = 0; c < comps.length; c++) {
        var key = comps[c];
        if (model[key]) {
            scaled[key] = {};
            for (var prop in model[key]) {
                scaled[key][prop] = model[key][prop];
            }
            if (scaled[key].M) scaled[key].M *= factor;
        }
    }
    return scaled;
}

/**
 * Interpolate the GFD curve (dataset 1) to get velocity at a given radius.
 */
/**
 * Interpolate any chart dataset curve at an arbitrary radius.
 * Uses linear interpolation between bracketing points.
 * @param {number} datasetIndex - Chart dataset index (0=Newton, 1=GFD, 2=MOND, 7=CDM)
 * @param {number} radius - Galactocentric radius in kpc
 * @returns {number|null} Interpolated velocity in km/s, or null if no data
 */
function interpolateCurve(datasetIndex, radius) {
    var data = chart.data.datasets[datasetIndex].data;
    if (!data || data.length === 0) return null;
    // Before first point
    if (radius <= data[0].x) return data[0].y;
    // Find bracketing points
    for (var i = 1; i < data.length; i++) {
        if (data[i].x >= radius) {
            var x0 = data[i-1].x, y0 = data[i-1].y;
            var x1 = data[i].x, y1 = data[i].y;
            if (x1 === x0) return y0;
            var t = (radius - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    // Beyond curve range -- return null to avoid stale extrapolation
    return null;
}

/** Shorthand: interpolate GFD curve (dataset 1) */
function interpolateGFDVelocity(radius) {
    return interpolateCurve(1, radius);
}

async function runMultiPointInference(accelRatio, massModel) {
    var multiDiv = document.getElementById('multi-inference-result');
    var multiBody = document.getElementById('multi-inference-body');
    if (!multiDiv || !multiBody) return;

    if (currentMode !== 'inference' && !isAutoFitted) {
        multiDiv.style.display = 'none';
        clearInferenceChart();
        return;
    }

    // Use pinned observations if user is fine-tuning, otherwise fall back to currentExample
    var observations = pinnedObservations || (currentExample ? getPredictionObservations(currentExample) : null);
    if (!observations || observations.length < 2) {
        multiDiv.style.display = 'none';
        clearInferenceChart();
        return;
    }

    try {
        var result = await fetchMultiPointInference(observations, accelRatio, massModel);
        if (!result || !result.points) {
            multiDiv.style.display = 'none';
            clearInferenceChart();
            return;
        }

        // Cache for band method switching
        var modelTotal = 0;
        var comps = ['bulge', 'disk', 'gas'];
        for (var c = 0; c < comps.length; c++) {
            if (massModel[comps[c]] && massModel[comps[c]].M) {
                modelTotal += massModel[comps[c]].M;
            }
        }
        lastMultiResult = result;
        lastModelTotal = modelTotal;
        lastAccelRatio = accelRatio;
        lastMassModel = massModel;

        // Clear inference markers (green diamonds removed for cleaner UI)
        chart.data.datasets[6].data = [];

        // Render sidebar first so #band-width-display exists in the DOM
        // before updateBand() calls updateBandLabel()
        renderMultiPointSidebar(result, modelTotal);

        // Compute and render the band for the selected method
        await updateBand();

    } catch (err) {
        console.error('Multi-point inference error:', err);
        multiDiv.style.display = 'none';
        clearInferenceChart();
    }
}

/**
 * Compute the band half-width for the selected method.
 * Returns the half-width in solar masses.
 */
function getBandHalfWidth(method) {
    var result = lastMultiResult;
    var modelTotal = lastModelTotal;
    if (!result || !result.points || modelTotal <= 0) return 0;

    var points = result.points;
    var methods = result.band_methods || {};

    if (method === 'weighted_rms') {
        // Weighted RMS from anchor (computed on frontend)
        var wSumSqDev = 0, wSumW = 0;
        for (var i = 0; i < points.length; i++) {
            var w = points[i].enclosed_frac;
            var diff = points[i].inferred_total - modelTotal;
            wSumSqDev += w * diff * diff;
            wSumW += w;
        }
        return wSumW > 0 ? Math.sqrt(wSumSqDev / wSumW) : 0;
    }
    if (method === 'weighted_scatter') return methods.weighted_scatter || 0;
    if (method === 'obs_error') return methods.obs_error || 0;
    if (method === 'min_max') {
        // Max absolute deviation from anchor -- guarantees all points
        // fall inside the band when centered on modelTotal.
        var maxDev = 0;
        for (var i = 0; i < points.length; i++) {
            var dev = Math.abs(points[i].inferred_total - modelTotal);
            if (dev > maxDev) maxDev = dev;
        }
        return maxDev;
    }
    if (method === 'iqr') return methods.iqr || 0;
    return 0;
}

/**
 * Get a human-readable label and description for a band method.
 */
function getBandMethodInfo(method) {
    var labels = {
        'weighted_rms':     {label: 'Weighted RMS',  desc: 'Spread of per-point mass estimates vs GFD model, weighted by enclosed fraction'},
        'weighted_scatter': {label: '1-sigma Scatter', desc: 'Weighted std dev of per-point mass estimates around their mean'},
        'obs_error':        {label: 'Obs. Error',   desc: 'Propagated velocity measurement uncertainties through the field equation'},
        'min_max':          {label: 'Min-Max',       desc: 'Full range of per-point inferred masses (most conservative)'},
        'iqr':              {label: 'IQR (Robust)',  desc: 'Interquartile range, resistant to outlier inner points'}
    };
    return labels[method] || {label: method, desc: ''};
}

/**
 * Recompute and render the confidence band.
 *
 * In Auto Fit mode the band is the (4/pi)^(1/4) geometric band
 * around the GFD base curve, derived from the k=4 coupling topology.
 * This gives a +/- 6.2% physics-derived uncertainty envelope.
 */
async function updateBand() {
    // After auto-map, wrap the band around GFD (Observed) (dataset 9)
    // since that is the primary answer curve. Otherwise use GFD base (dataset 1).
    var gfdData = isAutoFitted
        ? chart.data.datasets[9].data
        : chart.data.datasets[1].data;
    if (!gfdData || gfdData.length === 0) {
        chart.data.datasets[4].data = [];
        chart.data.datasets[5].data = [];
        chart.update('none');
        return;
    }

    var pct = parseFloat(lensSlider.value) / 100.0;
    var upperData = [], lowerData = [];
    for (var i = 0; i < gfdData.length; i++) {
        var v = gfdData[i].y;
        var x = gfdData[i].x;
        upperData.push({x: x, y: v * (1 + pct)});
        lowerData.push({x: x, y: v * (1 - pct)});
    }
    chart.data.datasets[4].data = upperData;
    chart.data.datasets[5].data = lowerData;

    chart.update('none');
    updateBandLabel();
}

/**
 * Update just the band width display in the sidebar.
 */
function updateBandLabel() {
    var el = document.getElementById('band-width-display');
    if (!el) return;
    if (isAutoFitted) {
        el.innerHTML = '<strong style="color:#e0e0e0;">Band:</strong> '
            + '<span style="color:#76FF03;">(4/\u03C0)<sup>1/4</sup> = \u00B16.2%</span>'
            + '<div style="font-size:0.8em; color:#606060; margin-top:2px;">'
            + 'Geometric envelope from k=4 coupling topology</div>';
    } else {
        el.innerHTML = '';
    }
}

/**
 * Render the lens throughput band from the slider value around the GFD curve.
 * Called when the lens slider changes or after chart data updates.
 */
function updateLensBand() {
    // After auto-map, wrap the band around GFD (Observed) (dataset 9)
    var gfdData = isAutoFitted
        ? chart.data.datasets[9].data
        : chart.data.datasets[1].data;
    if (!gfdData || gfdData.length === 0) {
        chart.data.datasets[4].data = [];
        chart.data.datasets[5].data = [];
        chart.update('none');
        return;
    }
    var pct = parseFloat(lensSlider.value) / 100.0;
    var upperData = [], lowerData = [];
    for (var i = 0; i < gfdData.length; i++) {
        var v = gfdData[i].y;
        var x = gfdData[i].x;
        upperData.push({x: x, y: v * (1 + pct)});
        lowerData.push({x: x, y: v * (1 - pct)});
    }
    chart.data.datasets[4].data = upperData;
    chart.data.datasets[5].data = lowerData;
    chart.update('none');
}

/**
 * Render the multi-point sidebar statistics and table.
 */
function renderMultiPointSidebar(result, modelTotal) {
    var multiDiv = document.getElementById('multi-inference-result');
    var multiBody = document.getElementById('multi-inference-body');
    if (!multiDiv || !multiBody) return;

    var wMean = result.weighted_mean;
    var wStd = result.weighted_std;
    var wCv = result.weighted_cv_percent;
    var wExp = Math.floor(Math.log10(wMean));
    var wCoeff = wMean / Math.pow(10, wExp);
    var wStdCoeff = wStd / Math.pow(10, wExp);

    var anchorExp = Math.floor(Math.log10(modelTotal));
    var anchorCoeff = modelTotal / Math.pow(10, anchorExp);

    var html = '';

    html += '<div style="margin-bottom: 10px; line-height: 1.5;">';
    html += 'Mass inferred at <strong style="color:#e0e0e0;">' + result.n_points + '</strong> radii. ';
    html += '<span style="color:#76FF03;">Band</span> = GFD / GFD\u03C6 envelope.';
    html += '</div>';

    html += '<div style="margin-bottom: 6px;">';
    html += '<strong style="color:#e0e0e0;">GFD Model:</strong> ';
    html += '<span style="color:#4da6ff;">' + anchorCoeff.toFixed(2);
    html += ' \u00D7 10' + superscript(anchorExp) + ' M\u2609</span>';
    html += '</div>';

    html += '<div style="margin-bottom: 6px;">';
    html += '<strong style="color:#e0e0e0;">Obs. Mean:</strong> ';
    html += '<span style="color:#e0e0e0;">' + wCoeff.toFixed(2) + ' \u00B1 ' + wStdCoeff.toFixed(2);
    html += ' \u00D7 10' + superscript(wExp) + ' M\u2609</span>';
    html += '</div>';

    // Band width display (updated dynamically when method changes)
    html += '<div id="band-width-display" style="margin-bottom: 6px;"></div>';

    var agreementPct = Math.abs(wMean - modelTotal) / modelTotal * 100;
    var agColor = agreementPct < 10 ? '#4caf50' : agreementPct < 30 ? '#ffa726' : '#ef5350';
    var agSign = wMean < modelTotal ? '-' : '+';
    html += '<div style="margin-bottom: 10px;">';
    html += '<strong style="color:#e0e0e0;">Mass Offset:</strong> ';
    html += '<span style="color:' + agColor + ';">' + agSign + agreementPct.toFixed(1) + '%</span>';
    html += ' <span style="color:#606060; font-size:0.85em;">(obs. mean vs GFD model)</span>';
    html += '</div>';

    // Per-point table
    html += '<table style="width:100%; border-collapse:collapse; margin-top:8px; font-size:0.85em;">';
    html += '<tr style="border-bottom:1px solid #404040; color:#808080;">';
    html += '<th style="text-align:left; padding:4px 6px;">r (kpc)</th>';
    html += '<th style="text-align:left; padding:4px 6px;">v (km/s)</th>';
    html += '<th style="text-align:right; padding:4px 6px;">M enc.</th>';
    html += '<th style="text-align:right; padding:4px 6px;" title="Velocity residual: observed minus GFD model">\u0394v</th>';
    html += '<th style="text-align:right; padding:4px 6px;" title="Sigma deviation: |v_obs - v_GFD| / error">\u03C3</th>';
    html += '</tr>';

    for (var i = 0; i < result.points.length; i++) {
        var pt = result.points[i];
        var errStr = pt.err ? ' \u00B1 ' + pt.err : '';
        var encPct = (pt.enclosed_frac * 100).toFixed(0);
        var rowOpacity = pt.enclosed_frac < 0.1 ? 'opacity: 0.5;' : '';
        // Velocity residual: compare observed v to GFD prediction at this radius
        var gfdV = interpolateGFDVelocity(pt.r_kpc);
        var deltaV = gfdV !== null ? (pt.v_km_s - gfdV) : null;
        var sigmaAway = (deltaV !== null && pt.err > 0) ? Math.abs(deltaV) / pt.err : null;
        var dvColor = '#4caf50';
        if (sigmaAway !== null) {
            dvColor = sigmaAway < 1 ? '#4caf50' : sigmaAway < 2 ? '#ffa726' : '#ef5350';
        }
        var dvStr = deltaV !== null ? ((deltaV >= 0 ? '+' : '') + deltaV.toFixed(1)) : '\u2014';
        var sigStr = sigmaAway !== null ? sigmaAway.toFixed(1) : '\u2014';
        html += '<tr style="border-bottom:1px solid #2a2a2a; ' + rowOpacity + '">';
        html += '<td style="padding:3px 6px; color:#e0e0e0;">' + pt.r_kpc + '</td>';
        html += '<td style="padding:3px 6px; color:#e0e0e0;">' + pt.v_km_s + errStr + '</td>';
        html += '<td style="text-align:right; padding:3px 6px; color:#808080;">' + encPct + '%</td>';
        html += '<td style="text-align:right; padding:3px 6px; color:' + dvColor + ';">' + dvStr + '</td>';
        html += '<td style="text-align:right; padding:3px 6px; color:' + dvColor + ';">' + sigStr + '</td>';
        html += '</tr>';
    }
    html += '</table>';

    html += '<div style="margin-top: 8px; font-size: 0.8em; color: #606060;">';
    html += '\u0394v = v<sub>obs</sub> \u2212 v<sub>GFD</sub> (km/s). \u03C3 = |\u0394v| / error.';
    html += '</div>';

    // ── Shape Diagnostic ──
    // Split points into inner half and outer half, compute mean delta-v and sigma
    var dvArr = [];
    for (var i = 0; i < result.points.length; i++) {
        var pt = result.points[i];
        if (pt.enclosed_frac < 0.05) continue;  // skip unreliable innermost
        var gfdV = interpolateGFDVelocity(pt.r_kpc);
        if (gfdV === null) continue;
        var dv = pt.v_km_s - gfdV;
        var sig = (pt.err && pt.err > 0) ? Math.abs(dv) / pt.err : 0;
        dvArr.push({ r: pt.r_kpc, dv: dv, sigma: sig });
    }

    if (dvArr.length >= 4) {
        var mid = Math.floor(dvArr.length / 2);
        var innerPts = dvArr.slice(0, mid);
        var outerPts = dvArr.slice(mid);

        var avgInnerDv = innerPts.reduce(function(s, p) { return s + p.dv; }, 0) / innerPts.length;
        var avgOuterDv = outerPts.reduce(function(s, p) { return s + p.dv; }, 0) / outerPts.length;
        var avgInnerSig = innerPts.reduce(function(s, p) { return s + p.sigma; }, 0) / innerPts.length;
        var avgOuterSig = outerPts.reduce(function(s, p) { return s + p.sigma; }, 0) / outerPts.length;

        var innerR = innerPts[innerPts.length - 1].r;
        var outerR0 = outerPts[0].r;

        var innerColor = avgInnerSig < 1 ? '#4caf50' : avgInnerSig < 2 ? '#ffa726' : '#ef5350';
        var outerColor = avgOuterSig < 1 ? '#4caf50' : avgOuterSig < 2 ? '#ffa726' : '#ef5350';

        html += '<div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #404040;">';
        html += '<strong style="color: #e0e0e0;">Shape Diagnostic</strong>';
        html += '<table style="width:100%; border-collapse:collapse; margin-top:6px; font-size:0.85em;">';
        html += '<tr style="border-bottom:1px solid #404040; color:#808080;">';
        html += '<th style="text-align:left; padding:3px 6px;">Region</th>';
        html += '<th style="text-align:right; padding:3px 6px;">Mean \u0394v</th>';
        html += '<th style="text-align:right; padding:3px 6px;">Mean \u03C3</th>';
        html += '</tr>';
        html += '<tr style="border-bottom:1px solid #2a2a2a;">';
        html += '<td style="padding:3px 6px; color:#b0b0b0;">Inner (r < ' + innerR.toFixed(1) + ')</td>';
        html += '<td style="text-align:right; padding:3px 6px; color:' + innerColor + ';">' + (avgInnerDv >= 0 ? '+' : '') + avgInnerDv.toFixed(1) + '</td>';
        html += '<td style="text-align:right; padding:3px 6px; color:' + innerColor + ';">' + avgInnerSig.toFixed(1) + '</td>';
        html += '</tr>';
        html += '<tr style="border-bottom:1px solid #2a2a2a;">';
        html += '<td style="padding:3px 6px; color:#b0b0b0;">Outer (r > ' + outerR0.toFixed(1) + ')</td>';
        html += '<td style="text-align:right; padding:3px 6px; color:' + outerColor + ';">' + (avgOuterDv >= 0 ? '+' : '') + avgOuterDv.toFixed(1) + '</td>';
        html += '<td style="text-align:right; padding:3px 6px; color:' + outerColor + ';">' + avgOuterSig.toFixed(1) + '</td>';
        html += '</tr>';
        html += '</table>';

        // Recommendation logic: sigma is the gatekeeper, not raw delta-v.
        // If both regions have low sigma, the fit is good regardless of
        // the raw km/s offset (which may just reflect measurement scatter).
        var overallSig = (avgInnerSig + avgOuterSig) / 2;
        var bothGood = avgInnerSig < 1.5 && avgOuterSig < 1.5;
        var innerBad = avgInnerSig >= 2;
        var outerBad = avgOuterSig >= 2;
        var innerOver = avgInnerDv < 0;   // GFD overpredicts inner = negative dv
        var outerOver = avgOuterDv < 0;
        // Gradient only meaningful when at least one region has significant sigma
        var hasShapeIssue = (innerBad || outerBad) && Math.abs(avgInnerSig - avgOuterSig) > 1.5;

        if (bothGood) {
            // Both regions within ~1.5 sigma -- excellent fit
            html += '<div style="margin-top: 8px; padding: 8px; background: #1a1a1a; border-radius: 4px; border-left: 3px solid #4caf50; font-size: 0.85em;">';
            html += '<div style="color: #4caf50;">';
            html += '<strong>Excellent shape fit</strong> \u2014 all residuals within measurement error.';
            html += '</div>';
            html += '</div>';
        } else if (hasShapeIssue) {
            html += '<div style="margin-top: 8px; padding: 8px; background: #1a1a1a; border-radius: 4px; border-left: 3px solid #4da6ff; font-size: 0.85em;">';

            if (innerBad && !outerBad && innerOver) {
                html += '<div style="color: #e0e0e0; margin-bottom: 4px;">';
                html += '<strong>Mass too centrally concentrated</strong>';
                html += '</div>';
                html += '<div style="color: #b0b0b0;">';
                html += 'Inner radii: mean ' + avgInnerSig.toFixed(1) + '\u03C3 deviation. ';
                html += 'Try <strong style="color:#4da6ff;">increasing</strong> the Disk or Gas ';
                html += '<strong style="color:#4da6ff;">Scale length</strong> to redistribute mass outward.';
                html += '</div>';
            } else if (innerBad && !outerBad && !innerOver) {
                html += '<div style="color: #e0e0e0; margin-bottom: 4px;">';
                html += '<strong>Mass deficit at center</strong>';
                html += '</div>';
                html += '<div style="color: #b0b0b0;">';
                html += 'Inner radii: mean ' + avgInnerSig.toFixed(1) + '\u03C3 deviation. ';
                html += 'Try <strong style="color:#4da6ff;">decreasing</strong> the Disk or Gas ';
                html += '<strong style="color:#4da6ff;">Scale length</strong>, ';
                html += 'or increasing the Bulge <strong style="color:#4da6ff;">Scale radius</strong>.';
                html += '</div>';
            } else if (!innerBad && outerBad && outerOver) {
                html += '<div style="color: #e0e0e0; margin-bottom: 4px;">';
                html += '<strong>Mass too radially extended</strong>';
                html += '</div>';
                html += '<div style="color: #b0b0b0;">';
                html += 'Outer radii: mean ' + avgOuterSig.toFixed(1) + '\u03C3 deviation. ';
                html += 'Try <strong style="color:#4da6ff;">decreasing</strong> the Disk or Gas ';
                html += '<strong style="color:#4da6ff;">Scale length</strong> to concentrate mass inward.';
                html += '</div>';
            } else if (!innerBad && outerBad && !outerOver) {
                html += '<div style="color: #e0e0e0; margin-bottom: 4px;">';
                html += '<strong>Mass deficit at outer radii</strong>';
                html += '</div>';
                html += '<div style="color: #b0b0b0;">';
                html += 'Outer radii: mean ' + avgOuterSig.toFixed(1) + '\u03C3 deviation. ';
                html += 'Try <strong style="color:#4da6ff;">increasing</strong> the Gas ';
                html += '<strong style="color:#4da6ff;">Scale length</strong> to extend the mass distribution.';
                html += '</div>';
            } else {
                html += '<div style="color: #e0e0e0; margin-bottom: 4px;">';
                html += '<strong>Shape mismatch</strong>';
                html += '</div>';
                html += '<div style="color: #b0b0b0;">';
                html += 'Adjust the Disk and Gas <strong style="color:#4da6ff;">Scale length</strong> sliders to improve the inner/outer balance.';
                html += '</div>';
            }

            html += '</div>';
        } else if (innerBad && outerBad) {
            // Both regions have high sigma -- systematic issue
            html += '<div style="margin-top: 8px; padding: 8px; background: #1a1a1a; border-radius: 4px; border-left: 3px solid #ffa726; font-size: 0.85em;">';
            html += '<div style="color: #e0e0e0; margin-bottom: 4px;">';
            html += '<strong>Systematic ' + (innerOver ? 'overestimate' : 'underestimate') + '</strong>';
            html += '</div>';
            html += '<div style="color: #b0b0b0;">';
            html += 'Mean deviation: ' + overallSig.toFixed(1) + '\u03C3 across all radii. ';
            html += 'The overall normalization may need adjustment.';
            html += '</div>';
            html += '</div>';
        } else {
            // Moderate fit -- one region borderline
            html += '<div style="margin-top: 8px; padding: 8px; background: #1a1a1a; border-radius: 4px; border-left: 3px solid #ffa726; font-size: 0.85em;">';
            html += '<div style="color: #ffa726;">';
            html += '<strong>Adequate fit</strong> \u2014 mean ' + overallSig.toFixed(1) + '\u03C3 deviation. Adjust the Scale length sliders to improve.';
            html += '</div>';
            html += '</div>';
        }

        html += '</div>';
    }

    multiBody.innerHTML = html;
    multiDiv.style.display = 'block';

    // Initialize band label for current method
    updateBandLabel();
}

/**
 * Clear inference-specific chart elements (band + markers).
 */
function clearInferenceChart() {
    if (chart.data.datasets.length > 6) {
        chart.data.datasets[6].data = [];
    }
    lastMultiResult = null;
    lastModelTotal = 0;
    chart.update('none');
}

// =====================================================================
// CHART UPDATE (calls API)
// =====================================================================

function debouncedUpdateChart() {
    clearTimeout(updateTimer);
    updateTimer = setTimeout(updateChart, DEBOUNCE_MS);
}

async function updateChart() {
    const maxRadius = parseFloat(distanceSlider.value);
    const accelRatio = parseFloat(accelSlider.value);
    var lastApiData = null; // local ref, also stored as lastApiResponse

    if (currentMode === 'inference') {
        const massModel = getMassModelFromSliders();

        try {
            var chartMaxR = maxRadius;
            var predObs = pinnedObservations || (currentExample ? getPredictionObservations(currentExample) : null);
            if (predObs && predObs.length > 0) {
                var maxObsR = Math.max.apply(null, predObs.map(function(o) { return o.r; }));
                chartMaxR = Math.max(chartMaxR, maxObsR * 1.15);
            }
            // In observation mode, extend chart to show the field horizon.
            // Use derived R_env if available, otherwise the slider value.
            var horizonR = (inferredFieldGeometry && inferredFieldGeometry.envelope_radius_kpc)
                ? inferredFieldGeometry.envelope_radius_kpc
                : getGalacticRadius();
            if (horizonR) {
                chartMaxR = Math.max(chartMaxR, horizonR * 1.15);
            }

            // Run inference only when entering Observation mode.
            // After that, use prediction endpoint so the optimizer
            // doesn't overwrite user slider adjustments.
            // In observation mode, use the derived R_env (from
            // solve_field_geometry) so the prediction endpoint's sigma
            // stage stays consistent with the inference result.
            var effectiveGR = (isAutoFitted
                && inferredFieldGeometry
                && inferredFieldGeometry.envelope_radius_kpc)
                ? inferredFieldGeometry.envelope_radius_kpc
                : getGalacticRadius();
            var data;
            if (inferenceNeeded) {
                inferenceNeeded = false;
                data = await fetchInferRotationCurve(chartMaxR, accelRatio, massModel, predObs, effectiveGR);
            } else {
                data = await fetchRotationCurve(chartMaxR, accelRatio, massModel, predObs, effectiveGR);
            }
            lastApiData = data;
            lastApiResponse = data;
            if (data.field_geometry) {
                inferredFieldGeometry = data.field_geometry;
            }
            renderCurves(data);
            syncThroughputFromResponse(data);

            // Show observation points
            var obsEnabled = isChipEnabled('observed');
            var visibleObs = predObs;
            if (visibleObs && visibleObs.length > 0) {
                chart.data.datasets[3].data = visibleObs.map(function(obs) {
                    return {x: obs.r, y: obs.v, err: obs.err || 0};
                });
                chart.data.datasets[3].hidden = !obsEnabled;
                observedLegend.style.display = obsEnabled ? 'flex' : 'none';
            } else if (currentExample) {
                chart.data.datasets[3].data = [{x: r_obs, y: v_obs}];
                chart.data.datasets[3].hidden = !obsEnabled;
                observedLegend.style.display = obsEnabled ? 'flex' : 'none';
            } else {
                chart.data.datasets[3].data = [];
                chart.data.datasets[3].hidden = true;
                observedLegend.style.display = 'none';
            }

            // Multi-point consistency analysis (fire-and-forget)
            runMultiPointInference(accelRatio, massModel);

        } catch (err) {
            console.error('Auto Fit API error:', err);
        }
    } else {
        // Explore mode: use distributed mass model from sliders
        const massModel = getMassModelFromSliders();

        try {
            // Pass observations for CDM halo fitting when available
            var predObs = pinnedObservations || (currentExample ? currentExample.observations : null);
            var predMaxR = maxRadius;
            if (predObs && predObs.length > 0) {
                var maxObsR = Math.max.apply(null, predObs.map(function(o) { return o.r; }));
                predMaxR = Math.max(predMaxR, maxObsR * 1.15);
            }
            const data = await fetchRotationCurve(predMaxR, accelRatio, massModel, predObs, getGalacticRadius());
            lastApiData = data;
            lastApiResponse = data;
            renderCurves(data);
            syncThroughputFromResponse(data);

            // Handle observed data
            var visibleObs = pinnedObservations || (currentExample ? currentExample.observations : null);
            var obsEnabled = isChipEnabled('observed');
            if (visibleObs && visibleObs.length > 0) {
                chart.data.datasets[3].data = visibleObs.map(obs => ({
                    x: obs.r, y: obs.v, err: obs.err || 0
                }));
                chart.data.datasets[3].hidden = !obsEnabled;
                observedLegend.style.display = obsEnabled ? 'flex' : 'none';
            } else {
                chart.data.datasets[3].data = [];
                chart.data.datasets[3].hidden = true;
                observedLegend.style.display = 'none';
            }
        } catch (err) {
            console.error('Rotation curve API error:', err);
        }
    }

    chart.update('none');
    updateRightPanel(lastApiData);
}

function renderCurves(data) {
    const newtonianData = [];
    const dtgData = [];
    const mondData = [];
    const cdmData = [];
    const gfdSymmetricData = [];

    for (let i = 0; i < data.radii.length; i++) {
        newtonianData.push({x: data.radii[i], y: data.newtonian[i]});
        dtgData.push({x: data.radii[i], y: data.dtg[i]});
        mondData.push({x: data.radii[i], y: data.mond[i]});
        if (data.cdm) {
            cdmData.push({x: data.radii[i], y: data.cdm[i]});
        }
        if (data.gfd_symmetric) {
            gfdSymmetricData.push({x: data.radii[i], y: data.gfd_symmetric[i]});
        }
    }

    chart.data.labels = [];

    // When sandbox data is loaded, it owns datasets 1 and 8; still fill
    // Newton/MOND/CDM from API so researchers can toggle them on.
    if (sandboxResult) {
        chart.data.datasets[0].data = newtonianData;
        chart.data.datasets[2].data = mondData;
        chart.data.datasets[7].data = cdmData;
        chart.data.datasets[0].hidden = !isChipEnabled('newtonian');
        chart.data.datasets[2].hidden = !isChipEnabled('mond');
        chart.data.datasets[7].hidden = !isChipEnabled('cdm');
        chart.data.datasets[4].data = [];
        chart.data.datasets[5].data = [];
        chart.data.datasets[9].data = [];
    } else {
        chart.data.datasets[0].data = newtonianData;
        chart.data.datasets[1].data = dtgData;
        chart.data.datasets[2].data = mondData;
        chart.data.datasets[7].data = cdmData;
        chart.data.datasets[0].hidden = !isChipEnabled('newtonian');
        chart.data.datasets[2].hidden = !isChipEnabled('mond');
        chart.data.datasets[7].hidden = !isChipEnabled('cdm');
        chart.data.datasets[8].data = [];
        chart.data.datasets[9].data = gfdSymmetricData;

        if (dtgData.length > 0) {
            var pct = parseFloat(lensSlider.value) / 100.0;
            var bandUpper = [];
            var bandLower = [];
            for (var i = 0; i < dtgData.length; i++) {
                var v = dtgData[i].y;
                var r = dtgData[i].x;
                bandUpper.push({x: r, y: v * (1 + pct)});
                bandLower.push({x: r, y: v * (1 - pct)});
            }
            chart.data.datasets[4].data = bandUpper;
            chart.data.datasets[5].data = bandLower;
        }
    }

    // Store CDM halo info for display
    lastCdmHalo = data.cdm_halo || null;
    updateCdmHaloPanel();

    // Update right metrics panel with live data
    updateMetricsPanel(data);
}

function updateCdmHaloPanel() {
    var panel = document.getElementById('cdm-halo-info');
    var details = document.getElementById('cdm-halo-details');
    var method = document.getElementById('cdm-fit-method');
    var paramCount = document.getElementById('cdm-param-count');
    if (!panel || !details) return;

    if (!lastCdmHalo || !chart.data.datasets[7].data.length) {
        panel.style.display = 'none';
        return;
    }

    var h = lastCdmHalo;
    var m200 = h.m200;
    var m200Exp = Math.floor(Math.log10(m200));
    var m200Coeff = m200 / Math.pow(10, m200Exp);

    var html = '';
    html += '<div><strong style="color:#e0e0e0;">Halo Mass M<sub>200</sub>:</strong> ';
    html += '<span style="color:#ffffff;">' + m200Coeff.toFixed(2) + ' \u00D7 10' + superscript(m200Exp) + ' M\u2609</span></div>';
    html += '<div><strong style="color:#e0e0e0;">Concentration:</strong> c = ' + h.c + '</div>';
    if (h.r200_kpc) {
        html += '<div><strong style="color:#e0e0e0;">Virial Radius:</strong> r<sub>200</sub> = ' + h.r200_kpc + ' kpc</div>';
    }
    if (h.chi2_reduced !== undefined) {
        html += '<div><strong style="color:#e0e0e0;">Reduced \u03C7\u00B2:</strong> ' + h.chi2_reduced + '</div>';
    }

    details.innerHTML = html;
    method.textContent = h.method || '';
    if (paramCount) {
        paramCount.textContent = h.n_params_total || 2;
    }
    panel.style.display = 'block';
}

// =====================================================================
// RIGHT METRICS PANEL
// =====================================================================

/**
 * Linearly interpolate the GFD curve value at a given radius.
 * gfdData is an array of {x, y} sorted by x.
 */
function interpolateGfdAt(r, gfdData) {
    if (!gfdData || gfdData.length === 0) return null;
    if (r <= gfdData[0].x) return gfdData[0].y;
    if (r >= gfdData[gfdData.length - 1].x) return gfdData[gfdData.length - 1].y;
    for (var i = 1; i < gfdData.length; i++) {
        if (gfdData[i].x >= r) {
            var x0 = gfdData[i - 1].x, y0 = gfdData[i - 1].y;
            var x1 = gfdData[i].x,     y1 = gfdData[i].y;
            var t = (r - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    return null;
}

/**
 * Compute fit quality metrics from observations vs GFD curve.
 * Returns {rms, chi2r, within1s, within2s, nObs, residuals[]}.
 */
function computeFitMetrics(observations, gfdData) {
    if (!observations || observations.length === 0 || !gfdData || gfdData.length === 0) {
        return null;
    }

    var sumSq = 0;
    var chi2 = 0;
    var n = 0;
    var w1 = 0, w2 = 0;
    var residuals = [];

    for (var i = 0; i < observations.length; i++) {
        var obs = observations[i];
        var vGfd = interpolateGfdAt(obs.r, gfdData);
        if (vGfd === null) continue;

        var dv = obs.v - vGfd;
        var err = obs.err || 0;
        var sigDev = (err > 0) ? Math.abs(dv) / err : null;

        sumSq += dv * dv;
        if (err > 0) {
            chi2 += (dv * dv) / (err * err);
        }
        n++;

        if (sigDev !== null) {
            if (sigDev <= 1) w1++;
            if (sigDev <= 2) w2++;
        }

        residuals.push({
            r: obs.r,
            vObs: obs.v,
            vGfd: vGfd,
            dv: dv,
            sigma: sigDev
        });
    }

    if (n === 0) return null;

    var dof = Math.max(n - 1, 1);
    return {
        rms: Math.sqrt(sumSq / n),
        chi2r: chi2 / dof,
        within1s: w1,
        within2s: w2,
        nObs: n,
        residuals: residuals
    };
}

/**
 * Format a number in scientific notation like "1.23e10".
 */
function fmtSci(val) {
    if (!val || val === 0) return '--';
    var exp = Math.floor(Math.log10(Math.abs(val)));
    var coeff = val / Math.pow(10, exp);
    return coeff.toFixed(2) + 'e' + exp;
}

/**
 * Toggle a metrics section open/closed.
 */
function toggleMetricsSection(headerEl) {
    var section = headerEl.parentElement;
    var body = headerEl.nextElementSibling;
    var chevron = headerEl.querySelector('.metrics-chevron');
    if (!body) return;

    if (body.style.display === 'none') {
        body.style.display = '';
        section.classList.remove('collapsed');
        if (chevron) chevron.innerHTML = '&#9660;';
    } else {
        body.style.display = 'none';
        section.classList.add('collapsed');
        if (chevron) chevron.innerHTML = '&#9654;';
    }
}

/**
 * Update the right metrics panel with live data.
 * Called after every renderCurves() and chart update.
 */
function updateMetricsPanel(apiData) {
    var gfdData = chart.data.datasets[1].data;
    var obs = pinnedObservations || (currentExample ? currentExample.observations : null);

    // --- Fit Quality ---
    var rmsEl = document.getElementById('metric-rms');
    var chi2El = document.getElementById('metric-chi2');
    var hits1El = document.getElementById('metric-hits-1s');
    var hits2El = document.getElementById('metric-hits-2s');

    var metrics = computeFitMetrics(obs, gfdData);
    if (metrics) {
        rmsEl.textContent = metrics.rms.toFixed(1) + ' km/s';
        rmsEl.className = 'metrics-value' + (metrics.rms < 10 ? ' good' : metrics.rms < 25 ? ' warn' : ' bad');

        chi2El.textContent = metrics.chi2r.toFixed(2);
        chi2El.className = 'metrics-value' + (metrics.chi2r < 1.5 ? ' good' : metrics.chi2r < 3 ? ' warn' : ' bad');

        hits1El.textContent = metrics.within1s + '/' + metrics.nObs;
        hits1El.className = 'metrics-value' + (metrics.within1s / metrics.nObs >= 0.68 ? ' good' : ' warn');

        hits2El.textContent = metrics.within2s + '/' + metrics.nObs;
        hits2El.className = 'metrics-value' + (metrics.within2s / metrics.nObs >= 0.95 ? ' good' : ' warn');
    } else {
        rmsEl.textContent = '--';  rmsEl.className = 'metrics-value';
        chi2El.textContent = '--'; chi2El.className = 'metrics-value';
        hits1El.textContent = '--'; hits1El.className = 'metrics-value';
        hits2El.textContent = '--'; hits2El.className = 'metrics-value';
    }

    // --- Observation Summary ---
    var npEl = document.getElementById('metric-npoints');
    var rangeEl = document.getElementById('metric-radial-range');
    var errEl = document.getElementById('metric-mean-err');

    if (obs && obs.length > 0) {
        npEl.textContent = obs.length;
        var radii = obs.map(function(o) { return o.r; });
        rangeEl.textContent = Math.min.apply(null, radii).toFixed(1) + ' - ' + Math.max.apply(null, radii).toFixed(1) + ' kpc';
        var meanErr = obs.reduce(function(s, o) { return s + (o.err || 0); }, 0) / obs.length;
        errEl.textContent = '+/- ' + meanErr.toFixed(1) + ' km/s';
    } else {
        npEl.textContent = '--';
        rangeEl.textContent = '--';
        errEl.textContent = '--';
    }

    // --- Mass Model ---
    var mm = getMassModelFromSliders();
    var totalM = mm.bulge.M + mm.disk.M + mm.gas.M;
    var gasFrac = totalM > 0 ? (mm.gas.M / totalM * 100) : 0;

    document.getElementById('metric-total-mass').textContent = fmtSci(totalM) + ' M_sun';
    document.getElementById('metric-gas-frac').textContent = gasFrac.toFixed(1) + '%';
    // NOTE: metric-field-origin and metric-field-horizon are written by
    // updateRightPanel() which runs after this function and correctly
    // uses inferredFieldGeometry when in observation mode.

    // --- CDM Comparison ---
    var m200El = document.getElementById('metric-cdm-m200');
    var cEl = document.getElementById('metric-cdm-c');
    var r200El = document.getElementById('metric-cdm-r200');
    var methodEl = document.getElementById('metric-cdm-method');
    var paramsEl = document.getElementById('metric-cdm-params');

    if (lastCdmHalo) {
        m200El.textContent = fmtSci(lastCdmHalo.m200) + ' M_sun';
        cEl.textContent = lastCdmHalo.c;
        r200El.textContent = lastCdmHalo.r200_kpc ? (lastCdmHalo.r200_kpc + ' kpc') : '--';
        var shortMethod = lastCdmHalo.method || '';
        if (shortMethod.indexOf('abundance') >= 0) shortMethod = 'Abundance matching';
        else if (shortMethod.indexOf('chi') >= 0) shortMethod = 'Chi-sq fit';
        methodEl.textContent = shortMethod;
        paramsEl.textContent = lastCdmHalo.n_params_total || 2;
    } else {
        m200El.textContent = '--';
        cEl.textContent = '--';
        r200El.textContent = '--';
        methodEl.textContent = '--';
        paramsEl.textContent = '--';
    }

    // --- Residuals Table ---
    var tbody = document.getElementById('metrics-residuals-tbody');
    if (tbody && metrics && metrics.residuals.length > 0) {
        var html = '';
        for (var i = 0; i < metrics.residuals.length; i++) {
            var res = metrics.residuals[i];
            var rowClass = '';
            if (res.sigma !== null) {
                if (res.sigma <= 1) rowClass = 'sigma-good';
                else if (res.sigma <= 2) rowClass = 'sigma-warn';
                else rowClass = 'sigma-bad';
            }
            html += '<tr class="' + rowClass + '">';
            html += '<td>' + res.r.toFixed(1) + '</td>';
            html += '<td>' + res.vObs.toFixed(0) + '</td>';
            html += '<td>' + res.vGfd.toFixed(0) + '</td>';
            html += '<td>' + (res.dv >= 0 ? '+' : '') + res.dv.toFixed(1) + '</td>';
            html += '<td>' + (res.sigma !== null ? res.sigma.toFixed(1) : '--') + '</td>';
            html += '</tr>';
        }
        tbody.innerHTML = html;
    } else if (tbody) {
        tbody.innerHTML = '';
    }
}

// =====================================================================
// DISPLAY UPDATES
// =====================================================================

function updateDisplays() {
    const maxRadius = parseFloat(distanceSlider.value);
    const logMass = parseFloat(massSlider.value);
    const accelRatio = parseFloat(accelSlider.value);
    const velocity = parseFloat(velocitySlider.value);

    distanceValue.textContent = maxRadius + ' kpc';

    const massExponent = Math.floor(logMass);
    const massCoeff = Math.pow(10, logMass - massExponent);
    massValue.textContent = massCoeff.toFixed(1) + '\u00D7' + '10' + superscript(massExponent) + ' M\u2609';

    velocityValue.textContent = velocity + ' km/s';
    accelValue.textContent = 'a/a\u2080 = ' + accelRatio.toFixed(2);
    galacticRadiusValue.textContent = galacticRadiusSlider.value + ' kpc';
}

// =====================================================================
// EXAMPLES
// =====================================================================

function updateExamplesDropdown() {
    const dropdown = document.getElementById('examples-dropdown');
    dropdown.innerHTML = '';

    // Add placeholder
    const placeholder = document.createElement('option');
    placeholder.value = '0';
    placeholder.textContent = 'Select a galaxy...';
    dropdown.appendChild(placeholder);

    // Always use prediction galaxies (unified observation flow)
    const galaxies = galaxyCatalog.prediction || [];
    galaxies.forEach((galaxy, index) => {
        const option = document.createElement('option');
        option.value = String(index + 1);
        option.textContent = galaxy.name;
        dropdown.appendChild(option);
    });

    dropdown.selectedIndex = 0;
}

function loadExample() {
    const dropdown = document.getElementById('examples-dropdown');
    const selectedIndex = parseInt(dropdown.value, 10);

    if (selectedIndex === 0) {
        currentExample = null;
        pinnedObservations = null;
        pinnedGalaxyLabel = null;
        pinnedGalaxyExample = null;
        sandboxResult = null;
        navigateTo('charts', 'chart');
        chart.options.plugins.title.text = 'Rotation Curve: Gravitational Theory Comparison';
        chart.options.plugins.zoom.limits.x.min = 0;
        chart.options.plugins.zoom.limits.x.max = 100;
        chart.resetZoom();
        updateChart();
        return;
    }

    isLoadingExample = true;

    // Always use prediction galaxies (unified observation flow)
    const galaxies = galaxyCatalog.prediction || [];
    const example = galaxies[selectedIndex - 1];
    if (!example) { isLoadingExample = false; return; }

    // Reset to Mass Model mode when changing galaxy
    isAutoFitted = false;
    inferredFieldGeometry = null;
    currentMode = 'prediction';
    document.querySelector('.mass-model-header-text').textContent = 'Mass Distribution';
    hideAutoMapDiagnostics();
    // Reset inference-mode UI
    velocityControl.style.display = 'none';
    inferenceResult.classList.remove('visible');
    var multiDiv = document.getElementById('multi-inference-result');
    if (multiDiv) multiDiv.style.display = 'none';
    setMassSliderEditable(true);
    unlockAllChips();
    updateBandLabel();

    // Extract galaxy display name (strip mass info in parentheses)
    const galaxyLabel = example.name.replace(/\s*\(.*\)/, '');
    chart.options.plugins.title.text = galaxyLabel + ': GFD Velocity Curves';

    chart.data.datasets[3].data = [];
    chart.data.datasets[3].hidden = true;
    clearInferenceChart();

    currentExample = example;

    // Pin observations so they survive slider adjustments
    var pinObs = example.observations;
    pinnedObservations = pinObs || null;
    pinnedGalaxyLabel = galaxyLabel;
    pinnedGalaxyExample = example;

    accelSlider.value = example.accel;
    distanceSlider.value = example.distance;
    anchorRadiusInput.value = example.distance;
    massSlider.value = example.mass || 11;

    navigateTo('charts', 'chart');

    // Set zoom limits based on observational data or distance
    var obsData = example.observations;
    if (obsData && obsData.length > 0) {
        const obsRadii = obsData.map(obs => obs.r);
        const minR = Math.min(...obsRadii);
        const maxR = Math.max(...obsRadii);
        const range = maxR - minR;
        const padding = Math.max(5, range * 0.2);
        chart.options.plugins.zoom.limits.x.min = Math.max(0, minR - padding);
        chart.options.plugins.zoom.limits.x.max = Math.min(100, maxR + padding);
    } else {
        chart.options.plugins.zoom.limits.x.min = 0;
        chart.options.plugins.zoom.limits.x.max = 100;
    }

    // Sync mass distribution panel
    if (example.mass_model) {
        setMassModelSliders(example.mass_model);
    }
    updateMassModelDisplays();

    // Sync galactic radius slider
    if (example.galactic_radius) {
        galacticRadiusSlider.value = example.galactic_radius;
        galacticRadiusValue.textContent = example.galactic_radius + ' kpc';
    }

    // Set to auto mode: the backend computes Origin Throughput from
    // gas leverage. No jumping since the first (and only) render uses
    // the auto value. The slider syncs from the API response.
    isAutoThroughput = true;
    inferenceNeeded = false;
    vortexAutoBtn.classList.add('active');
    vortexStrengthValue.textContent = 'auto';

    massModelManuallyModified = false;
    updateResetToPhotometricButton();
    updateDisplays();

    // Render observations directly onto the chart
    var obsEnabled = isChipEnabled('observed');
    if (pinnedObservations && pinnedObservations.length > 0) {
        chart.data.datasets[3].data = pinnedObservations.map(function(obs) {
            return { x: obs.r, y: obs.v, err: obs.err || 0 };
        });
        chart.data.datasets[3].hidden = !obsEnabled;
        chart.data.datasets[3].errorBars = pinnedObservations.map(function(obs) {
            return obs.err || 0;
        });
    }

    // Phase 1: fast photometric (mass model only, instant)
    // Phase 2: observation fits (GFD Sigma + Accel, ~2-3s, background)
    // Also fetch rotation curve for Newton/MOND/CDM so researchers can toggle them on.
    if (example.id) {
        var massModel = getMassModelFromSliders();
        var predObs = example.observations || [];
        var predMaxR = parseFloat(distanceSlider.value);
        if (predObs.length > 0) {
            var maxObsR = Math.max.apply(null, predObs.map(function(o) { return o.r; }));
            predMaxR = Math.max(predMaxR, maxObsR * 1.15);
        }
        var gr = example.galactic_radius ? parseFloat(example.galactic_radius) : parseFloat(distanceSlider.value);
        fetchPhotometricData(example.id).then(function() {
            hideResetButton();
            isLoadingExample = false;
            massModelManuallyModified = false;
            updateResetToPhotometricButton();
            fetchObservationData(example.id);
            fetchRotationCurve(predMaxR, parseFloat(accelSlider.value), massModel, predObs, gr).then(function(data) {
                var i;
                var newtonianData = [];
                var mondData = [];
                var cdmData = [];
                for (i = 0; i < (data.radii || []).length; i++) {
                    newtonianData.push({ x: data.radii[i], y: data.newtonian[i] });
                    mondData.push({ x: data.radii[i], y: data.mond[i] });
                    if (data.cdm) cdmData.push({ x: data.radii[i], y: data.cdm[i] });
                }
                chart.data.datasets[0].data = newtonianData;
                chart.data.datasets[2].data = mondData;
                chart.data.datasets[7].data = cdmData;
                chart.data.datasets[0].hidden = !isChipEnabled('newtonian');
                chart.data.datasets[2].hidden = !isChipEnabled('mond');
                chart.data.datasets[7].hidden = !isChipEnabled('cdm');
                if (data.cdm_halo) lastCdmHalo = data.cdm_halo;
                chart.update('none');
            }).catch(function() {});
        });
    } else {
        chart.update('none');
        chart.resetZoom();
        hideResetButton();
        isLoadingExample = false;
    }
}

// =====================================================================
// ANALYSIS VIEW STATE AND NAVIGATION PIPELINE
// =====================================================================
// Single source of truth for what is shown in left panel, right panel (tabs + content).
// All tab clicks and programmatic view changes go through navigateTo().
//
// Contract: for any (rightPanelTab, chartsSubmenuTab),
//   - Left panel:  mass-model-panel only for Charts+Mass model; else observation-mass-panel (read-only).
//   - Right panel: one of [Charts tab content | Chart Data tab content].
//   - If Charts: submenu visible when galaxy loaded; one of [chart container | vortex face | field-analysis face].
//   - Chart state (GFD chips, VPS, datasets) is applied by applyMassModelChartState or applyObservationChartState.
// Adding a new right-panel tab or Charts submenu: extend navigateTo() and analysisViewState.
// =====================================================================

/** Current analysis view: right-panel tab and Charts submenu tab. */
var analysisViewState = {
    rightPanelTab: 'charts',
    chartsSubmenuTab: 'chart'
};

/**
 * Navigate to a view. Single pipeline for tab switches; determines left panel,
 * right panel tab content, and chart/submenu state. Call this from tab clicks
 * and when loading or clearing a galaxy.
 * @param {string} rightPanelTab - 'charts' | 'chart-data'
 * @param {string} [chartsSubmenuTab] - 'chart' | 'obs-chart' | 'vortex' | 'field-analysis' (only when rightPanelTab === 'charts')
 */
function navigateTo(rightPanelTab, chartsSubmenuTab) {
    if (!rightPanelTab) return;
    if (rightPanelTab === 'charts' && chartsSubmenuTab) {
        analysisViewState.rightPanelTab = 'charts';
        analysisViewState.chartsSubmenuTab = chartsSubmenuTab;
    } else if (rightPanelTab === 'chart-data') {
        analysisViewState.rightPanelTab = 'chart-data';
        analysisViewState.chartsSubmenuTab = 'chart';
    } else {
        analysisViewState.rightPanelTab = rightPanelTab;
        analysisViewState.chartsSubmenuTab = chartsSubmenuTab || 'chart';
    }

    var rpTab = analysisViewState.rightPanelTab;
    var csTab = analysisViewState.chartsSubmenuTab;

    var massModelPanel = document.getElementById('mass-model-panel');
    var obsMassPanel = document.getElementById('observation-mass-panel');
    var rightPanel = document.querySelector('.right-panel');
    var tabBar = document.getElementById('obs-tab-bar');
    var rightPanelTabs = document.getElementById('right-panel-tabs');
    var contents = document.querySelectorAll('.right-panel-tab-content');
    var theoryToggles = document.getElementById('theory-toggles');
    var chartContainer = document.querySelector('.chart-container');
    var obsFace = document.getElementById('obs-data-face');
    var vortexFace = document.getElementById('vortex-face');
    var fieldFace = document.getElementById('field-analysis-face');

    // ----- Left panel -----
    if (massModelPanel && obsMassPanel) {
        if (rpTab === 'charts' && csTab === 'chart') {
            massModelPanel.style.display = '';
            obsMassPanel.style.display = 'none';
        } else {
            obsMassPanel.style.display = '';
            massModelPanel.style.display = 'none';
            if (rpTab === 'charts') populateObservationMassPanel();
        }
    }

    // ----- Right panel: which tab content is visible -----
    if (rightPanelTabs) {
        var tabs = rightPanelTabs.querySelectorAll('.right-panel-tab');
        tabs.forEach(function(t) {
            t.classList.toggle('active', t.getAttribute('data-panel-tab') === rpTab);
        });
    }
    contents.forEach(function(el) {
        var contentId = el.getAttribute('data-tab-content');
        el.style.display = contentId === rpTab ? '' : 'none';
    });

    // ----- Right panel: Charts submenu and main content -----
    var hasGalaxy = !!(currentExample || pinnedGalaxyExample || (pinnedObservations && pinnedObservations.length > 0));
    if (tabBar) {
        tabBar.style.display = rpTab === 'charts' && hasGalaxy ? '' : 'none';
        if (rpTab === 'charts') {
            var subTabs = tabBar.querySelectorAll('.obs-tab');
            subTabs.forEach(function(t) {
                t.classList.toggle('active', t.getAttribute('data-tab') === csTab);
            });
        }
    }

    if (theoryToggles) theoryToggles.style.display = 'none';
    if (chartContainer) chartContainer.style.display = 'none';
    if (obsFace) obsFace.style.display = 'none';
    if (vortexFace) vortexFace.style.display = 'none';
    if (fieldFace) fieldFace.style.display = 'none';
    if (rightPanel) rightPanel.classList.remove('obs-data-active');

    if (rpTab === 'chart-data') {
        if (rightPanel) rightPanel.classList.add('obs-data-active');
        var chartDataFace = document.getElementById('chart-data-face');
        if (chartDataFace) chartDataFace.style.display = '';
        renderChartDataTab();
        return;
    }

    // rpTab === 'charts': show the active Charts submenu content
    if (csTab === 'chart') {
        if (theoryToggles) theoryToggles.style.display = '';
        if (chartContainer) chartContainer.style.display = '';
        applyMassModelChartState();
    } else if (csTab === 'obs-chart') {
        if (theoryToggles) theoryToggles.style.display = '';
        if (chartContainer) chartContainer.style.display = '';
        applyObservationChartState();
    } else if (csTab === 'vortex') {
        if (vortexFace) vortexFace.style.display = '';
        if (rightPanel) rightPanel.classList.add('obs-data-active');
        loadVortexTab(vortexCurrentVariance);
    } else if (csTab === 'field-analysis') {
        if (fieldFace) fieldFace.style.display = '';
        if (rightPanel) rightPanel.classList.add('obs-data-active');
        loadFieldAnalysis();
    }
}

/**
 * Chart-only state for Mass Model view (GFD Photometric, no VPS). No panel visibility.
 */
function applyMassModelChartState() {
    isAutoFitted = false;
    var vpsPanel = document.getElementById('vps-panel');
    if (vpsPanel) vpsPanel.style.display = 'none';
    updateResetToPhotometricButton();
    var accelChip = document.querySelector('.theory-chip[data-series="gfd_accel"]');
    if (accelChip) accelChip.style.display = 'none';
    if (typeof chart !== 'undefined' && chart && chart.data && chart.data.datasets) {
        chart.data.datasets[8].hidden = true;
        chart.data.datasets[10].hidden = true;
        chart.data.datasets[11].hidden = true;
    }
    blankRightPaneMetrics();
    renderSandboxCurves();
}

/**
 * Chart-only state for Observations view (GFD Observed, VPS). No panel visibility.
 */
async function applyObservationChartState() {
    isAutoFitted = true;
    var accelChip = document.querySelector('.theory-chip[data-series="gfd_accel"]');
    if (accelChip) accelChip.style.display = '';
    var hasObsData = sandboxResult && sandboxResult.chart &&
        sandboxResult.chart.gfd_spline &&
        sandboxResult.chart.gfd_spline.length > 0;
    if (!hasObsData && observationFetchPromise) {
        showChartLoading();
        await observationFetchPromise;
        hideChartLoading();
    }
    if (typeof chart !== 'undefined' && chart && chart.data && chart.data.datasets) {
        chart.data.datasets[10].hidden = true;
        chart.data.datasets[11].hidden = false;
    }
    var vpsPanel = document.getElementById('vps-panel');
    if (vpsPanel) vpsPanel.style.display = '';
    ensureVpsChart();
    blankRightPaneMetrics();
    renderSandboxCurves();
    updateVpsChart();
}

// =====================================================================
// VIEW MODE TOGGLE (legacy)
// =====================================================================

function initViewModeToggle() {
    // No-op: view-mode toggle removed; navigation goes through navigateTo().
}

/**
 * Top-level right-panel tabs (Charts | Chart Data). All view changes go through navigateTo().
 */
function initRightPanelTabs() {
    var tabContainer = document.getElementById('right-panel-tabs');
    if (!tabContainer) return;
    var tabs = tabContainer.querySelectorAll('.right-panel-tab');
    tabs.forEach(function(tab) {
        tab.addEventListener('click', function() {
            var panelTab = this.getAttribute('data-panel-tab');
            navigateTo(panelTab, analysisViewState.chartsSubmenuTab);
        });
    });
}

/**
 * Switch to Observation view. Delegates to central pipeline (panels + chart state).
 */
function enterObservationMode() {
    navigateTo('charts', 'obs-chart');
}

/**
 * Switch to Mass Model view. Delegates to central pipeline (panels + chart state).
 */
function enterMassModelMode() {
    navigateTo('charts', 'chart');
}

/**
 * Legacy entry point kept for backward compatibility (global onclick).
 * Toggles between Mass Model and Observation modes.
 */
function runAutoFit() {
    if (isAutoFitted) {
        enterMassModelMode();
    } else {
        enterObservationMode();
    }
}

/**
 * Reset the view-mode toggle to Mass Model (default). Called when
 * loading a new galaxy or when sliders are manually adjusted.
 */
function resetViewModeToggle() {
    navigateTo('charts', 'chart');
}

// =====================================================================
// OBSERVATION DATA TABS (Chart | Observation Data)
// =====================================================================

/**
 * Charts submenu (Mass model | Observations | Vortex). All view changes go through navigateTo().
 */
function initObsTabs() {
    var tabBar = document.getElementById('obs-tab-bar');
    if (!tabBar) return;
    var tabs = tabBar.querySelectorAll('.obs-tab');
    tabs.forEach(function(tab) {
        tab.addEventListener('click', function() {
            var target = this.getAttribute('data-tab');
            navigateTo('charts', target);
        });
    });
}

function isVortexChipEnabled(seriesKey) {
    var face = document.getElementById('vortex-face');
    if (!face) return true;
    var chip = face.querySelector('.theory-chip[data-series="' + seriesKey + '"]');
    if (!chip) return true;
    var cb = chip.querySelector('input[type="checkbox"]');
    return cb ? cb.checked : true;
}

var vortexChipToDatasetIndex = { vortex_raw: 0, vortex_smooth: 1, vortex_observed: 2, vortex_mond: 3, vortex_cdm: 4, vortex_gfd: 5 };

function initVortexChipListeners() {
    var container = document.getElementById('vortex-theory-toggles');
    if (!container) return;
    var bound = container.getAttribute('data-vortex-listeners');
    if (bound === '1') return;
    container.setAttribute('data-vortex-listeners', '1');
    container.addEventListener('click', function(ev) {
        var chip = ev.target.closest('.theory-chip[data-series^="vortex_"]');
        if (!chip) return;
        ev.preventDefault();
        ev.stopPropagation();
        var key = chip.getAttribute('data-series');
        var cb = chip.querySelector('input[type="checkbox"]');
        var dsIdx = vortexChipToDatasetIndex[key];
        if (!cb || dsIdx === undefined) return;
        cb.checked = !cb.checked;
        chip.classList.toggle('active', cb.checked);
        if (vortexChartAInstance && vortexChartAInstance.data && vortexChartAInstance.data.datasets[dsIdx]) {
            vortexChartAInstance.data.datasets[dsIdx].hidden = !cb.checked;
            vortexChartAInstance.update('none');
        }
    }, true);
}

function syncVortexChipActiveState() {
    var face = document.getElementById('vortex-face');
    if (!face) return;
    var chips = face.querySelectorAll('.theory-chip[data-series^="vortex_"]');
    for (var i = 0; i < chips.length; i++) {
        var chip = chips[i];
        var cb = chip.querySelector('input[type="checkbox"]');
        if (cb) chip.classList.toggle('active', cb.checked);
    }
}

/**
 * Load Vortex tab data (figure-a, figure-b, and bridged theory curves).
 * Fetches vortex-mode rotation and photometric so GFD (Photometric), MOND, CDM
 * are symmetric (no spurious dip for r < 0).
 */
function loadVortexTab(variancePct) {
    var example = pinnedGalaxyExample || currentExample;
    var galaxyId = example && example.id ? example.id : null;
    if (!galaxyId) {
        lastVortexRotationData = null;
        lastVortexPhotometricData = null;
        renderVortexCharts(null, null);
        return;
    }
    var body = JSON.stringify({ galaxy_id: galaxyId, variance_pct: variancePct });
    var massModel = getMassModelFromSliders();
    var predObs = pinnedObservations || (example && example.observations) || [];
    var predMaxR = parseFloat(distanceSlider.value);
    if (predObs.length > 0) {
        var maxObsR = Math.max.apply(null, predObs.map(function(o) { return o.r; }));
        predMaxR = Math.max(predMaxR, maxObsR * 1.15);
    }
    var gr = (example && example.galactic_radius) ? parseFloat(example.galactic_radius) : parseFloat(distanceSlider.value);
    var accelRatio = parseFloat(accelSlider.value);

    Promise.all([
        fetch('/api/vortex/figure-a', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body }).then(function(r) { return r.ok ? r.json() : null; }),
        fetch('/api/vortex/figure-b', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body }).then(function(r) { return r.ok ? r.json() : null; }),
        fetchRotationCurve(predMaxR, accelRatio, massModel, predObs, gr, 'default').then(function(d) { return d; }).catch(function() { return null; }),
        fetchPhotometricData(galaxyId, 'vortex_chart', predMaxR).then(function(d) { return d; }).catch(function() { return null; })
    ]).then(function(results) {
        lastVortexRotationData = results[2] || null;
        lastVortexPhotometricData = results[3] || null;
        renderVortexCharts(results[0], results[1]);
    }).catch(function() {
        lastVortexRotationData = null;
        lastVortexPhotometricData = null;
        renderVortexCharts(null, null);
    });
}

/**
 * Render or clear Figure 1a and Figure 1b vortex charts.
 * figA: { figure_a: { chart1, chart2 } }; figB: { figure_b: { chart1, chart2 } }.
 */
function renderVortexCharts(figA, figB) {
    var canvasA = document.getElementById('vortexChartA');
    var canvasB = document.getElementById('vortexChartB');
    if (!canvasA || !canvasB) return;

    if (vortexChartAInstance) {
        vortexChartAInstance.destroy();
        vortexChartAInstance = null;
    }
    if (vortexChartBInstance) {
        vortexChartBInstance.destroy();
        vortexChartBInstance = null;
    }

    if (!figA || !figA.figure_a || !figB || !figB.figure_b) {
        if (vortexChartAInstance) { vortexChartAInstance.destroy(); vortexChartAInstance = null; }
        if (vortexChartBInstance) { vortexChartBInstance.destroy(); vortexChartBInstance = null; }
        var msg = document.getElementById('vortex-message');
        if (msg) {
            msg.style.display = '';
            msg.textContent = 'Select an observation to load data.';
        }
        return;
    }
    var msg = document.getElementById('vortex-message');
    if (msg) msg.style.display = 'none';

    var c1 = figA.figure_a.chart1;
    var c2 = figA.figure_a.chart2;
    var radii = c1.radii || [];
    var vRaw = c1.v_raw || [];
    var vFv = c2.v_fv || [];
    var obsR = c1.obs_r || [];
    var obsV = c1.obs_v || [];
    var obsErr = c1.obs_err || [];

    var rawPoints = radii.map(function(r, i) { return { x: r, y: vRaw[i] }; });
    var fvPoints = radii.map(function(r, i) {
        var y = vFv[i];
        return (y != null && typeof y === 'number') ? { x: r, y: y } : null;
    }).filter(function(p) { return p !== null; });
    var obsPoints = obsR.map(function(r, i) { return { x: r, y: obsV[i] }; });

    function mirrorCurveForVortex(points) {
        if (!points || points.length === 0) return [];
        var out = [];
        for (var i = 0; i < points.length; i++) {
            var p = points[i];
            if (p.x > 0) out.push({ x: -p.x, y: p.y });
        }
        for (var j = 0; j < points.length; j++) out.push({ x: points[j].x, y: points[j].y });
        out.sort(function(a, b) { return a.x - b.x; });
        return out;
    }
    function seriesToPoints(radii, vals) {
        if (!radii || !vals || radii.length !== vals.length) return [];
        return radii.map(function(r, i) { return { x: r, y: vals[i] }; });
    }
    var gfdData = [];
    if (lastVortexPhotometricData && lastVortexPhotometricData.chart && lastVortexPhotometricData.chart.radii && lastVortexPhotometricData.chart.gfd_photometric) {
        var pr = lastVortexPhotometricData.chart.radii;
        var pv = lastVortexPhotometricData.chart.gfd_photometric;
        if (pr.length === pv.length && pr.length > 0) gfdData = seriesToPoints(pr, pv);
    }
    if (gfdData.length === 0 && chart && chart.data && chart.data.datasets[1] && chart.data.datasets[1].data) {
        gfdData = chart.data.datasets[1].data.slice();
    }
    var mondData = [];
    var cdmData = [];
    if (lastVortexRotationData && lastVortexRotationData.radii && lastVortexRotationData.radii.length > 0) {
        var rr = lastVortexRotationData.radii;
        if (lastVortexRotationData.mond) mondData = seriesToPoints(rr, lastVortexRotationData.mond);
        if (lastVortexRotationData.cdm) cdmData = seriesToPoints(rr, lastVortexRotationData.cdm);
    }
    if (mondData.length === 0 && chart && chart.data && chart.data.datasets[2] && chart.data.datasets[2].data) {
        mondData = mirrorCurveForVortex(chart.data.datasets[2].data);
    }
    if (cdmData.length === 0 && chart && chart.data && chart.data.datasets[7] && chart.data.datasets[7].data) {
        cdmData = mirrorCurveForVortex(chart.data.datasets[7].data);
    }

    var vortexDs = [
        { label: 'GFD (Velocity raw)', data: rawPoints, borderColor: '#888', borderDash: [4, 4], borderWidth: 2, tension: 0.3, pointRadius: 0, hidden: !isVortexChipEnabled('vortex_raw') },
        { label: 'GFD (Velocity smooth)', data: fvPoints, borderColor: '#aa44ff', borderWidth: 2, tension: 0.3, pointRadius: 0, hidden: !isVortexChipEnabled('vortex_smooth') },
        { label: 'Observed', data: obsPoints, borderColor: '#FFC107', backgroundColor: '#FFC107', pointRadius: 5, showLine: false, hidden: !isVortexChipEnabled('vortex_observed') },
        { label: 'MOND', data: mondData, borderColor: '#9966ff', borderWidth: 2, tension: 0.3, pointRadius: 0, hidden: !isVortexChipEnabled('vortex_mond') },
        { label: 'CDM', data: cdmData, borderColor: '#ffffff', borderWidth: 2, tension: 0.3, pointRadius: 0, hidden: !isVortexChipEnabled('vortex_cdm') },
        { label: 'GFD (Photometric)', data: gfdData, borderColor: 'rgba(0, 229, 160, 0.9)', backgroundColor: 'transparent', borderDash: [8, 4], borderWidth: 2, tension: 0.3, showLine: true, pointRadius: 0, pointHoverRadius: 0, pointStyle: 'circle', pointBackgroundColor: 'rgba(0, 229, 160, 0)', pointBorderColor: 'rgba(0, 229, 160, 0)', pointBorderWidth: 0, hidden: !isVortexChipEnabled('vortex_gfd') }
    ];

    vortexChartAInstance = new Chart(canvasA.getContext('2d'), {
        type: 'line',
        data: {
            datasets: vortexDs
        },
        plugins: [{
            id: 'vortexLockScale',
            afterUpdate: function(chart) {
                if (chart._vortexScaleLocked) return;
                var sx = chart.scales.x;
                var sy = chart.scales.y;
                if (!sx || !sy) return;
                chart.options.scales.x.min = sx.min;
                chart.options.scales.x.max = sx.max;
                chart.options.scales.y.min = sy.min;
                chart.options.scales.y.max = sy.max;
                chart._vortexScaleLocked = true;
            }
        }],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: false },
                title: { display: true, text: 'GFD Velocity Field: Covariant Action Applied to Observed Kinematics', color: '#e0e0e0', font: { size: 16, weight: '500' } }
            },
            scales: {
                x: { type: 'linear', title: { display: true, text: 'Galactocentric radius r (kpc)' }, grid: { color: 'rgba(64,64,64,0.3)' }, ticks: { color: '#b0b0b0' } },
                y: { type: 'linear', title: { display: true, text: 'v(r) [km/s]' }, grid: { color: 'rgba(64,64,64,0.3)' }, ticks: { color: '#b0b0b0' } }
            }
        }
    });
    initVortexChipListeners();
    syncVortexChipActiveState();

    var b1 = figB.figure_b.chart1;
    var b2 = figB.figure_b.chart2;
    var rKpc1 = b1.r_kpc || [];
    var ratio1 = b1.ratio_T || [];
    var rKpc2 = b2.r_kpc || [];
    var ratio2 = b2.ratio_T || [];

    var line1 = rKpc1.map(function(r, i) { return { x: r, y: ratio1[i] }; });
    var line2 = rKpc2.map(function(r, i) { return { x: r, y: ratio2[i] }; });

    vortexChartBInstance = new Chart(canvasB.getContext('2d'), {
        type: 'line',
        data: {
            datasets: [
                { label: 'Observed rotation curve vs GFD (Photometric mass model)', data: line1, borderColor: '#2e7d32', borderDash: [4, 4], borderWidth: 2, tension: 0.3, pointRadius: 0 },
                { label: 'GFD Velocity Field (smoothed) vs GFD (Photometric mass model)', data: line2, borderColor: '#1565c0', borderWidth: 2, tension: 0.3, pointRadius: 0 }
            ]
        },
        plugins: [{
            id: 'vortexLockScaleB',
            afterUpdate: function(chart) {
                if (chart._vortexScaleLocked) return;
                var sx = chart.scales.x;
                var sy = chart.scales.y;
                if (!sx || !sy) return;
                chart.options.scales.x.min = sx.min;
                chart.options.scales.x.max = sx.max;
                chart.options.scales.y.min = sy.min;
                chart.options.scales.y.max = sy.max;
                chart._vortexScaleLocked = true;
            }
        }],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: true, position: 'top' },
                title: { display: true, text: 'GFD Velocity Field vs GFD (Photometric mass model)', color: '#e0e0e0', font: { size: 16, weight: '500' } }
            },
            scales: {
                x: { type: 'linear', title: { display: true, text: 'Galactocentric radius (kpc)' }, grid: { color: 'rgba(64,64,64,0.3)' }, ticks: { color: '#b0b0b0' } },
                y: { type: 'linear', title: { display: true, text: 'Mass ratio T(r) (derived / photometric)' }, grid: { color: 'rgba(64,64,64,0.3)' }, ticks: { color: '#b0b0b0' }, min: 0 }
            }
        }
    });
}

/**
 * Show the observation tab bar and populate the observation data content.
 * Renders a markdown-preview style layout grouped by data source.
 * Called when entering Observation mode.
 */
function showObsTabs() {
    var tabBar = document.getElementById('obs-tab-bar');
    if (tabBar) tabBar.style.display = '';

    var container = document.getElementById('obs-data-content');
    if (!container) return;

    var obs = pinnedObservations;
    var example = pinnedGalaxyExample || currentExample;
    var galaxyName = example ? example.name.replace(/\s*\(.*\)/, '') : 'Galaxy';

    if (!obs || obs.length === 0) {
        container.innerHTML = '<p style="color:#606060;">No observation data available.</p>';
        return;
    }

    // Group observations by source. Use obs.src if present, otherwise
    // derive a default from the galaxy's first reference.
    var groups = [];
    var groupMap = {};
    var defaultSrc = 'Published data';
    if (example && example.references && example.references.length > 0) {
        // Extract short author+year from the first rotation curve reference
        defaultSrc = example.references[0];
    }

    for (var i = 0; i < obs.length; i++) {
        var src = obs[i].src || defaultSrc;
        if (!groupMap[src]) {
            groupMap[src] = { src: src, points: [] };
            groups.push(groupMap[src]);
        }
        groupMap[src].points.push(obs[i]);
    }

    // Build markdown-preview style HTML
    var html = '';
    html += '<h2>' + galaxyName + '</h2>';
    html += '<div class="obs-summary">' + obs.length + ' observations across '
          + groups.length + ' source' + (groups.length > 1 ? 's' : '') + '</div>';

    for (var g = 0; g < groups.length; g++) {
        var grp = groups[g];
        var radii = grp.points.map(function(p) { return p.r; });
        var minR = Math.min.apply(null, radii).toFixed(1);
        var maxR = Math.max.apply(null, radii).toFixed(1);

        html += '<div class="obs-source-group">';
        html += '<h3>' + grp.src + '</h3>';
        html += '<div class="obs-source-meta">'
              + grp.points.length + ' point' + (grp.points.length > 1 ? 's' : '')
              + ', ' + minR + ' \u2013 ' + maxR + ' kpc</div>';

        html += '<table class="obs-data-table">';
        html += '<thead><tr>'
              + '<th>r (kpc)</th>'
              + '<th>v (km/s)</th>'
              + '<th>err (km/s)</th>'
              + '</tr></thead><tbody>';

        for (var p = 0; p < grp.points.length; p++) {
            var pt = grp.points[p];
            html += '<tr>';
            html += '<td>' + pt.r.toFixed(1) + '</td>';
            html += '<td>' + pt.v.toFixed(0) + '</td>';
            html += '<td>' + (pt.err != null ? pt.err.toFixed(0) : '--') + '</td>';
            html += '</tr>';
        }

        html += '</tbody></table>';
        html += '</div>';
    }

    // References at the bottom
    if (example && example.references && example.references.length > 0) {
        html += '<div class="obs-refs-section">';
        html += '<h4>References</h4>';
        for (var j = 0; j < example.references.length; j++) {
            html += '<div>' + example.references[j] + '</div>';
        }
        html += '</div>';
    }

    container.innerHTML = html;
}

/**
 * Hide Charts submenu and show chart view. Delegates to pipeline (e.g. when no galaxy).
 */
function hideObsTabs() {
    navigateTo('charts', 'chart');
    _fieldAnalysisLoaded = false;
}

// =====================================================================
// CHART DATA TAB (all data used to render the chart, exportable)
// =====================================================================

/** Build CSV string from rows and headers. */
function tableToCSV(headers, rows) {
    var lines = [headers.join(',')];
    rows.forEach(function(row) {
        lines.push(row.map(function(cell) { return String(cell); }).join(','));
    });
    return lines.join('\n');
}

/** Build full chart data payload for JSON export. */
function chartDataToJSON() {
    var out = { metadata: {}, mass_model: null, observational_data: [], series: {} };
    var example = currentExample || pinnedGalaxyExample;
    var galaxyName = example ? example.name.replace(/\s*\(.*\)/, '') : 'Galaxy';
    out.metadata.galaxy = galaxyName;
    out.metadata.R_HI_kpc = (sandboxResult && sandboxResult.sparc_r_hi_kpc != null) ? sandboxResult.sparc_r_hi_kpc : null;
    out.metadata.mass_model_source = (sandboxResult && sandboxResult.gfd_source === 'manual') ? 'Manual' : 'Photometric';
    var pm = sandboxResult && sandboxResult.photometric_mass_model;
    if (pm) {
        var gasM = pm.gas ? pm.gas.M : 0, gasRd = pm.gas ? pm.gas.Rd : 0;
        var diskM = pm.disk ? pm.disk.M : 0, diskRd = pm.disk ? pm.disk.Rd : 0;
        var bulgeM = pm.bulge ? pm.bulge.M : 0, bulgeA = pm.bulge ? pm.bulge.a : 0;
        out.metadata.total_baryonic_M_sun = gasM + diskM + bulgeM;
        out.mass_model = { gas: { M: gasM, Rd_kpc: gasRd }, disk: { M: diskM, Rd_kpc: diskRd }, bulge: { M: bulgeM, a_kpc: bulgeA } };
    }
    if (pinnedObservations && pinnedObservations.length > 0) {
        out.observational_data = pinnedObservations.map(function(o) {
            return { r_kpc: Math.round(o.r * 100) / 100, v_km_s: Math.round((o.v || 0) * 100) / 100, err_km_s: (o.err != null) ? Math.round(o.err * 100) / 100 : null };
        });
    }
    var seriesIndices = [0, 1, 2, 3, 7, 11];
    var seriesKeys = ['newtonian', 'gfd_photometric', 'mond', 'observed', 'cdm', 'gfd_observed'];
    if (typeof chart !== 'undefined' && chart && chart.data && chart.data.datasets) {
        for (var s = 0; s < seriesIndices.length; s++) {
            var idx = seriesIndices[s];
            var key = seriesKeys[s];
            var ds = chart.data.datasets[idx];
            if (!ds || !ds.data || ds.data.length === 0) continue;
            var rows = [];
            for (var i = 0; i < ds.data.length; i++) {
                var pt = ds.data[i];
                var r = (pt.x != null) ? Math.round(pt.x * 100) / 100 : null;
                var v = (pt.y != null) ? Math.round(pt.y * 100) / 100 : null;
                var err = (pt.err != null) ? Math.round(pt.err * 100) / 100 : (ds.errorBars && ds.errorBars[i] != null) ? Math.round(ds.errorBars[i] * 100) / 100 : null;
                if (key === 'observed' && (err != null || pt.err != null)) rows.push({ r_kpc: r, v_km_s: v, err_km_s: err });
                else rows.push({ r_kpc: r, v_km_s: v });
            }
            out.series[key] = rows;
        }
    }
    return out;
}

/** Copy table CSV to clipboard; el is the container that holds a table, or a data-table id. */
function copyTableCSV(el) {
    var table = typeof el === 'string' ? document.getElementById(el) : el;
    if (!table) return;
    table = table.tagName === 'TABLE' ? table : table.querySelector('table');
    if (!table || !table.tHead || !table.tBodies.length) return;
    var headers = [];
    table.tHead.querySelectorAll('th').forEach(function(th) { headers.push(th.textContent.trim()); });
    var rows = [];
    table.tBodies[0].querySelectorAll('tr').forEach(function(tr) {
        var row = [];
        tr.querySelectorAll('td').forEach(function(td) { row.push(td.textContent.trim()); });
        rows.push(row);
    });
    var csv = tableToCSV(headers, rows);
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(csv).then(function() { /* done */ }).catch(function() {});
    }
}

/** Download CSV with given filename. */
function downloadCSV(csv, filename) {
    var blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename || 'chart-data.csv';
    a.click();
    URL.revokeObjectURL(a.href);
}

/** Download full chart data as JSON. */
function downloadChartDataJSON() {
    var json = chartDataToJSON();
    var str = JSON.stringify(json, null, 2);
    var blob = new Blob([str], { type: 'application/json' });
    var a = document.createElement('a');
    var galaxy = (json.metadata && json.metadata.galaxy) ? json.metadata.galaxy.replace(/\s+/g, '_') : 'galaxy';
    a.href = URL.createObjectURL(blob);
    a.download = galaxy + '_chart_data.json';
    a.click();
    URL.revokeObjectURL(a.href);
}

/** Toggle collapsible section. */
function toggleChartDataSection(headerEl) {
    var section = headerEl.closest('.chart-data-section');
    if (section) section.classList.toggle('chart-data-section-collapsed');
}

/**
 * Populate the Chart Data tab with metadata, mass model, observational data,
 * and one table per plot series (Observed, GFD Photometric, Newtonian, MOND, CDM, GFD Observed).
 * Lazy-called when the user switches to the Chart Data tab.
 */
function renderChartDataTab() {
    var container = document.getElementById('chart-data-content');
    if (!container) return;

    var example = currentExample || pinnedGalaxyExample;
    var galaxyName = example ? example.name.replace(/\s*\(.*\)/, '') : 'Galaxy';
    var pm = sandboxResult && sandboxResult.photometric_mass_model;
    var hasObs = pinnedObservations && pinnedObservations.length > 0;

    if (!example && !sandboxResult && (typeof chart === 'undefined' || !chart)) {
        container.innerHTML = '<p class="chart-data-empty">Load a galaxy to see chart data.</p>';
        return;
    }

    var html = '';

    // Global export
    html += '<div class="chart-data-toolbar">';
    html += '<button type="button" class="chart-data-export-btn" onclick="downloadChartDataJSON()">Download all as JSON</button>';
    html += '</div>';

    // Metadata (no export buttons; expanded by default)
    html += '<div class="chart-data-section">';
    html += '<div class="chart-data-section-header" onclick="toggleChartDataSection(this)"><span class="chart-data-header-title"><span class="chart-data-chevron">&#9660;</span> Metadata</span></div>';
    html += '<div class="chart-data-section-body">';
    html += '<table class="obs-data-table chart-data-kv"><tbody>';
    html += '<tr><td>Galaxy</td><td>' + (galaxyName || '--') + '</td></tr>';
    var rHi = (sandboxResult && sandboxResult.sparc_r_hi_kpc != null) ? Number(sandboxResult.sparc_r_hi_kpc).toFixed(2) + ' kpc' : '--';
    html += '<tr><td>R_HI (kpc)</td><td>' + rHi + '</td></tr>';
    var src = (sandboxResult && sandboxResult.gfd_source === 'manual') ? 'Manual' : 'Photometric';
    html += '<tr><td>Mass model source</td><td>' + src + '</td></tr>';
    if (pm) {
        var totalM = (pm.gas ? pm.gas.M : 0) + (pm.disk ? pm.disk.M : 0) + (pm.bulge ? pm.bulge.M : 0);
        html += '<tr><td>Total baryonic mass (M_sun)</td><td>' + totalM.toExponential(2) + '</td></tr>';
    }
    html += '</tbody></table></div></div>';

    // Mass model (expanded by default; buttons in header when model exists)
    html += '<div class="chart-data-section">';
    html += '<div class="chart-data-section-header" onclick="toggleChartDataSection(this)"><span class="chart-data-header-title"><span class="chart-data-chevron">&#9660;</span> Mass model</span>';
    if (pm) {
        html += '<span class="chart-data-header-actions" onclick="event.stopPropagation()">';
        html += '<button type="button" class="chart-data-export-btn" onclick="copyTableCSV(document.getElementById(\'chart-data-table-mass\'))">Copy CSV</button>';
        html += '<button type="button" class="chart-data-export-btn" onclick="var t=document.getElementById(\'chart-data-table-mass\');if(t){var h=[],r=[];t.tHead.querySelectorAll(\'th\').forEach(function(x){h.push(x.textContent);});t.tBodies[0].querySelectorAll(\'tr\').forEach(function(tr){var row=[];tr.querySelectorAll(\'td\').forEach(function(td){row.push(td.textContent);});r.push(row);});downloadCSV(tableToCSV(h,r),\'mass_model.csv\');}">Download CSV</button>';
        html += '</span>';
    }
    html += '</div>';
    html += '<div class="chart-data-section-body">';
    if (pm) {
        var gasM = pm.gas ? pm.gas.M : 0, gasRd = pm.gas ? pm.gas.Rd : 0;
        var diskM = pm.disk ? pm.disk.M : 0, diskRd = pm.disk ? pm.disk.Rd : 0;
        var bulgeM = pm.bulge ? pm.bulge.M : 0, bulgeA = pm.bulge ? pm.bulge.a : 0;
        var totalMass = gasM + diskM + bulgeM;
        html += '<table class="obs-data-table" id="chart-data-table-mass">';
        html += '<thead><tr><th>Component</th><th>M (M_sun)</th><th>Scale (kpc)</th></tr></thead><tbody>';
        html += '<tr><td>Gas disk</td><td>' + gasM.toExponential(2) + '</td><td>R_d = ' + gasRd.toFixed(2) + '</td></tr>';
        html += '<tr><td>Stellar disk</td><td>' + diskM.toExponential(2) + '</td><td>R_d = ' + diskRd.toFixed(2) + '</td></tr>';
        html += '<tr><td>Bulge</td><td>' + bulgeM.toExponential(2) + '</td><td>a = ' + bulgeA.toFixed(2) + '</td></tr>';
        html += '<tr><td><strong>Total</strong></td><td><strong>' + totalMass.toExponential(2) + '</strong></td><td></td></tr>';
        html += '</tbody></table>';
    } else {
        html += '<p class="chart-data-empty">No mass model loaded.</p>';
    }
    html += '</div></div>';

    // Observational data (collapsed by default; buttons in header when data exists)
    html += '<div class="chart-data-section chart-data-section-collapsed">';
    html += '<div class="chart-data-section-header" onclick="toggleChartDataSection(this)"><span class="chart-data-header-title"><span class="chart-data-chevron">&#9660;</span> Observational data</span>';
    if (hasObs) {
        html += '<span class="chart-data-header-actions" onclick="event.stopPropagation()">';
        html += '<button type="button" class="chart-data-export-btn" onclick="copyTableCSV(document.getElementById(\'chart-data-table-obs\'))">Copy CSV</button>';
        html += '<button type="button" class="chart-data-export-btn" onclick="var t=document.getElementById(\'chart-data-table-obs\');if(t){var h=[],r=[];t.tHead.querySelectorAll(\'th\').forEach(function(x){h.push(x.textContent);});t.tBodies[0].querySelectorAll(\'tr\').forEach(function(tr){var row=[];tr.querySelectorAll(\'td\').forEach(function(td){row.push(td.textContent);});r.push(row);});downloadCSV(tableToCSV(h,r),\'observations.csv\');}">Download CSV</button>';
        html += '</span>';
    }
    html += '</div>';
    html += '<div class="chart-data-section-body">';
    if (hasObs) {
        html += '<table class="obs-data-table" id="chart-data-table-obs">';
        html += '<thead><tr><th>r (kpc)</th><th>v (km/s)</th><th>err (km/s)</th></tr></thead><tbody>';
        for (var o = 0; o < pinnedObservations.length; o++) {
            var ob = pinnedObservations[o];
            var errStr = (ob.err != null) ? ob.err.toFixed(2) : '--';
            html += '<tr><td>' + ob.r.toFixed(2) + '</td><td>' + (ob.v != null ? ob.v.toFixed(2) : '--') + '</td><td>' + errStr + '</td></tr>';
        }
        html += '</tbody></table>';
    } else {
        html += '<p class="chart-data-empty">No observations.</p>';
    }
    html += '</div></div>';

    // Plot series (one subsection per badge series with data)
    var seriesConfig = [
        { idx: 3, id: 'observed', filename: 'observed.csv' },
        { idx: 1, id: 'gfd-photometric', filename: 'gfd_photometric.csv' },
        { idx: 0, id: 'newtonian', filename: 'newtonian.csv' },
        { idx: 2, id: 'mond', filename: 'mond.csv' },
        { idx: 7, id: 'cdm', filename: 'cdm.csv' },
        { idx: 11, id: 'gfd-observed', filename: 'gfd_observed.csv' }
    ];
    if (typeof chart !== 'undefined' && chart && chart.data && chart.data.datasets) {
        for (var sc = 0; sc < seriesConfig.length; sc++) {
            var cfg = seriesConfig[sc];
            var ds = chart.data.datasets[cfg.idx];
            if (!ds || !ds.data || ds.data.length === 0) continue;
            var label = (ds.label || 'Series ' + cfg.idx).replace(/</g, '&lt;').replace(/>/g, '&gt;');
            var tableId = 'chart-data-table-' + cfg.id;
            html += '<div class="chart-data-section chart-data-section-collapsed">';
            html += '<div class="chart-data-section-header" onclick="toggleChartDataSection(this)"><span class="chart-data-header-title"><span class="chart-data-chevron">&#9660;</span> ' + label + '</span>';
            html += '<span class="chart-data-header-actions" onclick="event.stopPropagation()">';
            html += '<button type="button" class="chart-data-export-btn" onclick="copyTableCSV(document.getElementById(\'' + tableId + '\'))">Copy CSV</button>';
            html += '<button type="button" class="chart-data-export-btn" onclick="var t=document.getElementById(\'' + tableId + '\');if(t){var h=[],r=[];t.tHead.querySelectorAll(\'th\').forEach(function(x){h.push(x.textContent);});t.tBodies[0].querySelectorAll(\'tr\').forEach(function(tr){var row=[];tr.querySelectorAll(\'td\').forEach(function(td){row.push(td.textContent);});r.push(row);});downloadCSV(tableToCSV(h,r),\'' + cfg.filename + '\');}">Download CSV</button>';
            html += '</span></div>';
            html += '<div class="chart-data-section-body">';
            var hasErr = cfg.idx === 3;
            html += '<table class="obs-data-table" id="' + tableId + '">';
            if (hasErr) html += '<thead><tr><th>r (kpc)</th><th>v (km/s)</th><th>err (km/s)</th></tr></thead>';
            else html += '<thead><tr><th>r (kpc)</th><th>v (km/s)</th></tr></thead>';
            html += '<tbody>';
            for (var i = 0; i < ds.data.length; i++) {
                var pt = ds.data[i];
                var rx = (pt.x != null) ? pt.x.toFixed(2) : '--';
                var vy = (pt.y != null) ? pt.y.toFixed(2) : '--';
                if (hasErr) {
                    var errVal = (pt.err != null) ? pt.err : (ds.errorBars && ds.errorBars[i] != null) ? ds.errorBars[i] : null;
                    var errStr = (errVal != null) ? errVal.toFixed(2) : '--';
                    html += '<tr><td>' + rx + '</td><td>' + vy + '</td><td>' + errStr + '</td></tr>';
                } else {
                    html += '<tr><td>' + rx + '</td><td>' + vy + '</td></tr>';
                }
            }
            html += '</tbody></table>';
            html += '</div></div>';
        }
    }

    container.innerHTML = html;
}

// =====================================================================
// FIELD ANALYSIS TAB (GFD metrics + equation display)
// =====================================================================

// Cache flag so we only fetch once per observation mode session.
var _fieldAnalysisLoaded = false;
var _fieldAnalysisData = null;

/**
 * Fetch field analysis metrics from the backend and render them into
 * the Field Analysis tab content area. Uses the current slider values
 * and the fitted Origin Throughput from the last inference run.
 */
async function loadFieldAnalysis() {
    if (_fieldAnalysisLoaded && _fieldAnalysisData) {
        return;
    }

    var container = document.getElementById('field-analysis-content');
    if (!container) return;

    container.innerHTML = '<p class="fa-loading">Loading field analysis...</p>';

    var massModel = getMassModelFromSliders();
    var gr = parseFloat(galacticRadiusSlider.value) || 0;
    var accelRatio = parseFloat(accelSlider.value) || 1.0;

    var throughput = autoVortexStrength;
    if (throughput === null && vortexStrengthSlider) {
        throughput = parseFloat(vortexStrengthSlider.value);
    }
    if (!throughput || gr <= 0) {
        container.innerHTML = '<p class="fa-loading">Observation data required for field analysis.</p>';
        return;
    }

    try {
        var resp = await fetch('/api/rotation/field_analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mass_model: massModel,
                galactic_radius: gr,
                vortex_strength: throughput,
                accel_ratio: accelRatio
            })
        });
        if (!resp.ok) throw new Error('API error: ' + resp.status);
        _fieldAnalysisData = await resp.json();
        _fieldAnalysisLoaded = true;
        renderFieldAnalysis(container, _fieldAnalysisData);
    } catch (err) {
        console.error('Field analysis error:', err);
        container.innerHTML = '<p style="color: #ff6b6b;">Failed to load field analysis.</p>';
    }
}

/**
 * Post-render KaTeX pass. After innerHTML is set, find all elements
 * with data-katex attributes and render them with KaTeX. This avoids
 * timing issues with deferred script loading and ensures the DOM is
 * ready before KaTeX processes each expression.
 */
function renderKatexElements(root) {
    if (typeof katex === 'undefined') return;
    var els = root.querySelectorAll('[data-katex]');
    for (var i = 0; i < els.length; i++) {
        var latex = els[i].getAttribute('data-katex');
        try {
            katex.render(latex, els[i], {
                displayMode: false,
                throwOnError: false
            });
        } catch (e) {
            els[i].textContent = latex;
        }
    }
}

/**
 * Build the Field Analysis tab content from the API response.
 * Structured as an engineering classification report: prioritised
 * sections, collapsible panels, clean enterprise typography.
 */
function renderFieldAnalysis(container, data) {
    var galaxyName = '';
    var example = pinnedGalaxyExample || currentExample;
    if (example) {
        galaxyName = example.name.replace(/\s*\(.*\)/, '');
    }

    var h = '';

    // --- Report Header ---
    h += '<div class="fa-report-header">';
    h += '<div class="fa-report-id">FIELD ANALYSIS REPORT</div>';
    h += '<h2 class="fa-report-name">' + (galaxyName || 'UNKNOWN SYSTEM') + '</h2>';
    h += '</div>';

    // =========================================================
    // SECTION 1: FITTED MODEL (open)
    // =========================================================
    var pe = data.parametric;
    if (pe) {
        h += _faSection('fitted-model', 'Fitted Model', true);
        h += '<p class="fa-prose">Complete callable equation with all resolved parameters.'
           + ' Reproduces the rotation curve without the GFD pipeline.</p>';

        h += '<p class="fa-step-label">1. Enclosed mass profiles'
           + ' <span class="fa-step-note">(Hernquist bulge + exponential disk + gas)</span></p>';
        h += _eqBlock(pe.mass_bulge);
        h += _eqBlock(pe.mass_disk);
        h += _eqBlock(pe.mass_gas);
        h += _eqBlock(pe.mass_total);

        h += '<p class="fa-step-label">2. Newtonian acceleration</p>';
        h += _eqBlock(pe.g_newtonian);

        h += '<p class="fa-step-label">3. GFD field solve</p>';
        h += _eqBlock(pe.gfd_solve);

        h += '<p class="fa-step-label">4a. Structural correction'
           + ' <span class="fa-step-note">(outer arm, r > R<sub>t</sub>)</span></p>';
        h += _eqBlock(pe.throughput_outer);

        if (pe.vortex_reflect) {
            h += '<p class="fa-step-label">4b. Vortex reflection'
               + ' <span class="fa-step-note">(inner arm, r &le; R<sub>t</sub>)</span></p>';
            h += _eqBlock(pe.vortex_reflect);
        }

        h += '<p class="fa-step-label">5. Circular velocity</p>';
        h += _eqBlock(pe.velocity_final);

        h += '<div class="fa-constants-block">';
        h += _eqBlock(pe.constants);
        h += _eqBlock(pe.units);
        h += '</div>';

        h += '</div></div>';
    }

    // =========================================================
    // SECTION 2: FIELD GEOMETRY (open)
    // =========================================================
    var fg = data.field_geometry;
    if (fg) {
        h += _faSection('field-geom', 'Field Geometry', true);

        // Geometry comparison: from observation (topological) vs profile (catalog)
        var hasPredicted = fg.predicted_origin_kpc !== null && fg.predicted_origin_kpc !== undefined;
        if (hasPredicted) {
            // Build a delta badge with color coding
            function _deltaBadge(val) {
                if (val === null || val === undefined) return '<span style="color:#888;">N/A</span>';
                var abs = Math.abs(val);
                var color = abs < 3 ? '#4caf50' : abs < 10 ? '#ff9800' : '#ff6b6b';
                return '<span style="color:' + color + '; font-weight:600;">'
                     + (val >= 0 ? '+' : '') + val.toFixed(1) + '%</span>';
            }

            h += '<p class="fa-step-label">Derived from Mass Model vs Galaxy Profile</p>';
            h += '<table class="fa-table">';
            h += '<tr class="fa-row fa-row-header">'
               + '<td class="fa-label"></td>'
               + '<td class="fa-symbol"></td>'
               + '<td class="fa-value" style="text-align:right; font-weight:600; color:#9ecfff;">From Observation</td>'
               + '<td class="fa-value" style="text-align:right; color:#888;">Profile</td>'
               + '<td class="fa-value" style="text-align:right;">&Delta;</td>'
               + '</tr>';
            h += '<tr class="fa-row">'
               + '<td class="fa-label">Field Origin</td>'
               + '<td class="fa-symbol">' + _katexInline('R_t') + '</td>'
               + '<td class="fa-value" style="text-align:right; font-weight:600;">' + fg.predicted_origin_kpc + ' kpc</td>'
               + '<td class="fa-value" style="text-align:right; color:#888;">' + fg.catalog_origin_kpc + ' kpc</td>'
               + '<td class="fa-value" style="text-align:right;">' + _deltaBadge(fg.origin_delta_pct) + '</td>'
               + '</tr>';
            h += '<tr class="fa-row">'
               + '<td class="fa-label">Field Horizon</td>'
               + '<td class="fa-symbol">' + _katexInline('R_{\\mathrm{env}}') + '</td>'
               + '<td class="fa-value" style="text-align:right; font-weight:600;">' + fg.predicted_horizon_kpc + ' kpc</td>'
               + '<td class="fa-value" style="text-align:right; color:#888;">' + fg.catalog_horizon_kpc + ' kpc</td>'
               + '<td class="fa-value" style="text-align:right;">' + _deltaBadge(fg.horizon_delta_pct) + '</td>'
               + '</tr>';
            h += '<tr class="fa-row">'
               + '<td class="fa-label">Throat Fraction</td>'
               + '<td class="fa-symbol">' + _katexInline('R_t / R_{\\mathrm{env}}') + '</td>'
               + '<td class="fa-value" style="text-align:right; font-weight:600;">' + fg.throat_fraction + '</td>'
               + '<td class="fa-value" style="text-align:right; color:#888;">0.3000</td>'
               + '<td class="fa-value" style="text-align:right;"></td>'
               + '</tr>';
            h += '</table>';
            if (fg.yN_at_throat !== null && fg.yN_at_throat !== undefined) {
                h += '<p class="fa-prose" style="margin-top:8px;">'
                   + 'Throat condition: ' + _katexInline('y_N(R_t) = 18/65')
                   + ' &nbsp; Measured: ' + fg.yN_at_throat
                   + ' &nbsp; Method: ' + fg.prediction_method + '</p>';
            }
        } else {
            h += '<p class="fa-prose" style="margin-top:4px; color:#ff9800;">Deep-field system: '
               + 'y<sub>N</sub> never reaches 18/65. Catalog geometry used.</p>';
            h += '<table class="fa-table">';
            h += _faRow('Field Origin (profile)', _katexInline('R_t'), fg.catalog_origin_kpc + ' kpc');
            h += _faRow('Field Horizon (profile)', _katexInline('R_{\\mathrm{env}}'), fg.catalog_horizon_kpc + ' kpc');
            h += '</table>';
        }

        // Throughput and throat fraction
        h += '<table class="fa-table" style="margin-top:12px;">';
        h += _faRow('Throat Fraction', _katexInline('R_t / R_{\\mathrm{env}}'), fg.throat_fraction);
        h += _faRow('Origin Throughput (fitted)', _katexInline('\\sigma'), fg.origin_throughput_fitted);
        h += _faRow('Origin Throughput (theoretical)', '', fg.origin_throughput_theoretical);
        h += _faRow('Throughput Delta', '', _signed(fg.throughput_delta_pct) + '%');
        h += '</table>';
        h += '</div></div>';
    }

    // =========================================================
    // SECTION 3: RESOLVED MASS (open)
    // =========================================================
    var mp = data.mass_properties;
    if (mp) {
        h += _faSection('resolved-mass', 'Resolved Mass', true);
        h += '<table class="fa-table">';
        h += _faRow('Total Baryonic', _katexInline('M_{\\mathrm{bary}}'), _sciNot(mp.total_baryonic_Msun) + ' M<sub>&#x2609;</sub>');
        h += _faRow('Stellar Bulge', _katexInline('M_{\\mathrm{b}}'), _sciNot(mp.bulge_Msun) + ' M<sub>&#x2609;</sub>');
        h += _faRow('Stellar Disk', _katexInline('M_{\\mathrm{d}}'), _sciNot(mp.disk_Msun) + ' M<sub>&#x2609;</sub>');
        h += _faRow('Gas Disk', _katexInline('M_{\\mathrm{g}}'), _sciNot(mp.gas_Msun) + ' M<sub>&#x2609;</sub>');
        h += _faRow('Gas Fraction', _katexInline('f_{\\mathrm{gas}}'), (mp.gas_fraction * 100).toFixed(1) + '%');
        h += _faRow('Bulge / Total', _katexInline('B/T'), (mp.bulge_to_total * 100).toFixed(1) + '%');
        h += '</table>';
        h += '</div></div>';
    }

    // =========================================================
    // SECTION 4: VELOCITY PROFILE (collapsed)
    // =========================================================
    var dyn = data.dynamics;
    if (dyn) {
        h += _faSection('velocity', 'Velocity Profile', false);
        h += '<table class="fa-table">';
        h += _faRow('Peak velocity (GFD base)', '', dyn.v_peak_gfd_kms + ' km/s at ' + dyn.r_peak_gfd_kpc + ' kpc');
        h += _faRow('Peak velocity (observed)', '', dyn.v_peak_observed_kms + ' km/s at ' + dyn.r_peak_observed_kpc + ' kpc');
        h += _faRow('v at Field Origin (base)', '', dyn.v_at_origin_gfd_kms + ' km/s');
        h += _faRow('v at Field Origin (observed)', '', dyn.v_at_origin_observed_kms + ' km/s');
        h += _faRow('v at Field Horizon (base)', '', dyn.v_at_horizon_gfd_kms + ' km/s');
        h += _faRow('v at Field Horizon (observed)', '', dyn.v_at_horizon_observed_kms + ' km/s');
        if (dyn.r_transition_kpc > 0) {
            h += _faRow('Transition radius (y<sub>N</sub> = 1)', _katexInline('r_{\\mathrm{trans}}'), dyn.r_transition_kpc + ' kpc');
        }
        h += '</table>';
        h += '</div></div>';
    }

    // =========================================================
    // SECTION 5: STRUCTURAL CORRECTION (collapsed)
    // =========================================================
    var te = data.throughput_effect;
    if (te) {
        h += _faSection('struct-corr', 'Structural Correction', false);
        h += '<table class="fa-table">';
        h += _faRow('Delta v at Origin', _katexInline('\\Delta v(R_t)'), _signed(te.delta_v_at_origin_kms) + ' km/s');
        h += _faRow('Delta v at Horizon', _katexInline('\\Delta v(R_{\\mathrm{env}})'), _signed(te.delta_v_at_horizon_kms) + ' km/s');
        h += _faRow('Delta % at Horizon', '', _signed(te.delta_pct_at_horizon) + '%');
        h += _faRow('GFD / Newtonian at Origin', '', te.gfd_newt_ratio_at_origin + 'x');
        h += _faRow('GFD / Newtonian at Horizon', '', te.gfd_newt_ratio_at_horizon + 'x');
        h += '</table>';
        h += '</div></div>';
    }

    // =========================================================
    // SECTION 6: FIELD CONSTANTS (collapsed)
    // =========================================================
    var fc = data.field_coupling;
    if (fc) {
        h += _faSection('field-const', 'Field Constants', false);
        h += '<table class="fa-table">';
        h += _faRow('Acceleration scale', _katexInline('a_0'), fc.a0_ms2.toExponential(4) + ' m/s<sup>2</sup>');
        h += _faRow('Simplex number', _katexInline('k'), fc.k_simplex);
        h += _faRow('y<sub>N</sub> at Field Origin', _katexInline('y_N(R_t)'), fc.yN_at_origin);
        h += _faRow('y<sub>N</sub> at Field Horizon', _katexInline('y_N(R_{\\mathrm{env}})'), fc.yN_at_horizon);
        h += _faRow('Effective coupling', _katexInline('f_{\\mathrm{eff}}'), fc.f_eff);
        h += _faRow('Structural release amplitude', _katexInline('g_0'), fc.g0_ms2.toExponential(4) + ' m/s<sup>2</sup>');
        h += _faRow('Structural fraction', '', fc.structural_frac);
        h += _faRow('Outer power law', '', fc.power_law);
        h += '</table>';
        h += '</div></div>';
    }

    // =========================================================
    // SECTION 7: THEORY REFERENCE (collapsed)
    // =========================================================
    if (data.equations) {
        h += _faSection('theory-ref', 'Theory Reference', false);
        h += '<p class="fa-prose">The complete GFD derivation chain, from the covariant action'
           + ' through to the velocity formula. All terms are fixed by the topology'
           + ' of the stellated octahedron.</p>';

        h += '<p class="fa-step-label">Covariant Action</p>';
        h += _eqBlock(data.equations.action);

        h += '<p class="fa-step-label">Coupling Polynomial'
           + ' <span class="fa-step-note">(stellated octahedron)</span></p>';
        h += _eqBlock(data.equations.coupling_poly);

        h += '<p class="fa-step-label">Scalar Lagrangian</p>';
        h += _eqBlock(data.equations.lagrangian);

        h += '<p class="fa-step-label">Algebraic Field Equation'
           + ' <span class="fa-step-note">(Euler-Lagrange in spherical symmetry)</span></p>';
        h += _eqBlock(data.equations.field_eq);

        h += '<p class="fa-step-label">Analytic Solution</p>';
        h += _eqBlock(data.equations.solution);

        h += '<p class="fa-step-label">Circular Velocity</p>';
        h += _eqBlock(data.equations.velocity);

        h += '<p class="fa-step-label">Acceleration Scale'
           + ' <span class="fa-step-note">(zero free parameters)</span></p>';
        h += _eqBlock(data.equations.acceleration_scale);

        h += '</div></div>';
    }

    container.innerHTML = h;

    // Post-render: KaTeX processes all data-katex elements
    renderKatexElements(container);

    // Wire collapsible section headers
    _wireFaCollapse(container);
}

/**
 * Build the opening HTML for a collapsible field analysis section.
 * Returns the header + opening body div. Caller must close with </div></div>.
 */
function _faSection(id, title, open) {
    var cls = open ? 'fa-section' : 'fa-section fa-collapsed';
    var chevron = open ? '&#9660;' : '&#9654;';
    return '<div class="' + cls + '" data-fa-section="' + id + '">'
         + '<div class="fa-section-header" onclick="_toggleFaSection(this)">'
         + '<span class="fa-chevron">' + chevron + '</span>'
         + '<h3>' + title + '</h3>'
         + '</div>'
         + '<div class="fa-section-body"' + (open ? '' : ' style="display:none"') + '>';
}

/** Toggle a collapsible field analysis section. */
function _toggleFaSection(headerEl) {
    var section = headerEl.parentElement;
    var body = headerEl.nextElementSibling;
    var chevron = headerEl.querySelector('.fa-chevron');
    if (!body) return;
    if (body.style.display === 'none') {
        body.style.display = '';
        chevron.innerHTML = '&#9660;';
        section.classList.remove('fa-collapsed');
    } else {
        body.style.display = 'none';
        chevron.innerHTML = '&#9654;';
        section.classList.add('fa-collapsed');
    }
}

/** Wire collapse handlers after innerHTML render. */
function _wireFaCollapse(container) {
    // KaTeX elements inside initially-hidden sections need rendering
    // when the section is first opened.
    var sections = container.querySelectorAll('.fa-section[data-fa-section]');
    sections.forEach(function(sec) {
        var header = sec.querySelector('.fa-section-header');
        if (!header) return;
        var body = sec.querySelector('.fa-section-body');
        if (!body) return;
        var rendered = !sec.classList.contains('fa-collapsed');
        header.addEventListener('click', function() {
            if (!rendered) {
                renderKatexElements(body);
                rendered = true;
            }
        });
    });
}

/** Build a left-aligned block equation placeholder for post-render KaTeX. */
function _eqBlock(latex) {
    return '<div class="fa-eq" data-katex="' + _escAttr(latex) + '"></div>';
}

/** Build an inline KaTeX placeholder span (rendered in post-pass). */
function _katexInline(latex) {
    return '<span class="fa-sym" data-katex="' + _escAttr(latex) + '"></span>';
}

/** Escape a string for safe use inside an HTML attribute. */
function _escAttr(s) {
    return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;')
            .replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** Build a three-column metrics row: label, symbol, value. */
function _faRow(label, sym, value) {
    return '<tr>'
         + '<td class="fa-label">' + label + '</td>'
         + '<td class="fa-sym-cell">' + sym + '</td>'
         + '<td class="fa-value">' + value + '</td>'
         + '</tr>';
}

/** Format a solar mass value in compact scientific notation. */
function _sciNot(val) {
    if (val === 0 || val == null) return '0';
    return parseFloat(val).toExponential(3);
}

/** Prefix positive numbers with +, leave negatives as-is. */
function _signed(val) {
    var n = parseFloat(val);
    if (isNaN(n)) return val;
    return (n >= 0 ? '+' : '') + val;
}

// =====================================================================
// FONT SIZE CONTROLS (Observation Data + Field Analysis tabs)
// =====================================================================

var _obsFontSizes = [0.75, 0.85, 1.0, 1.15, 1.3];
var _obsFontIndex = 2;

/**
 * Wire up all A-/A+ buttons inside .obs-data-face panels.
 * Each button steps the shared font index and applies to all
 * .obs-data-content elements simultaneously.
 */
function initObsFontControls() {
    var btns = document.querySelectorAll('.obs-font-btn');
    btns.forEach(function(btn) {
        btn.addEventListener('click', function() {
            var dir = parseInt(this.getAttribute('data-dir'));
            _obsFontIndex = Math.max(0, Math.min(_obsFontSizes.length - 1, _obsFontIndex + dir));
            var size = _obsFontSizes[_obsFontIndex] + 'em';
            var panels = document.querySelectorAll('.obs-data-content');
            panels.forEach(function(p) { p.style.fontSize = size; });
        });
    });
}

// Toggle mass value sliders between editable (prediction) and read-only output (inference)
function setMassSliderEditable(editable) {
    const massSliders = [bulgeMassSlider, diskMassSlider, gasMassSlider];
    const massLabels = [
        bulgeMassSlider.previousElementSibling,
        diskMassSlider.previousElementSibling,
        diskMassSlider.previousElementSibling
    ];

    massSliders.forEach(s => {
        s.disabled = !editable;
        // CSS handles the heavy dimming via :disabled selector
    });

    // Dim the "Mass" label text in inference mode, keep value visible
    document.querySelectorAll('.mass-component .control-label').forEach((label, idx) => {
        // Even-indexed labels are mass labels, odd are scale labels
        if (idx % 2 === 0) {
            // Mass label row
            const labelText = label.querySelector('span:first-child');
            const valueText = label.querySelector('.control-value');
            if (!editable) {
                if (labelText) labelText.style.opacity = '0.4';
                // Keep value text bright -- it's the output
                if (valueText) valueText.style.opacity = '1';
            } else {
                if (labelText) labelText.style.opacity = '1';
                if (valueText) valueText.style.opacity = '1';
            }
        }
    });

    // Scale length sliders are always editable (user controls shape)
    const scaleSliders = [bulgeScaleSlider, diskScaleSlider, gasScaleSlider];
    scaleSliders.forEach(s => {
        s.disabled = false;
        s.style.opacity = '1';
    });
}

// =====================================================================
// SLIDER EVENT LISTENERS
// =====================================================================

distanceSlider.addEventListener('input', () => {
    if (!isLoadingExample && currentExample) {
        currentExample = null;
        // Keep pinned observations -- user is fine-tuning
    }
    if (!isLoadingExample) {
        updateDisplays();
        debouncedUpdateChart();
    }
});

massSlider.addEventListener('input', () => {
    if (!isLoadingExample && currentMode === 'prediction' && currentExample) {
        if (!currentExample.mass_model) {
            currentExample = null;
        }
    }
    if (!isLoadingExample) {
        updateDisplays();
        debouncedUpdateChart();
    }
});

velocitySlider.addEventListener('input', () => {
    if (!isLoadingExample) {
        currentExample = null;
        // Keep pinned observations -- user is fine-tuning
        updateDisplays();
        debouncedUpdateChart();
    }
});

accelSlider.addEventListener('input', () => {
    if (!isLoadingExample && currentExample) {
        currentExample = null;
        // Keep pinned observations -- user is fine-tuning
    }
    if (!isLoadingExample) {
        updateDisplays();
        debouncedUpdateChart();
    }
});

// =====================================================================
// RESET TO PHOTOMETRIC BUTTON (Mass Model panel)
// =====================================================================
var resetToPhotometricBtn = document.getElementById('reset-to-photometric-btn');
if (resetToPhotometricBtn) {
    resetToPhotometricBtn.addEventListener('click', function() {
        resetToPhotometric();
    });
}

// =====================================================================
// BAND METHOD SELECTOR
// =====================================================================
document.getElementById('band-method-select').addEventListener('change', async function() {
    if (lastMultiResult && lastModelTotal > 0) {
        await updateBand();
    }
});

// =====================================================================
// RESIZABLE PANEL
// =====================================================================

const leftPanel = document.getElementById('left-panel');
const resizeHandle = document.getElementById('resize-handle');
let isResizing = false;

resizeHandle.addEventListener('mousedown', () => {
    isResizing = true;
    resizeHandle.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
});

document.addEventListener('mousemove', (e) => {
    if (!isResizing) return;
    const newWidth = e.clientX;
    if (newWidth >= 300 && newWidth <= 800) {
        leftPanel.style.width = newWidth + 'px';
    }
});

document.addEventListener('mouseup', () => {
    if (isResizing) {
        isResizing = false;
        resizeHandle.classList.remove('dragging');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        chart.resize();
        if (vpsChart) {
            vpsChart.resize();
            if (isAutoFitted) setTimeout(function() { updateVpsChart(); }, 0);
        }
    }
    if (isResizingMetrics) {
        isResizingMetrics = false;
        metricsResizeHandle.classList.remove('dragging');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        chart.resize();
        if (vpsChart) {
            vpsChart.resize();
            if (isAutoFitted) setTimeout(function() { updateVpsChart(); }, 0);
        }
    }
});

// =====================================================================
// RIGHT METRICS PANEL RESIZE + COLLAPSE
// =====================================================================

const metricsPanel = document.getElementById('metrics-panel');
const metricsResizeHandle = document.getElementById('metrics-resize-handle');
let isResizingMetrics = false;
let metricsWidthBeforeCollapse = 280;

metricsResizeHandle.addEventListener('mousedown', () => {
    if (metricsPanel.classList.contains('collapsed')) return;
    isResizingMetrics = true;
    metricsResizeHandle.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
});

document.addEventListener('mousemove', (e) => {
    if (!isResizingMetrics) return;
    var containerRight = metricsPanel.parentElement.getBoundingClientRect().right;
    var newWidth = containerRight - e.clientX;
    if (newWidth >= 280 && newWidth <= 600) {
        metricsPanel.style.width = newWidth + 'px';
        metricsPanel.style.minWidth = newWidth + 'px';
        metricsWidthBeforeCollapse = newWidth;
    }
});

function toggleMetricsPanel() {
    var panel = metricsPanel;
    if (panel.classList.contains('collapsed')) {
        panel.classList.remove('collapsed');
        panel.style.width = metricsWidthBeforeCollapse + 'px';
        panel.style.minWidth = metricsWidthBeforeCollapse + 'px';
    } else {
        metricsWidthBeforeCollapse = parseInt(panel.style.width) || 280;
        panel.classList.add('collapsed');
    }
    setTimeout(function() {
        chart.resize();
        if (vpsChart) {
            vpsChart.resize();
            if (isAutoFitted) updateVpsChart();
        }
    }, 250);
}

// =====================================================================
// METRICS PANEL: VERTICAL SPLIT PANE
// =====================================================================

var metricsPaneTop = document.getElementById('metrics-pane-top');
var metricsPaneBottom = document.getElementById('metrics-pane-bottom');
var metricsPaneDivider = document.getElementById('metrics-pane-divider');
var isResizingPane = false;

if (metricsPaneDivider) {
    metricsPaneDivider.addEventListener('mousedown', function(e) {
        isResizingPane = true;
        metricsPaneDivider.classList.add('dragging');
        document.body.style.cursor = 'row-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });
}

document.addEventListener('mousemove', function(e) {
    if (!isResizingPane) return;
    var panelRect = metricsPanel.getBoundingClientRect();
    var offsetY = e.clientY - panelRect.top;
    var panelH = panelRect.height;
    var minPx = 100;
    var maxPx = panelH - 100;
    if (offsetY < minPx) offsetY = minPx;
    if (offsetY > maxPx) offsetY = maxPx;
    metricsPaneTop.style.height = offsetY + 'px';
    metricsPaneTop.style.flex = 'none';
    metricsPaneBottom.style.flex = '1';
});

document.addEventListener('mouseup', function() {
    if (isResizingPane) {
        isResizingPane = false;
        metricsPaneDivider.classList.remove('dragging');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
    }
});

/**
 * Show the auto-map diagnostics pane with gene report data.
 */
function showAutoMapDiagnostics(data) {
    if (!metricsPaneTop || !data) return;

    // Update header to reflect fit method
    var headerText = document.querySelector('#metrics-automap-summary .metrics-header-text');
    if (headerText) {
        var method = data.throughput_fit && data.throughput_fit.method;
        if (method && method === 'inference') {
            headerText.textContent = 'Observation Results (derived)';
        } else {
            headerText.textContent = 'Observation Results';
        }
    }

    // Summary metrics
    var fit = data.throughput_fit;
    if (fit) {
        document.getElementById('metric-am-gfd-rms').textContent =
            fit.gfd_rms_km_s != null ? fit.gfd_rms_km_s.toFixed(1) + ' km/s' : '--';
        var chi2El = document.getElementById('metric-am-chi2');
        if (fit.chi2_dof != null) {
            chi2El.textContent = fit.chi2_dof.toFixed(2);
            chi2El.className = 'metrics-value' + (fit.chi2_dof <= 1.5 ? ' good' : fit.chi2_dof <= 3 ? ' warn' : ' bad');
        }
    }

    // Band coverage
    var band = data.band_coverage;
    if (band) {
        document.getElementById('metric-am-band-hits').textContent =
            band.obs_hits + '/' + band.obs_total;
    }

    // Field Geometry section: prefer topologically derived values from
    // inferredFieldGeometry, fall back to slider + THROAT_FRAC constant.
    var fg = inferredFieldGeometry;
    var rEnv, rOrigin, throatRatio;
    if (fg && fg.envelope_radius_kpc != null && fg.throat_radius_kpc != null) {
        rEnv = fg.envelope_radius_kpc;
        rOrigin = fg.throat_radius_kpc;
        throatRatio = fg.throat_fraction != null ? fg.throat_fraction : (rEnv > 0 ? rOrigin / rEnv : 0);
    } else {
        rEnv = parseFloat(galacticRadiusSlider.value) || 0;
        rOrigin = THROAT_FRAC * rEnv;
        throatRatio = rEnv > 0 ? THROAT_FRAC : 0;
    }
    var throughputVal = data.auto_origin_throughput;

    var fgThroughput = document.getElementById('metric-fg-throughput');
    var fgOrigin = document.getElementById('metric-fg-origin');
    var fgHorizon = document.getElementById('metric-fg-horizon');
    var fgRatio = document.getElementById('metric-fg-ratio');

    if (fgThroughput) fgThroughput.textContent = throughputVal != null ? throughputVal.toFixed(2) : '--';
    if (fgOrigin) fgOrigin.textContent = rOrigin.toFixed(1) + ' kpc';
    if (fgHorizon) fgHorizon.textContent = rEnv.toFixed(1) + ' kpc';
    if (fgRatio) fgRatio.textContent = throatRatio > 0 ? throatRatio.toFixed(2) : '--';

    // Gene report (parameter changes table)
    var report = data.gene_report;
    var tbody = document.getElementById('metrics-automap-tbody');
    if (tbody && report && report.length > 0) {
        var html = '';
        for (var i = 0; i < report.length; i++) {
            var g = report[i];
            var pctChange = ((g.ratio - 1.0) * 100);
            var pctStr = (pctChange >= 0 ? '+' : '') + pctChange.toFixed(1) + '%';
            var sigStr = g.within_sigma ? '0' : g.sigma_excess.toFixed(1);
            var chi2Str = g.chi2_bought > 0 ? '+' + g.chi2_bought.toFixed(1) : g.chi2_bought.toFixed(1);

            var isMass = g.gene && (g.gene === 'Mb' || g.gene === 'Md' || g.gene === 'Mg');
            var priorStr = isMass ? g.published.toExponential(3) : g.published.toFixed(2);
            var postStr  = isMass ? g.fitted.toExponential(3) : g.fitted.toFixed(2);

            var cls = '';
            if (!g.within_sigma) {
                cls = g.sigma_excess <= 1.0 ? 'sigma-warn' : 'sigma-bad';
            } else {
                cls = 'sigma-good';
            }

            html += '<tr class="' + cls + '">'
                + '<td style="text-align:left;">' + g.name + '</td>'
                + '<td>' + priorStr + '</td>'
                + '<td>' + postStr + '</td>'
                + '<td>' + pctStr + '</td>'
                + '<td>' + sigStr + '</td>'
                + '<td>' + chi2Str + '</td>'
                + '</tr>';
        }
        tbody.innerHTML = html;
    }

    // Show the pane and activate split layout
    metricsPaneTop.style.display = '';
    metricsPaneDivider.style.display = '';

    // Set initial 50/50 split
    var panelH = metricsPanel.getBoundingClientRect().height;
    metricsPaneTop.style.height = Math.floor(panelH * 0.5) + 'px';
    metricsPaneTop.style.flex = 'none';
    metricsPaneBottom.style.flex = '1';
}

/**
 * Hide the auto-map diagnostics pane.
 */
function hideAutoMapDiagnostics() {
    if (!metricsPaneTop) return;
    metricsPaneTop.style.display = 'none';
    metricsPaneDivider.style.display = 'none';
    metricsPaneTop.style.height = '';
    metricsPaneTop.style.flex = '';
    metricsPaneBottom.style.flex = '';
}

function blankRightPaneMetrics() {
    if (metricsPaneTop) {
        metricsPaneTop.style.display = 'none';
        metricsPaneDivider.style.display = 'none';
    }
    if (metricsPaneBottom) {
        metricsPaneBottom.innerHTML = '';
    }
}

// =====================================================================
// THEORY TOGGLE BAR
// =====================================================================
//
// Maps each toggle chip's data-series attribute to the Chart.js dataset
// indices it controls. Toggling a chip shows or hides those datasets.

/**
 * Check whether a theory toggle chip is currently enabled (checked).
 * Returns true if the chip is checked or if no chip exists for the key.
 */
function isChipEnabled(seriesKey) {
    var chip = document.querySelector('.theory-chip[data-series="' + seriesKey + '"]');
    if (!chip) return true;
    var cb = chip.querySelector('input[type="checkbox"]');
    return cb ? cb.checked : true;
}

var theoryDatasetMap = {
    'observed':      [3],       // Observed Data points
    'newtonian':     [0],       // Newtonian Gravity
    'gfd':           [1],       // GFD (Photometric)
    'gfd_sigma_old': [8],       // Legacy Bayesian GFD Sigma (hidden)
    'gfd_spline':    [10],      // GFD Sigma (fast observation fit)
    'gfd_accel':     [11],      // GFD (with acceleration) -- 7-param fit
    'gfd_symmetric': [9],       // Legacy GFD (Observed) - hidden
    'mond':          [2],       // Classical MOND
    'cdm':           [7]        // CDM + NFW
};

/**
 * Force a theory chip on by data-series key.
 * Ensures the checkbox is checked, the chip has the active class,
 * the corresponding datasets are visible, and adds a "locked" CSS
 * class so the user can see the chip is non-toggleable.
 */
function forceChipOn(seriesKey) {
    var chip = document.querySelector('.theory-chip[data-series="' + seriesKey + '"]');
    if (!chip) return;
    // Unhide the chip if it was hidden via inline style
    chip.style.display = '';
    var cb = chip.querySelector('input[type="checkbox"]');
    if (cb) cb.checked = true;
    chip.classList.add('active');
    chip.classList.add('locked');
    var indices = theoryDatasetMap[seriesKey] || [];
    for (var j = 0; j < indices.length; j++) {
        chart.data.datasets[indices[j]].hidden = false;
    }
}

/**
 * Remove the "locked" CSS class from all theory chips.
 * Called when leaving inference mode so chips become toggleable again.
 */
function unlockAllChips() {
    var chips = document.querySelectorAll('.theory-chip.locked');
    for (var i = 0; i < chips.length; i++) {
        chips[i].classList.remove('locked');
    }
}

/**
 * Initialize theory toggle chips: sync checked state with active class,
 * attach click handlers to toggle chart dataset visibility.
 */
function initTheoryToggles() {
    var chips = document.querySelectorAll('.theory-chip');
    for (var i = 0; i < chips.length; i++) {
        var chip = chips[i];
        var checkbox = chip.querySelector('input[type="checkbox"]');

        // Set initial active class from checkbox state
        if (checkbox && checkbox.checked) {
            chip.classList.add('active');
        }

        // Click handler: toggle dataset visibility.
        // preventDefault() stops the <label> from toggling the checkbox
        // on its own, so we control the state exactly once per click.
        chip.addEventListener('click', function(e) {
            e.preventDefault();
            var seriesKey = this.getAttribute('data-series');

            var cb = this.querySelector('input[type="checkbox"]');
            cb.checked = !cb.checked;

            var indices = theoryDatasetMap[seriesKey] || [];
            var hidden = !cb.checked;

            // Apply active class for styling
            if (cb.checked) {
                this.classList.add('active');
            } else {
                this.classList.remove('active');
            }

            // Show or hide each mapped dataset
            for (var j = 0; j < indices.length; j++) {
                chart.data.datasets[indices[j]].hidden = hidden;
            }
            chart.update('none');
        });
    }
}

// =====================================================================
// CROSSHAIR READOUT
// =====================================================================
//
// Shows a floating panel near the cursor with interpolated values from
// each visible theory curve at the current x (radius) position.

var crosshairReadout = null;

/**
 * Create the crosshair readout DOM element and append to chart container.
 */
function initCrosshairReadout() {
    var container = document.querySelector('.chart-container');
    if (!container) return;

    crosshairReadout = document.createElement('div');
    crosshairReadout.className = 'crosshair-readout';
    container.appendChild(crosshairReadout);

    // Track mouse movement over the canvas
    var canvasEl = document.getElementById('gravityChart');
    if (!canvasEl) return;

    canvasEl.addEventListener('mousemove', function(e) {
        if (!crosshairReadout) return;

        // Get the x-axis scale to convert pixel to data coordinates
        var xScale = chart.scales.x;
        var yScale = chart.scales.y;
        if (!xScale || !yScale) return;

        // Convert mouse x to chart area pixel offset
        var rect = canvasEl.getBoundingClientRect();
        var pixelX = e.clientX - rect.left;

        // Only show readout when mouse is within the plot area
        if (pixelX < xScale.left || pixelX > xScale.right) {
            crosshairReadout.classList.remove('visible');
            return;
        }

        var radius = xScale.getValueForPixel(pixelX);
        if (radius < 0) {
            crosshairReadout.classList.remove('visible');
            return;
        }

        // Build readout content with one row per visible series
        var html = '<div style="margin-bottom:4px;color:#4da6ff;font-weight:600;">r = ' +
                   radius.toFixed(1) + ' kpc</div>';

        // Series definitions: name, dataset index, color, label
        var seriesDefs = [
            {key: 'observed',      idx: 3, color: '#FFC107', label: 'Observed'},
            {key: 'newtonian',     idx: 0, color: '#ff6b6b', label: 'Newtonian'},
            {key: 'gfd',           idx: 1, color: '#4da6ff', label: 'GFD'},
            {key: 'gfd_symmetric', idx: 9, color: '#00E5FF', label: 'GFD (Observed)'},
            {key: 'mond',          idx: 2, color: '#9966ff', label: 'MOND'},
            {key: 'cdm',           idx: 7, color: '#ffffff', label: 'CDM'}
        ];

        var hasValues = false;
        for (var i = 0; i < seriesDefs.length; i++) {
            var def = seriesDefs[i];
            var ds = chart.data.datasets[def.idx];
            if (!ds || ds.hidden || !ds.data || ds.data.length === 0) continue;

            // For observed data, find nearest point instead of interpolating
            var val = null;
            if (def.key === 'observed') {
                var nearest = null;
                var nearestDist = Infinity;
                for (var p = 0; p < ds.data.length; p++) {
                    var dist = Math.abs(ds.data[p].x - radius);
                    if (dist < nearestDist) {
                        nearestDist = dist;
                        nearest = ds.data[p];
                    }
                }
                // Only show if within 1 kpc of a data point
                if (nearest && nearestDist < 1.0) {
                    var errStr = nearest.err ? ' +/- ' + nearest.err.toFixed(1) : '';
                    html += '<div class="crosshair-readout-row">' +
                            '<span class="crosshair-dot" style="background:' + def.color + '"></span>' +
                            '<span class="crosshair-label">' + def.label + '</span>' +
                            '<span class="crosshair-value">' + nearest.y.toFixed(1) + errStr + ' km/s</span>' +
                            '</div>';
                    hasValues = true;
                }
                continue;
            }

            // For theory curves, interpolate
            val = interpolateCurve(def.idx, radius);
            if (val !== null) {
                html += '<div class="crosshair-readout-row">' +
                        '<span class="crosshair-dot" style="background:' + def.color + '"></span>' +
                        '<span class="crosshair-label">' + def.label + '</span>' +
                        '<span class="crosshair-value">' + val.toFixed(1) + ' km/s</span>' +
                        '</div>';
                hasValues = true;
            }
        }

        if (!hasValues) {
            crosshairReadout.classList.remove('visible');
            return;
        }

        crosshairReadout.innerHTML = html;
        crosshairReadout.classList.add('visible');

        // Position readout near cursor, offset to the right.
        // Use chart container as reference for absolute positioning.
        var containerRect = container.getBoundingClientRect();
        var readoutLeft = e.clientX - containerRect.left + 16;
        var readoutTop = e.clientY - containerRect.top - 20;

        // Clamp so it doesn't overflow the container
        var readoutWidth = crosshairReadout.offsetWidth || 200;
        var readoutHeight = crosshairReadout.offsetHeight || 100;
        if (readoutLeft + readoutWidth > containerRect.width - 10) {
            readoutLeft = e.clientX - containerRect.left - readoutWidth - 16;
        }
        if (readoutTop + readoutHeight > containerRect.height - 10) {
            readoutTop = containerRect.height - readoutHeight - 10;
        }
        if (readoutTop < 10) readoutTop = 10;

        crosshairReadout.style.left = readoutLeft + 'px';
        crosshairReadout.style.top = readoutTop + 'px';
    });

    // Hide readout when mouse leaves the canvas
    canvasEl.addEventListener('mouseleave', function() {
        if (crosshairReadout) {
            crosshairReadout.classList.remove('visible');
        }
    });
}

// =====================================================================
// INITIALIZATION
// =====================================================================

var SPLASH_MIN_DISPLAY_MS = 3000;
var LIBRARIES_MAX_WAIT_MS = 15000;

function librariesReady() {
    return typeof Chart !== 'undefined' && typeof Hammer !== 'undefined' && typeof katex !== 'undefined';
}

function waitForLibraries() {
    if (librariesReady()) return Promise.resolve();
    return new Promise(function(resolve, reject) {
        var start = Date.now();
        var t = setInterval(function() {
            if (librariesReady()) {
                clearInterval(t);
                resolve();
                return;
            }
            if (Date.now() - start >= LIBRARIES_MAX_WAIT_MS) {
                clearInterval(t);
                resolve();
            }
        }, 50);
    });
}

function dismissSplash() {
    var splash = document.getElementById('splash-overlay');
    if (!splash) return;
    var elapsed = performance.now();
    var remaining = Math.max(0, SPLASH_MIN_DISPLAY_MS - elapsed);
    setTimeout(function() {
        splash.style.opacity = '0';
        setTimeout(function() { splash.remove(); }, 500);
    }, remaining);
}

async function init() {
    await waitForLibraries();

    // Initialize UI components before data loads
    initViewModeToggle();
    initRightPanelTabs();
    initObsTabs();
    initVortexChipListeners();
    initObsFontControls();
    initTheoryToggles();
    initCrosshairReadout();

    try {
        galaxyCatalog = await fetchGalaxies();
    } catch (err) {
        console.error('Failed to load galaxy catalog:', err);
        // Fallback: empty catalog
        galaxyCatalog = { prediction: [], inference: [] };
    }

    updateExamplesDropdown();

    // Auto-load Milky Way so the chart is never empty on first visit
    const dropdown = document.getElementById('examples-dropdown');
    if (dropdown.options.length > 1) {
        dropdown.value = '1';
        loadExample();
    } else {
        updateDisplays();
        updateChart();
    }

    dismissSplash();
}

// Expose functions to global scope for onclick handlers
window.loadExample = loadExample;
window.runAutoFit = runAutoFit;
window.enterObservationMode = enterObservationMode;
window.enterMassModelMode = enterMassModelMode;

init();
