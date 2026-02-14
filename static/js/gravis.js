/**
 * GRAVIS - GRAvity VISion
 * Frontend JavaScript: fetches rotation curve data from Flask API,
 * renders charts with Chart.js.
 *
 * All physics computations happen server-side via /api/rotation-curve
 * and /api/infer-mass-model. The frontend handles only:
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
let lastCdmHalo = null;
let lastMultiResult = null;     // cached multi-point inference result
let lastModelTotal = 0;         // cached anchor model total mass
let lastAccelRatio = 1.0;       // cached accel ratio for band recompute
let lastMassModel = null;       // cached mass model for band recompute

// Debounce timer for API calls
let updateTimer = null;
const DEBOUNCE_MS = 80;

// =====================================================================
// DOM ELEMENTS
// =====================================================================
const distanceSlider = document.getElementById('distance-slider');
const massSlider = document.getElementById('mass-slider');
const velocitySlider = document.getElementById('velocity-slider');
const accelSlider = document.getElementById('accel-slider');
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
    return coeff.toFixed(2) + 'e' + exp;
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
    document.getElementById('bulge-scale-value').textContent = parseFloat(bulgeScaleSlider.value).toFixed(1) + ' kpc';
    document.getElementById('disk-mass-value').textContent = formatMassCompact(parseFloat(diskMassSlider.value)) + ' M_sun';
    document.getElementById('disk-scale-value').textContent = parseFloat(diskScaleSlider.value).toFixed(1) + ' kpc';
    document.getElementById('gas-mass-value').textContent = formatMassCompact(parseFloat(gasMassSlider.value)) + ' M_sun';
    document.getElementById('gas-scale-value').textContent = parseFloat(gasScaleSlider.value).toFixed(1) + ' kpc';

    const totalExp = Math.floor(Math.log10(total));
    const totalCoeff = total / Math.pow(10, totalExp);
    document.getElementById('mass-model-total-value').textContent =
        totalCoeff.toFixed(1) + 'e' + totalExp + ' M_sun';

    const logTotal = Math.log10(total);
    massSlider.value = logTotal;
}

// Update mass model display labels only (no side effects, for inference auto-scaling)
function updateMassModelDisplaysOnly() {
    const bm = Math.pow(10, parseFloat(bulgeMassSlider.value));
    const dm = Math.pow(10, parseFloat(diskMassSlider.value));
    const gm = Math.pow(10, parseFloat(gasMassSlider.value));
    const total = bm + dm + gm;

    document.getElementById('bulge-mass-value').textContent = formatMassCompact(parseFloat(bulgeMassSlider.value)) + ' M_sun';
    document.getElementById('bulge-scale-value').textContent = parseFloat(bulgeScaleSlider.value).toFixed(1) + ' kpc';
    document.getElementById('disk-mass-value').textContent = formatMassCompact(parseFloat(diskMassSlider.value)) + ' M_sun';
    document.getElementById('disk-scale-value').textContent = parseFloat(diskScaleSlider.value).toFixed(1) + ' kpc';
    document.getElementById('gas-mass-value').textContent = formatMassCompact(parseFloat(gasMassSlider.value)) + ' M_sun';
    document.getElementById('gas-scale-value').textContent = parseFloat(gasScaleSlider.value).toFixed(1) + ' kpc';

    const totalExp = Math.floor(Math.log10(total));
    const totalCoeff = total / Math.pow(10, totalExp);
    document.getElementById('mass-model-total-value').textContent =
        totalCoeff.toFixed(1) + 'e' + totalExp + ' M_sun';
}

// Mass model slider listeners (always active)
// In prediction mode: all sliders directly control the model
// In inference mode: scale length sliders control shape, masses are auto-computed
[bulgeMassSlider, bulgeScaleSlider, diskMassSlider, diskScaleSlider, gasMassSlider, gasScaleSlider].forEach(slider => {
    slider.addEventListener('input', () => {
        if (!isLoadingExample && currentExample) {
            // User is fine-tuning: keep pinned observations visible but mark
            // that we're no longer at the exact example configuration
            currentExample = null;
            // Don't reset the dropdown or pinned data -- observations stay
        }
        if (currentMode === 'prediction') {
            updateMassModelDisplays();
        }
        // In inference mode, scale length changes trigger re-inference
        // which will auto-update the mass values
        debouncedUpdateChart();
    });
});

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
            if (!point.err) return;
            const element = meta.data[index];
            if (!element) return;
            const x = element.x;
            const yScale = chart.scales.y;
            const yTop = yScale.getPixelForValue(point.y + point.err);
            const yBottom = yScale.getPixelForValue(point.y - point.err);
            ctx.beginPath(); ctx.moveTo(x, yTop); ctx.lineTo(x, yBottom); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(x - 4, yTop); ctx.lineTo(x + 4, yTop); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(x - 4, yBottom); ctx.lineTo(x + 4, yBottom); ctx.stroke();
        });
        ctx.restore();
    }
};
Chart.register(errorBarPlugin);

// =====================================================================
// CHART INITIALIZATION
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

    // Section 1: Velocity comparison
    html += ttSection('Velocity Comparison');
    html += '<table style="width:100%;padding:4px 12px 6px;border-spacing:0;">';
    var errStr = (err > 0) ? ' \u00B1 ' + err : '';
    html += ttRow('<span style="color:#ffa726;">\u25CF</span> Observed', vObs.toFixed(1) + errStr + ' km/s', '#ffa726');

    var gfdV = interpolateCurve(1, r);
    var newtonV = interpolateCurve(0, r);
    var mondV = interpolateCurve(2, r);
    var cdmV = interpolateCurve(7, r);
    if (gfdV !== null)    html += ttRow('<span style="color:#4da6ff;">\u25CF</span> GFD', gfdV.toFixed(1) + ' km/s', '#4da6ff');
    if (newtonV !== null) html += ttRow('<span style="color:#ef5350;">\u25CF</span> Newton', newtonV.toFixed(1) + ' km/s', '#ef5350');
    if (mondV !== null)   html += ttRow('<span style="color:#ab47bc;">\u25CF</span> MOND', mondV.toFixed(1) + ' km/s', '#ab47bc');
    if (cdmV !== null && chart.data.datasets[7].data.length > 0) {
        html += ttRow('<span style="color:#ffffff;">\u25CF</span> CDM+NFW', cdmV.toFixed(1) + ' km/s', '#ffffff');
    }
    html += '</table>';

    // Section 2: GFD Agreement
    if (gfdV !== null) {
        html += ttSection('GFD Agreement');
        var delta = gfdV - vObs;
        var resStr = (delta >= 0 ? '+' : '') + delta.toFixed(1) + ' km/s';
        html += '<div style="padding:4px 12px 6px;">';
        html += '<span style="color:#e0e0e0;">GFD &minus; Obs: </span>';
        html += '<span style="font-weight:600;color:#e0e0e0;">' + resStr + '</span>';
        if (err > 0) {
            var sigAway = Math.abs(delta) / err;
            var sigColor = sigAway < 1.0 ? '#4caf50'
                         : sigAway < 2.0 ? '#8bc34a'
                         : sigAway < 3.0 ? '#ffa726'
                         : '#ef5350';
            var sigLabel = sigAway < 1.0 ? 'within 1\u03C3'
                         : sigAway < 2.0 ? 'within 2\u03C3'
                         : sigAway < 3.0 ? '2-3\u03C3'
                         : '> 3\u03C3';
            html += '<br><span style="font-size:11px;color:' + sigColor + ';font-weight:600;">'
                + sigAway.toFixed(1) + '\u03C3 &mdash; ' + sigLabel + '</span>';
        }
        html += '</div>';
    }

    // Section 3: Theory rankings (closest to observation)
    var theories = [];
    if (gfdV !== null)    theories.push({name: 'GFD',     v: gfdV,    color: '#4da6ff'});
    if (newtonV !== null) theories.push({name: 'Newton',  v: newtonV, color: '#ef5350'});
    if (mondV !== null)   theories.push({name: 'MOND',    v: mondV,   color: '#ab47bc'});
    if (cdmV !== null && chart.data.datasets[7].data.length > 0) {
        theories.push({name: 'CDM+NFW', v: cdmV, color: '#ffffff'});
    }
    if (theories.length > 1) {
        theories.sort(function(a, b) { return Math.abs(a.v - vObs) - Math.abs(b.v - vObs); });
        html += ttSection('Closest to Observed');
        html += '<table style="width:100%;padding:4px 12px 6px;border-spacing:0;">';
        for (var i = 0; i < theories.length; i++) {
            var t = theories[i];
            var res = t.v - vObs;
            var resLabel = (res >= 0 ? '+' : '') + res.toFixed(1) + ' km/s';
            var rankStr = '';
            html += '<tr>'
                + '<td style="padding:2px 6px 2px 0;color:' + t.color + ';white-space:nowrap;">' + t.name + '</td>'
                + '<td style="padding:2px 4px;color:#e0e0e0;text-align:right;font-variant-numeric:tabular-nums;">' + resLabel + '</td>'
                + '<td style="padding:2px 0 2px 6px;color:#4caf50;font-size:11px;">' + rankStr + '</td>'
                + '</tr>';
        }
        html += '</table>';
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
                pointRadius: 0
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
                pointRadius: 0
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
            // Dataset 4: GFD confidence band upper edge (hidden from legend)
            {
                label: 'GFD +1\u03C3',
                data: [],
                borderColor: 'rgba(76, 175, 80, 0.35)',
                backgroundColor: 'rgba(76, 175, 80, 0.12)',
                borderWidth: 1,
                borderDash: [4, 4],
                tension: 0.4,
                pointRadius: 0,
                fill: {target: 5, above: 'rgba(76, 175, 80, 0.12)', below: 'rgba(76, 175, 80, 0.12)'}
            },
            // Dataset 5: GFD confidence band lower edge (hidden from legend)
            {
                label: 'GFD -1\u03C3',
                data: [],
                borderColor: 'rgba(76, 175, 80, 0.35)',
                backgroundColor: 'transparent',
                borderWidth: 1,
                borderDash: [4, 4],
                tension: 0.4,
                pointRadius: 0,
                fill: false
            },
            // Dataset 6: Inference markers (green diamonds)
            {
                label: 'GFD Inference',
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
                pointRadius: 0
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: 'top',
                align: 'end',
                labels: {
                    color: '#e0e0e0',
                    font: { size: 12 },
                    padding: 15,
                    usePointStyle: true,
                    boxWidth: 40,
                    boxHeight: 3,
                    filter: function(item, chartData) {
                        // Hide confidence band datasets from legend
                        if (item.datasetIndex === 4 || item.datasetIndex === 5) return false;
                        // Hide inference markers from legend when no data
                        if (item.datasetIndex === 6) {
                            var ds = chartData.datasets[6];
                            return ds && ds.data && ds.data.length > 0;
                        }
                        return true;
                    },
                    // Custom styling for CDM dashed line in legend
                    pointStyleWidth: 0
                }
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
            zoom: {
                pan: {
                    enabled: true,
                    mode: 'x',
                    modifierKey: 'shift',
                    onPanComplete: ({chart}) => {
                        const resetBtn = document.getElementById('reset-zoom-btn');
                        if (resetBtn) resetBtn.style.display = 'block';
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
                    onZoomComplete: ({chart}) => {
                        const resetBtn = document.getElementById('reset-zoom-btn');
                        if (resetBtn) resetBtn.style.display = 'block';
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

// =====================================================================
// ZOOM CONTROLS
// =====================================================================

const canvas = document.getElementById('gravityChart');
const resetZoomBtn = document.getElementById('reset-zoom-btn');

function hideResetButton() { resetZoomBtn.style.display = 'none'; }
function resetChartZoom() { chart.resetZoom(); hideResetButton(); }

resetZoomBtn.addEventListener('click', resetChartZoom);
canvas.addEventListener('dblclick', resetChartZoom);

// =====================================================================
// API COMMUNICATION
// =====================================================================

async function fetchRotationCurve(maxRadius, accelRatio, massModel, observations) {
    var body = {
        max_radius: maxRadius,
        num_points: 100,
        accel_ratio: accelRatio,
        mass_model: massModel
    };
    if (observations) {
        body.observations = observations;
    }
    const resp = await fetch('/api/rotation-curve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    if (!resp.ok) throw new Error('API error: ' + resp.status);
    return resp.json();
}

async function fetchInferredMassModel(rKpc, vKmS, accelRatio, massModel) {
    const resp = await fetch('/api/infer-mass-model', {
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
    const resp = await fetch('/api/galaxies');
    if (!resp.ok) throw new Error('API error: ' + resp.status);
    return resp.json();
}

async function fetchMultiPointInference(observations, accelRatio, massModel) {
    const resp = await fetch('/api/infer-mass-multi', {
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
    // Beyond curve range
    return data[data.length - 1].y;
}

/** Shorthand: interpolate GFD curve (dataset 1) */
function interpolateGFDVelocity(radius) {
    return interpolateCurve(1, radius);
}

async function runMultiPointInference(accelRatio, massModel) {
    var multiDiv = document.getElementById('multi-inference-result');
    var multiBody = document.getElementById('multi-inference-body');
    if (!multiDiv || !multiBody) return;

    if (currentMode !== 'inference') {
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

        // Green inference markers -- enriched meta for tooltips
        var markerData = [];
        for (var i = 0; i < result.points.length; i++) {
            var pt = result.points[i];
            var gfdV = interpolateGFDVelocity(pt.r_kpc);
            if (gfdV === null) continue;
            var deviation = ((pt.inferred_total - modelTotal) / modelTotal) * 100;
            // Sigma calculation: how many error bars away is GFD from observation
            var sigmaAway = (pt.err && pt.err > 0)
                ? Math.abs(gfdV - pt.v_km_s) / pt.err
                : null;
            markerData.push({
                x: pt.r_kpc, y: gfdV,
                meta: {
                    obs_v: pt.v_km_s, err: pt.err,
                    inferred_total: pt.inferred_total,
                    log10_total: pt.log10_total,
                    enclosed_frac: pt.enclosed_frac,
                    deviation: deviation,
                    gfd_total: modelTotal,
                    sigma_away: sigmaAway
                }
            });
        }
        chart.data.datasets[6].data = markerData;

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
        'weighted_rms':     {label: 'Weighted RMS',  desc: 'Spread of per-point inferences vs GFD prediction, weighted by enclosed fraction'},
        'weighted_scatter': {label: '1-sigma Scatter', desc: 'Weighted std dev of per-point inferences around their mean'},
        'obs_error':        {label: 'Obs. Error',   desc: 'Propagated velocity measurement uncertainties through the field equation'},
        'min_max':          {label: 'Min-Max',       desc: 'Full range of per-point inferred masses (most conservative)'},
        'iqr':              {label: 'IQR (Robust)',  desc: 'Interquartile range, resistant to outlier inner points'}
    };
    return labels[method] || {label: method, desc: ''};
}

/**
 * Recompute and render the confidence band using the currently selected method.
 */
async function updateBand() {
    if (!lastMultiResult || !lastMassModel || lastModelTotal <= 0) {
        chart.data.datasets[4].data = [];
        chart.data.datasets[5].data = [];
        chart.update('none');
        return;
    }

    var method = document.getElementById('band-method-select').value;
    var halfWidth = getBandHalfWidth(method);

    if (halfWidth <= 0) {
        chart.data.datasets[4].data = [];
        chart.data.datasets[5].data = [];
        chart.update('none');
        return;
    }

    var scaleHigh = (lastModelTotal + halfWidth) / lastModelTotal;
    var scaleLow = Math.max(0.01, (lastModelTotal - halfWidth) / lastModelTotal);
    var modelHigh = scaleMassModel(lastMassModel, scaleHigh);
    var modelLow = scaleMassModel(lastMassModel, scaleLow);

    var chartMaxR = parseFloat(distanceSlider.value);
    var observations = pinnedObservations || (currentExample ? getPredictionObservations(currentExample) : null);
    if (observations && observations.length > 0) {
        var maxObsR = Math.max.apply(null, observations.map(function(o) { return o.r; }));
        chartMaxR = Math.max(chartMaxR, maxObsR * 1.15);
    }

    var bandResults = await Promise.all([
        fetchRotationCurve(chartMaxR, lastAccelRatio, modelHigh),
        fetchRotationCurve(chartMaxR, lastAccelRatio, modelLow)
    ]);

    var upperData = [], lowerData = [];
    for (var i = 0; i < bandResults[0].radii.length; i++) {
        upperData.push({x: bandResults[0].radii[i], y: bandResults[0].dtg[i]});
    }
    for (var i = 0; i < bandResults[1].radii.length; i++) {
        lowerData.push({x: bandResults[1].radii[i], y: bandResults[1].dtg[i]});
    }
    chart.data.datasets[4].data = upperData;
    chart.data.datasets[5].data = lowerData;
    chart.update('none');

    // Update the band label in the sidebar
    updateBandLabel();
}

/**
 * Update just the band width display in the sidebar.
 */
function updateBandLabel() {
    var el = document.getElementById('band-width-display');
    if (!el || lastModelTotal <= 0) return;
    var method = document.getElementById('band-method-select').value;
    var halfWidth = getBandHalfWidth(method);
    var info = getBandMethodInfo(method);
    var anchorExp = Math.floor(Math.log10(lastModelTotal));
    var hwCoeff = halfWidth / Math.pow(10, anchorExp);
    el.innerHTML = '<strong style="color:#e0e0e0;">Band (' + info.label + '):</strong> '
        + '<span style="color:#4caf50;">\u00B1 ' + hwCoeff.toFixed(2)
        + ' \u00D7 10' + superscript(anchorExp) + ' M\u2609</span>'
        + '<div style="font-size:0.8em; color:#606060; margin-top:2px;">' + info.desc + '</div>';
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
    html += '<span style="color:#4caf50;">Diamonds</span> = GFD velocity; hover for details.';
    html += '</div>';

    html += '<div style="margin-bottom: 6px;">';
    html += '<strong style="color:#e0e0e0;">GFD Prediction:</strong> ';
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
    html += '<strong style="color:#e0e0e0;">Offset:</strong> ';
    html += '<span style="color:' + agColor + ';">' + agSign + agreementPct.toFixed(1) + '%</span>';
    html += '</div>';

    // Per-point table
    html += '<table style="width:100%; border-collapse:collapse; margin-top:8px; font-size:0.85em;">';
    html += '<tr style="border-bottom:1px solid #404040; color:#808080;">';
    html += '<th style="text-align:left; padding:4px 6px;">r (kpc)</th>';
    html += '<th style="text-align:left; padding:4px 6px;">v (km/s)</th>';
    html += '<th style="text-align:right; padding:4px 6px;">log10(M)</th>';
    html += '<th style="text-align:right; padding:4px 6px;">M enc.</th>';
    html += '<th style="text-align:right; padding:4px 6px;">vs GFD</th>';
    html += '</tr>';

    for (var i = 0; i < result.points.length; i++) {
        var pt = result.points[i];
        var deviation = ((pt.inferred_total - modelTotal) / modelTotal) * 100;
        var devColor = Math.abs(deviation) < 10 ? '#4caf50' : Math.abs(deviation) < 25 ? '#ffa726' : '#ef5350';
        var errStr = pt.err ? ' \u00B1 ' + pt.err : '';
        var encPct = (pt.enclosed_frac * 100).toFixed(0);
        var rowOpacity = pt.enclosed_frac < 0.1 ? 'opacity: 0.5;' : '';
        html += '<tr style="border-bottom:1px solid #2a2a2a; ' + rowOpacity + '">';
        html += '<td style="padding:3px 6px; color:#e0e0e0;">' + pt.r_kpc + '</td>';
        html += '<td style="padding:3px 6px; color:#e0e0e0;">' + pt.v_km_s + errStr + '</td>';
        html += '<td style="text-align:right; padding:3px 6px; color:#b0b0b0;">' + pt.log10_total.toFixed(2) + '</td>';
        html += '<td style="text-align:right; padding:3px 6px; color:#808080;">' + encPct + '%</td>';
        html += '<td style="text-align:right; padding:3px 6px; color:' + devColor + ';">' + (deviation >= 0 ? '+' : '') + deviation.toFixed(1) + '%</td>';
        html += '</tr>';
    }
    html += '</table>';

    html += '<div style="margin-top: 8px; font-size: 0.8em; color: #606060;">';
    html += 'M enc. = % of model mass within this radius. vs GFD = deviation from prediction.';
    html += '</div>';

    multiBody.innerHTML = html;
    multiDiv.style.display = 'block';

    // Initialize band label for current method
    updateBandLabel();
}

/**
 * Clear inference-specific chart elements (band + markers).
 */
function clearInferenceChart() {
    if (chart.data.datasets.length > 4) {
        chart.data.datasets[4].data = [];
        chart.data.datasets[5].data = [];
    }
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

    if (currentMode === 'inference') {
        // Inference mode: use mass distribution shape from sliders,
        // auto-scale component masses to match observation via DTG
        const r_obs = parseFloat(distanceSlider.value);
        const v_obs = parseFloat(velocitySlider.value);
        const shapeModel = getMassModelFromSliders();

        try {
            // Call the distributed inference endpoint
            const inferResult = await fetchInferredMassModel(r_obs, v_obs, accelRatio, shapeModel);

            // Update inferred mass display
            const totalMass = inferResult.inferred_total;
            const massExp = Math.floor(Math.log10(totalMass));
            const massCoeff = totalMass / Math.pow(10, massExp);
            inferredMassValue.textContent = massCoeff.toFixed(2) + '\u00D7' + '10' + superscript(massExp) + ' M\u2609';

            // Update BTFR display
            const btfrEl = document.getElementById('btfr-mass-value');
            if (btfrEl) {
                const btfr = inferResult.btfr_mass;
                const btfrExp = Math.floor(Math.log10(btfr));
                const btfrCoeff = btfr / Math.pow(10, btfrExp);
                btfrEl.textContent = btfrCoeff.toFixed(2) + '\u00D7' + '10' + superscript(btfrExp) + ' M\u2609';
            }

            // Update mass sliders to show the inferred (scaled) values
            // but only update masses, not scale lengths (user controls the shape)
            const scaledModel = inferResult.inferred_mass_model;
            if (scaledModel && !isLoadingExample) {
                // Silently update mass slider positions without triggering events
                if (scaledModel.bulge && scaledModel.bulge.M > 0)
                    bulgeMassSlider.value = Math.log10(Math.max(scaledModel.bulge.M, 1e7));
                if (scaledModel.disk && scaledModel.disk.M > 0)
                    diskMassSlider.value = Math.log10(Math.max(scaledModel.disk.M, 1e7));
                if (scaledModel.gas && scaledModel.gas.M > 0)
                    gasMassSlider.value = Math.log10(Math.max(scaledModel.gas.M, 1e7));
                // Update display labels (but don't retrigger updateChart)
                updateMassModelDisplaysOnly();
            }

            // Compute full rotation curve with the inferred distributed model
            const curveModel = scaledModel || shapeModel;

            // Extend chart range to cover all observed data if available (use pinned if tweaking)
            var chartMaxR = maxRadius;
            var predObs = pinnedObservations || (currentExample ? getPredictionObservations(currentExample) : null);
            if (predObs && predObs.length > 0) {
                var maxObsR = Math.max.apply(null, predObs.map(function(o) { return o.r; }));
                chartMaxR = Math.max(chartMaxR, maxObsR * 1.15);
            }

            const data = await fetchRotationCurve(chartMaxR, accelRatio, curveModel, predObs);
            renderCurves(data);

            // Show observation points -- use pinned observations if user is fine-tuning
            var visibleObs = predObs;
            if (visibleObs && visibleObs.length > 0) {
                chart.data.datasets[3].data = visibleObs.map(function(obs) {
                    return {x: obs.r, y: obs.v, err: obs.err || 0};
                });
                chart.data.datasets[3].hidden = false;
                observedLegend.style.display = 'flex';
            } else if (currentExample) {
                chart.data.datasets[3].data = [{x: r_obs, y: v_obs}];
                chart.data.datasets[3].hidden = false;
                observedLegend.style.display = 'flex';
            } else {
                chart.data.datasets[3].data = [];
                chart.data.datasets[3].hidden = true;
                observedLegend.style.display = 'none';
            }

            // Multi-point consistency analysis (fire-and-forget, updates chart independently)
            runMultiPointInference(accelRatio, curveModel);

        } catch (err) {
            console.error('Inference API error:', err);
        }
    } else {
        // Prediction mode: use distributed mass model from sliders
        const massModel = getMassModelFromSliders();

        try {
            // Pass observations for CDM halo fitting when available (use pinned if tweaking)
            var predObs = pinnedObservations || (currentExample ? currentExample.observations : null);
            // Extend chart range to cover all observed data
            var predMaxR = maxRadius;
            if (predObs && predObs.length > 0) {
                var maxObsR = Math.max.apply(null, predObs.map(function(o) { return o.r; }));
                predMaxR = Math.max(predMaxR, maxObsR * 1.15);
            }
            const data = await fetchRotationCurve(predMaxR, accelRatio, massModel, predObs);
            renderCurves(data);

            // Handle observed data -- show pinned observations even when sliders are adjusted
            var visibleObs = pinnedObservations || (currentExample ? currentExample.observations : null);
            if (visibleObs && visibleObs.length > 0) {
                chart.data.datasets[3].data = visibleObs.map(obs => ({
                    x: obs.r, y: obs.v, err: obs.err || 0
                }));
                chart.data.datasets[3].hidden = false;
                observedLegend.style.display = 'flex';
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
}

function renderCurves(data) {
    const newtonianData = [];
    const dtgData = [];
    const mondData = [];
    const cdmData = [];

    for (let i = 0; i < data.radii.length; i++) {
        newtonianData.push({x: data.radii[i], y: data.newtonian[i]});
        dtgData.push({x: data.radii[i], y: data.dtg[i]});
        mondData.push({x: data.radii[i], y: data.mond[i]});
        if (data.cdm) {
            cdmData.push({x: data.radii[i], y: data.cdm[i]});
        }
    }

    chart.data.labels = [];
    chart.data.datasets[0].data = newtonianData;
    chart.data.datasets[1].data = dtgData;
    chart.data.datasets[2].data = mondData;
    chart.data.datasets[7].data = cdmData;

    // Store CDM halo info for display
    lastCdmHalo = data.cdm_halo || null;
    updateCdmHaloPanel();
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
    placeholder.textContent = 'Select an example...';
    dropdown.appendChild(placeholder);

    // Add galaxies for current mode
    const galaxies = galaxyCatalog[currentMode] || [];
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
        chart.options.plugins.title.text = 'Rotation Curve: Gravitational Theory Comparison';
        chart.options.plugins.zoom.limits.x.min = 0;
        chart.options.plugins.zoom.limits.x.max = 100;
        chart.resetZoom();
        updateDataCardButton();
        updateChart();
        return;
    }

    isLoadingExample = true;

    const galaxies = galaxyCatalog[currentMode] || [];
    const example = galaxies[selectedIndex - 1];
    if (!example) { isLoadingExample = false; return; }

    // Extract galaxy display name (strip mass info in parentheses)
    const galaxyLabel = example.name.replace(/\s*\(.*\)/, '');
    chart.options.plugins.title.text = galaxyLabel + ' \u2014 Rotation Curve';

    chart.data.datasets[3].data = [];
    chart.data.datasets[3].hidden = true;
    clearInferenceChart();

    currentExample = example;

    // Pin observations so they survive slider adjustments
    var pinObs = example.observations;
    if (!pinObs && currentMode === 'inference') {
        pinObs = getPredictionObservations(example);
    }
    pinnedObservations = pinObs || null;
    pinnedGalaxyLabel = galaxyLabel;
    pinnedGalaxyExample = example;

    distanceSlider.value = example.distance;
    accelSlider.value = example.accel;

    if (currentMode === 'prediction') {
        massSlider.value = example.mass || 11;
    } else {
        velocitySlider.value = example.velocity || 200;
    }

    // Set zoom limits based on observational data or distance
    // For inference mode, use prediction counterpart's observations if available
    var obsData = example.observations;
    if (!obsData && currentMode === 'inference') {
        obsData = getPredictionObservations(example);
    }
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

    updateDisplays();
    updateChart().then(() => {
        chart.resetZoom();
        hideResetButton();
        updateDataCardButton();
        isLoadingExample = false;
    });
}

// =====================================================================
// MODE SWITCHING
// =====================================================================

function setMode(mode) {
    // Remember the current galaxy id before switching (including pinned state)
    var previousId = currentExample ? currentExample.id :
                     (pinnedGalaxyExample ? pinnedGalaxyExample.id : null);

    currentMode = mode;
    currentExample = null;
    pinnedObservations = null;
    pinnedGalaxyLabel = null;
    pinnedGalaxyExample = null;

    document.getElementById('mode-prediction').classList.toggle('active', mode === 'prediction');
    document.getElementById('mode-inference').classList.toggle('active', mode === 'inference');

    if (mode === 'prediction') {
        velocityControl.style.display = 'none';
        inferenceResult.classList.remove('visible');
        document.getElementById('mass-model-section').style.display = '';
        document.getElementById('mass-slider-group').style.display = 'none';
        // Hide multi-point panel and chart overlays in prediction mode
        var multiDiv = document.getElementById('multi-inference-result');
        if (multiDiv) multiDiv.style.display = 'none';
        clearInferenceChart();
        // In prediction mode, mass sliders are directly editable
        setMassSliderEditable(true);
        // Update header text
        document.querySelector('.mass-model-header-text').textContent = 'Mass Distribution';
    } else {
        velocityControl.style.display = 'block';
        inferenceResult.classList.add('visible');
        // Show mass distribution in inference mode too (shape controls)
        document.getElementById('mass-model-section').style.display = '';
        document.getElementById('mass-slider-group').style.display = 'none';
        // In inference mode, mass values are auto-computed (read-only feel)
        // but scale lengths remain editable (user controls the shape)
        setMassSliderEditable(false);
        // Update header to indicate shape mode
        document.querySelector('.mass-model-header-text').textContent = 'Mass Distribution Shape (masses auto-inferred)';
    }

    updateExamplesDropdown();

    // If a galaxy was loaded, find the same galaxy in the new mode and load it
    if (previousId) {
        // Normalize id: strip _inference suffix and handle mw -> milky_way
        var baseId = previousId.replace(/_inference$/, '');
        if (baseId === 'mw') baseId = 'milky_way';
        var galaxies = galaxyCatalog[currentMode] || [];
        var matchIndex = galaxies.findIndex(function(g) {
            var gBase = g.id.replace(/_inference$/, '');
            if (gBase === 'mw') gBase = 'milky_way';
            return gBase === baseId;
        });
        if (matchIndex >= 0) {
            var dropdown = document.getElementById('examples-dropdown');
            dropdown.value = String(matchIndex + 1);
            loadExample();
            return;
        }
    }

    updateDisplays();
    updateChart();
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
    }
});

// =====================================================================
// NAVIGATION SYSTEM
// =====================================================================

function navigateToScreen(screenName) {
    const screens = document.querySelectorAll('.screen');
    screens.forEach(screen => screen.classList.remove('active'));

    const selectedScreen = document.getElementById('screen-' + screenName);
    if (selectedScreen) selectedScreen.classList.add('active');

    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        if (item.dataset.screen === screenName) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });

    window.location.hash = screenName;

    if (screenName === 'analysis') {
        setTimeout(() => chart.resize(), 100);
    }
}

window.addEventListener('hashchange', () => {
    const hash = window.location.hash.substring(1) || 'analysis';
    navigateToScreen(hash);
});

// =====================================================================
// INITIALIZATION
// =====================================================================

async function init() {
    try {
        galaxyCatalog = await fetchGalaxies();
    } catch (err) {
        console.error('Failed to load galaxy catalog:', err);
        // Fallback: empty catalog
        galaxyCatalog = { prediction: [], inference: [] };
    }

    updateExamplesDropdown();

    // Auto-load Milky Way
    const dropdown = document.getElementById('examples-dropdown');
    if (dropdown.options.length > 1) {
        dropdown.value = '1';
        loadExample();
    } else {
        updateDisplays();
        updateChart();
    }

    // Handle URL hash
    const hash = window.location.hash.substring(1) || 'analysis';
    navigateToScreen(hash);
}

// =====================================================================
// DATA PROVENANCE MODAL
// =====================================================================

function formatMass(m) {
    if (m === 0) return '0';
    const exp = Math.floor(Math.log10(m));
    const coeff = m / Math.pow(10, exp);
    return coeff.toFixed(2) + '\u00D710' + superscript(exp);
}

function openDataCard() {
    var ex = currentExample || pinnedGalaxyExample;
    if (!ex) return;

    // Title
    const label = ex.name.replace(/\s*\(.*\)/, '');
    document.getElementById('dc-title').textContent = label + ' \u2014 Data Provenance';

    // Summary
    const mm = ex.mass_model;
    const totalM = (mm.bulge ? mm.bulge.M : 0) + (mm.disk ? mm.disk.M : 0) + (mm.gas ? mm.gas.M : 0);
    const gasFrac = mm.gas ? ((mm.gas.M / totalM) * 100).toFixed(0) : '0';
    const nObs = ex.observations ? ex.observations.length : 0;
    const rMin = nObs > 0 ? ex.observations[0].r : 0;
    const rMax = nObs > 0 ? ex.observations[nObs - 1].r : 0;

    document.getElementById('dc-summary').innerHTML =
        'Total baryonic mass <strong>' + formatMass(totalM) + ' M<sub>sun</sub></strong>' +
        ' &mdash; gas fraction ' + gasFrac + '%' +
        ' &mdash; ' + nObs + ' observed data points spanning ' + rMin + '\u2013' + rMax + ' kpc.' +
        ' All masses are independently measured; no parameters have been fitted to the rotation curve.';

    // Mass model table
    const tbody = document.getElementById('dc-mass-tbody');
    tbody.innerHTML = '';
    if (mm.bulge && mm.bulge.M > 0) {
        tbody.innerHTML += '<tr><td>Stellar Bulge</td><td>' + formatMass(mm.bulge.M) +
            '</td><td>a = ' + mm.bulge.a + '</td><td>Hernquist</td></tr>';
    }
    if (mm.disk && mm.disk.M > 0) {
        tbody.innerHTML += '<tr><td>Stellar Disk</td><td>' + formatMass(mm.disk.M) +
            '</td><td>R<sub>d</sub> = ' + mm.disk.Rd + '</td><td>Exponential</td></tr>';
    }
    if (mm.gas && mm.gas.M > 0) {
        tbody.innerHTML += '<tr><td>Gas (HI+He)</td><td>' + formatMass(mm.gas.M) +
            '</td><td>R<sub>d</sub> = ' + mm.gas.Rd + '</td><td>Exponential</td></tr>';
    }
    document.getElementById('dc-total-mass').innerHTML =
        'Total: ' + formatMass(totalM) + ' M<sub>sun</sub>';

    // Observations table
    const obsTbody = document.getElementById('dc-obs-tbody');
    obsTbody.innerHTML = '';
    if (ex.observations) {
        ex.observations.forEach(obs => {
            obsTbody.innerHTML += '<tr><td>' + obs.r + '</td><td>' +
                obs.v + '</td><td>\u00B1 ' + obs.err + '</td></tr>';
        });
    }

    // References
    const refList = document.getElementById('dc-references');
    refList.innerHTML = '';
    if (ex.references) {
        ex.references.forEach(ref => {
            refList.innerHTML += '<li>' + ref + '</li>';
        });
    }

    // Methodology note
    document.getElementById('dc-methodology').innerHTML =
        '<strong>Methodology:</strong> Stellar masses derived from Spitzer 3.6 \u03BCm luminosity ' +
        'with fixed M*/L = 0.5 M<sub>sun</sub>/L<sub>sun</sub> (stellar population synthesis, not fitted). ' +
        'Gas masses from 21 cm HI observations with 1.33\u00D7 He correction ' +
        '(Big Bang nucleosynthesis). Rotation curves from tilted-ring model fits to HI velocity fields. ' +
        'No dark matter. No free parameters.';

    // Flip: hide chart, show data
    document.getElementById('chart-face').style.display = 'none';
    document.getElementById('data-face').style.display = 'block';
}

function closeDataCard() {
    // Flip back: show chart, hide data
    document.getElementById('data-face').style.display = 'none';
    document.getElementById('chart-face').style.display = 'block';
}

// Show/hide Data Sources button based on whether a galaxy is loaded
function updateDataCardButton() {
    const btn = document.getElementById('data-card-btn');
    if (btn) {
        // Show Data Sources when an example is loaded OR observations are pinned
        var refExample = currentExample || pinnedGalaxyExample;
        btn.style.display = (refExample && refExample.mass_model) ? 'block' : 'none';
    }
}

// Expose navigation function to global scope for onclick handlers
window.navigateToScreen = navigateToScreen;
window.setMode = setMode;
window.loadExample = loadExample;
window.openDataCard = openDataCard;
window.closeDataCard = closeDataCard;

init();
