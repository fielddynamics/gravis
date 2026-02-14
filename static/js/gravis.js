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
const distanceLabel = document.getElementById('distance-label');
const anchorRadiusInput = document.getElementById('anchor-radius');
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
    html += '<strong style="color:#e0e0e0;">Mass Offset:</strong> ';
    html += '<span style="color:' + agColor + ';">' + agSign + agreementPct.toFixed(1) + '%</span>';
    html += ' <span style="color:#606060; font-size:0.85em;">(obs. mean vs GFD prediction)</span>';
    html += '</div>';

    // Per-point table
    html += '<table style="width:100%; border-collapse:collapse; margin-top:8px; font-size:0.85em;">';
    html += '<tr style="border-bottom:1px solid #404040; color:#808080;">';
    html += '<th style="text-align:left; padding:4px 6px;">r (kpc)</th>';
    html += '<th style="text-align:left; padding:4px 6px;">v (km/s)</th>';
    html += '<th style="text-align:right; padding:4px 6px;">M enc.</th>';
    html += '<th style="text-align:right; padding:4px 6px;" title="Velocity residual: observed minus GFD prediction">\u0394v</th>';
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
        const r_obs = parseFloat(anchorRadiusInput.value);
        const v_obs = parseFloat(velocitySlider.value);
        const shapeModel = getMassModelFromSliders();

        // Update anchor display
        var anchorDisplay = document.getElementById('anchor-display');
        if (anchorDisplay) {
            anchorDisplay.textContent = v_obs + ' km/s at ' + r_obs + ' kpc';
        }

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

    accelSlider.value = example.accel;

    if (currentMode === 'prediction') {
        distanceSlider.value = example.distance;
        anchorRadiusInput.value = example.distance;
        massSlider.value = example.mass || 11;
    } else {
        // In inference mode: anchor radius = the observation point,
        // distance slider = chart range (match prediction counterpart)
        anchorRadiusInput.value = example.distance;
        velocitySlider.value = example.velocity || 200;

        // Find the prediction counterpart's distance for the chart range
        var predDistance = example.distance;
        var baseId = example.id.replace(/_inference$/, '');
        if (baseId === 'mw') baseId = 'milky_way';
        var predictions = galaxyCatalog.prediction || [];
        for (var i = 0; i < predictions.length; i++) {
            if (predictions[i].id === baseId) {
                predDistance = predictions[i].distance;
                break;
            }
        }
        // Use the larger of: prediction distance, or max obs radius + padding
        var chartDist = predDistance;
        var obsForRange = getPredictionObservations(example);
        if (obsForRange && obsForRange.length > 0) {
            var maxObsR = Math.max.apply(null, obsForRange.map(function(o) { return o.r; }));
            chartDist = Math.max(chartDist, maxObsR * 1.15);
        }
        distanceSlider.value = chartDist;
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

    // Update distance label for context
    distanceLabel.textContent = mode === 'inference' ? 'Chart Range (r)' : 'Distance Scale (r)';

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
window.toggleFaqItem = toggleFaqItem;
window.toggleAllFaq = toggleAllFaq;

init();
