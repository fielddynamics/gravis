/**
 * GRAVIS - GRAvity VISion
 * Frontend JavaScript: fetches rotation curve data from Flask API,
 * renders charts with Chart.js.
 *
 * All physics computations happen server-side via
 * /api/rotation/charts/mass_model. The frontend handles only:
 *   - UI state management (sliders, galaxy browser)
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
let currentExample = null;
let pinnedObservations = null;
let pinnedGalaxyLabel = null;
let pinnedGalaxyExample = null;
let isLoadingExample = false;
let galaxyCatalog = [];
let galaxyDataCache = {};
let isAutoFitted = false;
let lastCdmHalo = null;
let massModelChartData = null;
let massModelManuallyModified = false;

function getGalacticRadius() {
    var val = parseFloat(galacticRadiusSlider.value);
    return (val && val > 0) ? val : null;
}

// Debounce timer for API calls
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
let autoVortexStrength = null;
let isAutoThroughput = true;
const distanceValue = document.getElementById('distance-value');
const massValue = document.getElementById('mass-value');
const velocityValue = document.getElementById('velocity-value');
const accelValue = document.getElementById('accel-value');
const velocityControl = document.getElementById('velocity-control');
const inferenceResult = document.getElementById('inference-result');
const inferredMassValue = document.getElementById('inferred-mass-value');
const observedLegend = document.getElementById('observed-legend');
const massLabel = document.getElementById('mass-label');

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

function updateResetToPhotometricButton() {
    var btn = document.getElementById('reset-to-photometric-btn');
    if (!btn) return;
    btn.style.display = (massModelManuallyModified && currentExample && currentExample.id) ? '' : 'none';
}

function resetToPhotometric() {
    var pm = massModelChartData && massModelChartData.photometric_mass_model;
    if (!currentExample || !currentExample.id || !pm) return;
    setMassModelSliders(pm);
    massModelManuallyModified = false;
    updateResetToPhotometricButton();
    fetchMassModelChart(currentExample.id).then(function() {
        applyMassModelChartFromApi(massModelChartData);
    }).catch(function() {});
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

// =====================================================================
// MASS MODEL SLIDER LISTENERS
// =====================================================================

function debouncedRefreshChart() {
    clearTimeout(manualGfdTimer);
    manualGfdTimer = setTimeout(function() {
        var example = pinnedGalaxyExample || currentExample;
        var galaxyId = example && example.id ? example.id : null;
        if (!galaxyId) return;
        if (massModelManuallyModified) {
            fetchMassModelChartManual(galaxyId);
        } else {
            fetchMassModelChart(galaxyId).then(function() {
                applyMassModelChartFromApi(massModelChartData);
            }).catch(function() {});
        }
    }, MANUAL_GFD_DEBOUNCE_MS);
}

[bulgeMassSlider, bulgeScaleSlider, diskMassSlider, diskScaleSlider, gasMassSlider, gasScaleSlider].forEach(slider => {
    slider.addEventListener('input', () => {
        if (!isLoadingExample && currentExample && currentExample.id) {
            massModelManuallyModified = true;
            updateResetToPhotometricButton();
            debouncedRefreshChart();
        }
        updateMassModelDisplays();
    });
});

galacticRadiusSlider.addEventListener('input', () => {
    galacticRadiusValue.textContent = galacticRadiusSlider.value + ' kpc';
});

lensSlider.addEventListener('input', () => {
    var pct = parseFloat(lensSlider.value);
    lensValue.textContent = '+/- ' + pct.toFixed(1) + '%';
});

vortexStrengthSlider.addEventListener('input', () => {
    vortexStrengthValue.textContent = parseFloat(vortexStrengthSlider.value).toFixed(2);
    isAutoThroughput = false;
    vortexAutoBtn.classList.remove('active');
});

vortexAutoBtn.addEventListener('click', () => {
    isAutoThroughput = true;
    vortexAutoBtn.classList.add('active');
    if (autoVortexStrength !== null) {
        vortexStrengthSlider.value = autoVortexStrength;
        vortexStrengthValue.textContent = 'auto';
    } else {
        vortexStrengthValue.textContent = 'auto';
    }
});

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
        var obsIdx = -1;
        if (chart.canvas.id === 'gravityChart') obsIdx = 3;
        else return;
        const dataset = chart.data.datasets[obsIdx];
        if (!dataset || dataset.hidden || !dataset.data || dataset.data.length === 0) return;
        const meta = chart.getDatasetMeta(obsIdx);
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
            var err = point.err;
            if (err == null && chart.data.datasets[obsIdx].errorBars && chart.data.datasets[obsIdx].errorBars[index] != null) {
                err = chart.data.datasets[obsIdx].errorBars[index];
            }
            if (err) {
                const yTop = yScale.getPixelForValue(point.y + err);
                const yBottom = yScale.getPixelForValue(point.y - err);
                ctx.beginPath(); ctx.moveTo(x, yTop); ctx.lineTo(x, yBottom); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(x - 4, yTop); ctx.lineTo(x + 4, yTop); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(x - 4, yBottom); ctx.lineTo(x + 4, yBottom); ctx.stroke();
            }
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

var fieldOriginBoundaryPlugin = {
    id: 'fieldOriginBoundary',
    afterDraw: function(chartInstance) {
        return;  // R_t line hidden for now
    }
};
Chart.register(fieldOriginBoundaryPlugin);

// =====================================================================
// FIELD HORIZON PLUGIN
// =====================================================================

var fieldHorizonPlugin = {
    id: 'fieldHorizon',
    afterDraw: function(chartInstance) {
        return;  // R_env line hidden for now
    }
};
Chart.register(fieldHorizonPlugin);

// =====================================================================
// SPARC R_HI PLUGIN
// =====================================================================

var sparcRhiPlugin = {
    id: 'sparcRhi',
    afterDraw: function(chartInstance) {
        var rHi = massModelChartData ? massModelChartData.sparc_r_hi_kpc : null;
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

var rVisBandPlugin = {
    id: 'rVisBand',
    beforeDraw: function(chartInstance) {
        return;  // Purple baryonic band hidden for now
    }
};
Chart.register(rVisBandPlugin);

// =====================================================================
// TOOLTIP SYSTEM
// =====================================================================

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

    html += '<div style="padding:8px 12px 4px;font-size:13px;font-weight:700;color:#e0e0e0;">';
    html += 'r = ' + r.toFixed(1) + ' kpc';
    html += '</div>';
    html += '<hr style="margin:0 12px;border:none;border-top:1px solid #444;">';

    html += ttSection('Velocity Comparison');
    var errStr = (err > 0) ? ' \u00B1 ' + err : '';

    var gfdV = interpolateCurve(1, r);
    var newtonV = interpolateCurve(0, r);
    var mondV = interpolateCurve(2, r);
    var cdmV = interpolateCurve(7, r);
    var topoV = interpolateCurve(13, r);

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
    if (gfdV !== null)    html += ttRow('<span style="color:#00E5A0;">\u25CF</span> GFD', gfdV.toFixed(1) + ' km/s' + fmtDelta(gfdV), '#00E5A0');
    if (newtonV !== null) html += ttRow('<span style="color:#ef5350;">\u25CF</span> Newton', newtonV.toFixed(1) + ' km/s' + fmtDelta(newtonV), '#ef5350');
    if (mondV !== null)   html += ttRow('<span style="color:#ab47bc;">\u25CF</span> MOND', mondV.toFixed(1) + ' km/s' + fmtDelta(mondV), '#ab47bc');
    if (cdmV !== null && chart.data.datasets[7].data.length > 0) {
        html += ttRow('<span style="color:#ffffff;">\u25CF</span> CDM+NFW', cdmV.toFixed(1) + ' km/s' + fmtDelta(cdmV), '#ffffff');
    }
    if (topoV !== null && chart.data.datasets[13].data.length > 0) {
        html += ttRow('<span style="color:#FF6D00;">\u25CF</span> GFD Topo', topoV.toFixed(1) + ' km/s' + fmtDelta(topoV), '#FF6D00');
    }
    html += '</table>';

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
        html += '<span style="color:#e0e0e0;">GFD - Obs: </span>';
        html += '<span style="font-weight:600;color:#e0e0e0;">' + (delta >= 0 ? '+' : '') + delta.toFixed(1) + ' km/s</span>';
        html += '<br><span style="font-size:11px;color:' + sigColor + ';font-weight:600;">'
            + sigAway.toFixed(1) + '\u03C3, ' + sigLabel + '</span>';
        html += '</div>';
    }

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

    if (tooltip.opacity === 0) {
        el.style.opacity = '0';
        return;
    }

    var dp = tooltip.dataPoints && tooltip.dataPoints[0];
    if (!dp) { el.style.opacity = '0'; return; }

    if (dp.datasetIndex === 4 || dp.datasetIndex === 5) {
        el.style.opacity = '0';
        return;
    }

    var html = '';
    if (dp.datasetIndex === 3) {
        html = buildObservedTooltip(dp);
    } else if (dp.datasetIndex === 7 && lastCdmHalo) {
        html = buildCdmTooltip(dp);
    } else {
        html = buildDefaultTooltip(dp);
    }

    el.innerHTML = html;

    var chartRect = tooltipContext.chart.canvas.getBoundingClientRect();
    var left = chartRect.left + window.scrollX + tooltip.caretX + 14;
    var top = chartRect.top + window.scrollY + tooltip.caretY - 20;

    var elWidth = el.offsetWidth || 280;
    if (left + elWidth > window.innerWidth - 10) {
        left = chartRect.left + window.scrollX + tooltip.caretX - elWidth - 14;
    }
    var elHeight = el.offsetHeight || 200;
    if (top + elHeight > window.innerHeight + window.scrollY - 10) {
        top = window.innerHeight + window.scrollY - elHeight - 10;
    }
    if (top < window.scrollY + 5) top = window.scrollY + 5;

    el.style.left = left + 'px';
    el.style.top = top + 'px';
    el.style.opacity = '1';
}

// =====================================================================
// CROSSHAIR LINE PLUGIN
// =====================================================================

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

// =====================================================================
// CHART CREATION
// =====================================================================

var CHART_Y_AXIS_LEFT_WIDTH = 80;

const ctx = document.getElementById('gravityChart').getContext('2d');
const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Newtonian',
                data: [],
                borderColor: '#ff6b6b',
                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                borderWidth: 2.5,
                tension: 0.4,
                pointRadius: 0,
                hidden: true
            },
            {
                label: 'GFD (Photometric)',
                data: [],
                borderColor: '#00E5A0',
                backgroundColor: 'rgba(0, 229, 160, 0.1)',
                borderDash: [8, 4],
                borderWidth: 2.5,
                tension: 0.4,
                pointRadius: 0
            },
            {
                label: 'MOND',
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
            // Dataset 4: GFD envelope upper edge (hidden from legend)
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
            // Dataset 5: GFD envelope lower edge (hidden from legend)
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
            // Dataset 6: Inference markers (reserved)
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
                label: 'CDM',
                data: [],
                borderColor: '#ffffff',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                borderWidth: 2,
                borderDash: [8, 4],
                tension: 0.4,
                pointRadius: 0,
                hidden: true
            },
            // Dataset 8: reserved
            {
                label: 'GFD\u03C6',
                data: [],
                borderColor: '#76FF03',
                backgroundColor: 'rgba(118, 255, 3, 0.08)',
                borderWidth: 2.5,
                tension: 0.4,
                pointRadius: 0
            },
            // Dataset 9: reserved
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
            // Dataset 10: reserved
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
            // Dataset 11: reserved (observations chart)
            {
                label: 'GFD (SST+Poisson photometric)',
                data: [],
                borderColor: '#aa44ff',
                backgroundColor: 'transparent',
                borderWidth: 2.5,
                cubicInterpolationMode: 'monotone',
                tension: 0.4,
                pointRadius: 0
            },
            // Dataset 12: reserved
            {
                label: 'GFD velocity decode (no mass input)',
                data: [],
                borderColor: '#1E88E5',
                backgroundColor: 'transparent',
                borderWidth: 2.2,
                cubicInterpolationMode: 'monotone',
                tension: 0.4,
                pointRadius: 0
            },
            // Dataset 13: GFD Topological (signed Burgers vortex)
            {
                label: 'GFD (Topological)',
                data: [],
                borderColor: '#FF6D00',
                backgroundColor: 'rgba(255, 109, 0, 0.08)',
                borderWidth: 2.5,
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
            legend: { display: false },
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

Chart.register(crosshairLinePlugin);

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
canvas.addEventListener('dblclick', resetChartZoom);

// =====================================================================
// API COMMUNICATION
// =====================================================================

async function fetchGalaxyCatalog() {
    const resp = await fetch('/api/rotation/galaxy-catalog');
    if (!resp.ok) throw new Error('API error: ' + resp.status);
    return resp.json();
}

async function fetchGalaxyById(galaxyId) {
    if (galaxyDataCache[galaxyId]) return galaxyDataCache[galaxyId];
    const resp = await fetch('/api/rotation/galaxies/' + encodeURIComponent(galaxyId));
    if (!resp.ok) throw new Error('Galaxy not found: ' + galaxyId);
    const data = await resp.json();
    galaxyDataCache[galaxyId] = data;
    return data;
}

function showChartLoading() {
    var overlay = document.getElementById('chart-loading-overlay');
    if (overlay) overlay.style.display = 'flex';
}

function hideChartLoading() {
    var overlay = document.getElementById('chart-loading-overlay');
    if (overlay) overlay.style.display = 'none';
}

/**
 * Fetch chart data for Mass model tab. POST /api/rotation/charts/mass_model.
 * Uses the galaxy's photometric mass model (no override).
 */
async function fetchMassModelChart(galaxyId) {
    if (!galaxyId) return null;
    try {
        var maxR = parseFloat(distanceSlider.value) || 50;
        var resp = await fetch('/api/rotation/charts/mass_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                galaxy_id: galaxyId,
                num_points: 500,
                max_radius: maxR,
                accel_ratio: parseFloat(accelSlider.value) || 1.0
            })
        });
        if (!resp.ok) return null;
        massModelChartData = await resp.json();
        return massModelChartData;
    } catch (e) {
        massModelChartData = null;
        return null;
    }
}

/**
 * Fetch chart data with slider-overridden mass model.
 * POST /api/rotation/charts/mass_model with mass_model_override.
 */
async function fetchMassModelChartManual(galaxyId) {
    if (!galaxyId) return;
    var mm = getMassModelFromSliders();
    var maxR = parseFloat(distanceSlider.value) || 50;
    if (pinnedObservations && pinnedObservations.length > 0) {
        var maxObsR = Math.max.apply(null, pinnedObservations.map(function(o) { return o.r; }));
        maxR = Math.max(maxR, maxObsR * 1.15);
    }
    try {
        var resp = await fetch('/api/rotation/charts/mass_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                galaxy_id: galaxyId,
                num_points: 500,
                max_radius: maxR,
                accel_ratio: parseFloat(accelSlider.value) || 1.0,
                mass_model_override: mm
            })
        });
        if (!resp.ok) return;
        massModelChartData = await resp.json();
        applyMassModelChartFromApi(massModelChartData);
    } catch (e) {
        console.error('Manual mass model chart error:', e);
    }
}

// =====================================================================
// CHART RENDERING
// =====================================================================

/**
 * Apply mass model chart from API response. Sets theory curves,
 * observations, axis bounds, and updates panels.
 */
function applyMassModelChartFromApi(data) {
    if (typeof chart === 'undefined' || !chart || !chart.data || !chart.data.datasets) return;
    if (!data || !data.chart) {
        chart.data.datasets[0].data = [];
        chart.data.datasets[1].data = [];
        chart.data.datasets[2].data = [];
        chart.data.datasets[3].data = [];
        chart.data.datasets[7].data = [];
        chart.data.datasets[13].data = [];
        chart.update('none');
        return;
    }
    var c = data.chart;
    var radii = c.radii || [];
    var obsMass = (data.observations && data.observations.length) ? data.observations : (pinnedObservations || []);
    var bounds = data.axis_bounds || {};
    var xMax = bounds.x_max || 50;
    var yMax = bounds.y_max || undefined;
    var newtonianData = [];
    var photoData = [];
    var mondData = [];
    var cdmData = [];
    var topoData = [];
    for (var i = 0; i < radii.length; i++) {
        if (radii[i] > xMax) break;
        var r = radii[i];
        if (c.newtonian_photometric && c.newtonian_photometric[i] !== undefined) newtonianData.push({ x: r, y: c.newtonian_photometric[i] });
        if (c.gfd_photometric && c.gfd_photometric[i] !== undefined) photoData.push({ x: r, y: c.gfd_photometric[i] });
        if (c.mond_photometric && c.mond_photometric[i] !== undefined) mondData.push({ x: r, y: c.mond_photometric[i] });
        if (c.cdm_photometric && c.cdm_photometric.length === radii.length && c.cdm_photometric[i] !== undefined) cdmData.push({ x: r, y: c.cdm_photometric[i] });
        if (c.gfd_topological && c.gfd_topological[i] !== undefined) topoData.push({ x: r, y: c.gfd_topological[i] });
    }
    var observedData = obsMass.map(function(o) { return { x: o.r, y: o.v }; });

    var example = currentExample || pinnedGalaxyExample;
    var galaxyLabel = example ? example.name.replace(/\s*\(.*\)/, '') : 'Galaxy';
    chart.options.plugins.title.text = galaxyLabel + ' Rotation Curve';

    chart.data.datasets[0].data = newtonianData;
    chart.data.datasets[0].hidden = !isChipEnabled('newtonian');
    chart.data.datasets[1].data = photoData;
    chart.data.datasets[1].label = massModelManuallyModified ? 'GFD (Manual mass params)' : 'GFD (Photometric)';
    chart.data.datasets[2].data = mondData;
    chart.data.datasets[2].hidden = !isChipEnabled('mond');
    chart.data.datasets[3].data = observedData;
    chart.data.datasets[7].data = cdmData;
    chart.data.datasets[7].hidden = !isChipEnabled('cdm');
    chart.data.datasets[13].data = topoData;
    chart.data.datasets[13].hidden = !isChipEnabled('gfd_topological');

    // Clear unused datasets
    chart.data.datasets[4].data = [];
    chart.data.datasets[5].data = [];
    chart.data.datasets[6].data = [];
    chart.data.datasets[8].data = [];
    chart.data.datasets[8].hidden = true;
    chart.data.datasets[9].data = [];
    chart.data.datasets[9].hidden = true;
    chart.data.datasets[10].data = [];
    chart.data.datasets[10].hidden = true;
    chart.data.datasets[11].data = [];
    chart.data.datasets[11].hidden = true;
    chart.data.datasets[12].data = [];
    chart.data.datasets[12].hidden = true;

    chart.options.scales.x.min = bounds.x_min || 0;
    chart.options.scales.x.max = xMax;
    chart.options.scales.y.min = bounds.y_min || 0;
    if (yMax) chart.options.scales.y.max = yMax;

    // Update CDM halo
    lastCdmHalo = data.cdm_halo || null;
    updateCdmHaloPanel();

    // Update GFD chip label
    var gfdChipLabel = document.getElementById('gfd-chip-label');
    if (gfdChipLabel) gfdChipLabel.textContent = chart.data.datasets[1].label;

    // Update panels
    updatePhotometricPanel();
    updateFieldGeometryMetrics(data.field_geometry);
    updateMetricsPanel();

    chart.update('none');
}

function updatePhotometricPanel() {
    var pm = massModelChartData && massModelChartData.photometric_mass_model;
    if (!pm) return;

    var gasMass = pm.gas ? pm.gas.M : 0;
    var gasScale = pm.gas ? pm.gas.Rd : 0;
    var diskMass = pm.disk ? pm.disk.M : 0;
    var diskScale = pm.disk ? pm.disk.Rd : 0;
    var bulgeMass = pm.bulge ? pm.bulge.M : 0;
    var bulgeScale = pm.bulge ? pm.bulge.a : 0;
    var totalMass = gasMass + diskMass + bulgeMass;

    document.getElementById('gas-mass-value').textContent = gasMass.toExponential(1) + ' M_sun';
    document.getElementById('gas-scale-value').textContent = gasScale.toFixed(1) + ' kpc';
    document.getElementById('disk-mass-value').textContent = diskMass.toExponential(1) + ' M_sun';
    document.getElementById('disk-scale-value').textContent = diskScale.toFixed(1) + ' kpc';
    document.getElementById('bulge-mass-value').textContent = bulgeMass.toExponential(1) + ' M_sun';
    document.getElementById('bulge-scale-value').textContent = bulgeScale.toFixed(1) + ' kpc';
    document.getElementById('mass-model-total-value').textContent = totalMass.toExponential(1) + ' M_sun';
}

function updateFieldGeometryMetrics(fg) {
    var originEl = document.getElementById('metric-field-origin');
    var horizonEl = document.getElementById('metric-field-horizon');
    if (!originEl || !horizonEl) return;
    if (fg && fg.throat_radius_kpc != null) {
        originEl.textContent = fg.throat_radius_kpc.toFixed(1) + ' kpc';
        horizonEl.textContent = (fg.envelope_radius_kpc || 0).toFixed(1) + ' kpc';
    } else {
        originEl.textContent = '--';
        horizonEl.textContent = '--';
    }
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
// CURVE INTERPOLATION
// =====================================================================

function interpolateCurve(datasetIndex, radius) {
    var data = chart.data.datasets[datasetIndex].data;
    if (!data || data.length === 0) return null;
    if (radius <= data[0].x) return data[0].y;
    if (radius >= data[data.length - 1].x) return data[data.length - 1].y;
    for (var i = 1; i < data.length; i++) {
        if (data[i].x >= radius) {
            var x0 = data[i-1].x, y0 = data[i-1].y;
            var x1 = data[i].x,   y1 = data[i].y;
            if (x1 === x0) return y0;
            var t = (radius - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    return null;
}

function interpolateGfdAt(r, gfdData) {
    if (!gfdData || gfdData.length === 0) return null;
    if (r <= gfdData[0].x) return gfdData[0].y;
    if (r >= gfdData[gfdData.length - 1].x) return gfdData[gfdData.length - 1].y;
    for (var i = 1; i < gfdData.length; i++) {
        if (gfdData[i].x >= r) {
            var x0 = gfdData[i - 1].x, y0 = gfdData[i - 1].y;
            var x1 = gfdData[i].x,     y1 = gfdData[i].y;
            if (x1 === x0) return y0;
            var t = (r - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    return null;
}

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

function fmtSci(val) {
    if (!val || val === 0) return '--';
    var exp = Math.floor(Math.log10(Math.abs(val)));
    var coeff = val / Math.pow(10, exp);
    return coeff.toFixed(2) + 'e' + exp;
}

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
 * Update the right metrics panel with client-computed fit quality
 * for every theory curve vs observations.
 */
function updateMetricsPanel() {
    var obs = pinnedObservations || (currentExample ? currentExample.observations : null);

    var theories = [
        { key: 'gfd',             label: 'GFD',              color: '#00E5A0', dsIndex: 1,  params: 0 },
        { key: 'gfd_topological', label: 'GFD (Topological)',color: '#FF6D00', dsIndex: 13, params: 2 },
        { key: 'newtonian',       label: 'Newtonian',        color: '#ff6b6b', dsIndex: 0,  params: 0 },
        { key: 'mond',            label: 'MOND',             color: '#9966ff', dsIndex: 2,  params: 0 },
        { key: 'cdm',             label: 'CDM+NFW',          color: '#ffffff', dsIndex: 7,  params: 1 },
    ];

    var tbody = document.getElementById('fit-compare-tbody');
    if (!tbody) return;

    if (!obs || obs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="color:#808080;text-align:center;padding:8px 0;">No observations</td></tr>';
    } else {
        var bestRms = Infinity;
        var results = [];
        for (var i = 0; i < theories.length; i++) {
            var t = theories[i];
            var ds = chart.data.datasets[t.dsIndex];
            if (!ds || !ds.data || ds.data.length === 0) {
                results.push(null);
                continue;
            }
            var m = computeFitMetrics(obs, ds.data);
            results.push(m);
            if (m && m.rms < bestRms) bestRms = m.rms;
        }

        var html = '';
        for (var j = 0; j < theories.length; j++) {
            var th = theories[j];
            var m = results[j];
            var isBest = m && Math.abs(m.rms - bestRms) < 0.01;
            html += '<tr class="' + (isBest ? 'fc-best' : '') + '">';
            html += '<td><span class="fc-swatch" style="background:' + th.color + '"></span>' + th.label;
            if (th.params > 0) html += ' <span style="color:#808080;font-weight:400;font-size:0.85em;">(' + th.params + 'p)</span>';
            html += '</td>';
            if (m) {
                var rmsCls = m.rms < 10 ? 'fc-good' : m.rms < 25 ? 'fc-warn' : 'fc-bad';
                var chi2Cls = m.chi2r < 1.5 ? 'fc-good' : m.chi2r < 3 ? 'fc-warn' : 'fc-bad';
                var s1Cls = (m.within1s / m.nObs >= 0.68) ? 'fc-good' : 'fc-warn';
                var s2Cls = (m.within2s / m.nObs >= 0.95) ? 'fc-good' : 'fc-warn';
                html += '<td class="' + rmsCls + '">' + m.rms.toFixed(1) + '</td>';
                html += '<td class="' + chi2Cls + '">' + m.chi2r.toFixed(1) + '</td>';
                html += '<td class="' + s1Cls + '">' + m.within1s + '/' + m.nObs + '</td>';
                html += '<td class="' + s2Cls + '">' + m.within2s + '/' + m.nObs + '</td>';
            } else {
                html += '<td>--</td><td>--</td><td>--</td><td>--</td>';
            }
            html += '</tr>';
        }
        tbody.innerHTML = html;
    }

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

    var mm = getMassModelFromSliders();
    var totalM = mm.bulge.M + mm.disk.M + mm.gas.M;
    var gasFrac = totalM > 0 ? (mm.gas.M / totalM * 100) : 0;

    document.getElementById('metric-total-mass').textContent = fmtSci(totalM) + ' M_sun';
    document.getElementById('metric-gas-frac').textContent = gasFrac.toFixed(1) + '%';

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

    var residualPanels = [
        { datasetIdx: 1,  tbodyId: 'metrics-residuals-photometric-tbody' },
        { datasetIdx: 13, tbodyId: 'metrics-residuals-topological-tbody' }
    ];
    for (var pi = 0; pi < residualPanels.length; pi++) {
        var panel = residualPanels[pi];
        var metrics = computeFitMetrics(obs, chart.data.datasets[panel.datasetIdx].data);
        var tbody = document.getElementById(panel.tbodyId);
        if (tbody && metrics && metrics.residuals.length > 0) {
            var html = '';
            for (var ri = 0; ri < metrics.residuals.length; ri++) {
                var res = metrics.residuals[ri];
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
// GALAXY BROWSER
// =====================================================================

var _galaxyBrowserSelectedId = null;

function renderGalaxyBrowser(filter) {
    var listEl = document.getElementById('galaxy-browser-list');
    var emptyEl = document.getElementById('galaxy-browser-empty');
    var countEl = document.getElementById('galaxy-browser-count');
    if (!listEl) return;

    listEl.innerHTML = '';
    var query = (filter || '').toLowerCase();
    var matched = [];
    for (var i = 0; i < galaxyCatalog.length; i++) {
        var g = galaxyCatalog[i];
        if (query && g.name.toLowerCase().indexOf(query) === -1 &&
            g.id.toLowerCase().indexOf(query) === -1) continue;
        matched.push(g);
    }

    if (countEl) {
        countEl.textContent = (query ? matched.length + ' / ' : '') + galaxyCatalog.length;
    }

    if (matched.length === 0) {
        emptyEl.style.display = 'block';
        return;
    }
    emptyEl.style.display = 'none';

    for (var j = 0; j < matched.length; j++) {
        var entry = matched[j];
        var li = document.createElement('li');
        li.className = 'galaxy-browser-item' + (entry.id === _galaxyBrowserSelectedId ? ' selected' : '');
        li.dataset.id = entry.id;
        li.setAttribute('role', 'option');
        li.setAttribute('tabindex', '0');

        var idSpan = document.createElement('span');
        idSpan.className = 'galaxy-browser-item-id';
        idSpan.textContent = entry.id;
        li.appendChild(idSpan);

        var nameSpan = document.createElement('span');
        nameSpan.className = 'galaxy-browser-item-name';
        nameSpan.textContent = entry.name;
        li.appendChild(nameSpan);

        li.addEventListener('click', (function (gid) {
            return function () { selectGalaxyById(gid); };
        })(entry.id));

        li.addEventListener('keydown', (function (gid) {
            return function (e) {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectGalaxyById(gid); }
            };
        })(entry.id));

        listEl.appendChild(li);
    }
}

function highlightGalaxyInBrowser(galaxyId) {
    _galaxyBrowserSelectedId = galaxyId;
    var items = document.querySelectorAll('.galaxy-browser-item');
    for (var i = 0; i < items.length; i++) {
        var isMatch = items[i].dataset.id === galaxyId;
        items[i].classList.toggle('selected', isMatch);
        items[i].setAttribute('aria-selected', isMatch ? 'true' : 'false');
    }
    var sel = document.querySelector('.galaxy-browser-item.selected');
    if (sel) sel.scrollIntoView({ block: 'nearest' });
}

function initGalaxyBrowserSearch() {
    var input = document.getElementById('galaxy-search-input');
    if (!input) return;
    input.addEventListener('input', function () {
        renderGalaxyBrowser(input.value.trim());
    });
}

async function selectGalaxyById(galaxyId) {
    if (!galaxyId || isLoadingExample) return;
    isLoadingExample = true;
    highlightGalaxyInBrowser(galaxyId);

    var example;
    try {
        example = await fetchGalaxyById(galaxyId);
    } catch (err) {
        console.error('Failed to load galaxy:', galaxyId, err);
        isLoadingExample = false;
        return;
    }

    isAutoFitted = false;
    velocityControl.style.display = 'none';
    inferenceResult.classList.remove('visible');
    var multiDiv = document.getElementById('multi-inference-result');
    if (multiDiv) multiDiv.style.display = 'none';
    setMassSliderEditable(true);
    unlockAllChips();
    hideAutoMapDiagnostics();

    var galaxyLabel = example.name.replace(/\s*\(.*\)/, '');
    chart.options.plugins.title.text = galaxyLabel + ' Rotation Curve';

    chart.data.datasets[3].data = [];
    chart.data.datasets[3].hidden = true;
    chart.data.datasets[6].data = [];

    currentExample = example;

    var pinObs = example.observations;
    pinnedObservations = pinObs || null;
    pinnedGalaxyLabel = galaxyLabel;
    pinnedGalaxyExample = example;

    accelSlider.value = example.accel;
    distanceSlider.value = example.distance;
    anchorRadiusInput.value = example.distance;
    massSlider.value = example.mass || 11;

    navigateTo('charts', 'chart');

    var obsData = example.observations;
    if (obsData && obsData.length > 0) {
        var obsRadii = obsData.map(function (obs) { return obs.r; });
        var minR = Math.min.apply(null, obsRadii);
        var maxR = Math.max.apply(null, obsRadii);
        var range = maxR - minR;
        var padding = Math.max(5, range * 0.2);
        chart.options.plugins.zoom.limits.x.min = Math.max(0, minR - padding);
        chart.options.plugins.zoom.limits.x.max = Math.min(100, maxR + padding);
    } else {
        chart.options.plugins.zoom.limits.x.min = 0;
        chart.options.plugins.zoom.limits.x.max = 100;
    }

    if (example.mass_model) {
        setMassModelSliders(example.mass_model);
    }
    updateMassModelDisplays();

    if (example.galactic_radius) {
        galacticRadiusSlider.value = example.galactic_radius;
        galacticRadiusValue.textContent = example.galactic_radius + ' kpc';
    }

    isAutoThroughput = true;
    vortexAutoBtn.classList.add('active');
    vortexStrengthValue.textContent = 'auto';

    massModelManuallyModified = false;
    updateResetToPhotometricButton();
    updateDisplays();

    var obsEnabled = isChipEnabled('observed');
    if (pinnedObservations && pinnedObservations.length > 0) {
        chart.data.datasets[3].data = pinnedObservations.map(function (obs) {
            return { x: obs.r, y: obs.v, err: obs.err || 0 };
        });
        chart.data.datasets[3].hidden = !obsEnabled;
        chart.data.datasets[3].errorBars = pinnedObservations.map(function (obs) {
            return obs.err || 0;
        });
    }

    if (!example.id) {
        chart.update('none');
        chart.resetZoom();
        hideResetButton();
        isLoadingExample = false;
    }
}

// =====================================================================
// ANALYSIS VIEW STATE AND NAVIGATION PIPELINE
// =====================================================================

var analysisViewState = {
    rightPanelTab: 'charts',
    chartsSubmenuTab: 'chart'
};

/**
 * Navigate to a view. Single pipeline for tab switches.
 * @param {string} rightPanelTab - 'charts' | 'chart-data'
 * @param {string} [chartsSubmenuTab] - 'chart' | 'obs-chart' | 'vortex'
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
    var vortexFace = document.getElementById('vortex-face');
    var fieldFace = document.getElementById('field-analysis-face');

    // Left panel
    if (massModelPanel && obsMassPanel) {
        massModelPanel.style.display = '';
        obsMassPanel.style.display = 'none';
    }

    // Right panel: which tab content is visible
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

    // Charts submenu
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

    // Charts submenu content
    if (csTab === 'chart') {
        if (theoryToggles) theoryToggles.style.display = '';
        if (chartContainer) chartContainer.style.display = '';
        applyMassModelChartState();
    } else if (csTab === 'obs-chart') {
        if (chartContainer) chartContainer.style.display = '';
        showPlaceholder('Observations chart: coming soon');
    } else if (csTab === 'vortex') {
        if (vortexFace) {
            vortexFace.style.display = '';
            vortexFace.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#606060;font-size:1.1em;">Vortex analysis: coming soon</div>';
        }
        if (rightPanel) rightPanel.classList.add('obs-data-active');
    }
}

function showPlaceholder(message) {
    chart.data.datasets.forEach(function(ds) { ds.data = []; });
    chart.options.plugins.title.text = message;
    chart.update('none');
}

/**
 * Chart state for Mass Model view. Fetches from /api/rotation/charts/mass_model.
 */
async function applyMassModelChartState() {
    isAutoFitted = false;
    var vpsPanel = document.getElementById('vps-panel');
    if (vpsPanel) vpsPanel.style.display = 'none';
    updateResetToPhotometricButton();
    var gfdVelChip = document.querySelector('.theory-chip[data-series="gfd_velocity"]');
    if (gfdVelChip) gfdVelChip.style.display = 'none';
    var gfdDecodeChip = document.querySelector('.theory-chip[data-series="gfd_sst_velocity_decode"]');
    if (gfdDecodeChip) gfdDecodeChip.style.display = 'none';
    blankRightPaneMetrics();
    var example = pinnedGalaxyExample || currentExample;
    var galaxyId = example && example.id ? example.id : null;
    if (galaxyId) {
        showChartLoading();
        await fetchMassModelChart(galaxyId);
        hideChartLoading();
    }
    applyMassModelChartFromApi(massModelChartData);
    isLoadingExample = false;
    massModelManuallyModified = false;
    hideResetButton();
    updateResetToPhotometricButton();
}

// =====================================================================
// RIGHT PANEL TABS
// =====================================================================

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
 * Charts submenu (Mass model | Observations | Vortex).
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

// =====================================================================
// CHART DATA TAB
// =====================================================================

function tableToCSV(headers, rows) {
    var lines = [headers.join(',')];
    rows.forEach(function(row) {
        lines.push(row.map(function(cell) { return String(cell); }).join(','));
    });
    return lines.join('\n');
}

function chartDataToJSON() {
    var out = { metadata: {}, mass_model: null, observational_data: [], series: {} };
    var example = currentExample || pinnedGalaxyExample;
    var galaxyName = example ? example.name.replace(/\s*\(.*\)/, '') : 'Galaxy';
    out.metadata.galaxy = galaxyName;
    out.metadata.R_HI_kpc = (massModelChartData && massModelChartData.sparc_r_hi_kpc != null) ? massModelChartData.sparc_r_hi_kpc : null;
    out.metadata.mass_model_source = massModelManuallyModified ? 'Manual' : 'Photometric';
    var pm = massModelChartData && massModelChartData.photometric_mass_model;
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
    var seriesIndices = [0, 1, 2, 3, 7];
    var seriesKeys = ['newtonian', 'gfd_photometric', 'mond', 'observed', 'cdm'];
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
        navigator.clipboard.writeText(csv).then(function() {}).catch(function() {});
    }
}

function downloadCSV(csv, filename) {
    var blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename || 'chart-data.csv';
    a.click();
    URL.revokeObjectURL(a.href);
}

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

function toggleChartDataSection(headerEl) {
    var section = headerEl.closest('.chart-data-section');
    if (section) section.classList.toggle('chart-data-section-collapsed');
}

function renderChartDataTab() {
    var container = document.getElementById('chart-data-content');
    if (!container) return;

    var example = currentExample || pinnedGalaxyExample;
    var galaxyName = example ? example.name.replace(/\s*\(.*\)/, '') : 'Galaxy';
    var pm = massModelChartData && massModelChartData.photometric_mass_model;
    var hasObs = pinnedObservations && pinnedObservations.length > 0;

    if (!example && !massModelChartData && (typeof chart === 'undefined' || !chart)) {
        container.innerHTML = '<p class="chart-data-empty">Load a galaxy to see chart data.</p>';
        return;
    }

    var html = '';

    html += '<div class="chart-data-toolbar">';
    html += '<button type="button" class="chart-data-export-btn" onclick="downloadChartDataJSON()">Download all as JSON</button>';
    html += '</div>';

    html += '<div class="chart-data-section">';
    html += '<div class="chart-data-section-header" onclick="toggleChartDataSection(this)"><span class="chart-data-header-title"><span class="chart-data-chevron">&#9660;</span> Metadata</span></div>';
    html += '<div class="chart-data-section-body">';
    html += '<table class="obs-data-table chart-data-kv"><tbody>';
    html += '<tr><td>Galaxy</td><td>' + (galaxyName || '--') + '</td></tr>';
    var rHi = (massModelChartData && massModelChartData.sparc_r_hi_kpc != null) ? Number(massModelChartData.sparc_r_hi_kpc).toFixed(2) + ' kpc' : '--';
    html += '<tr><td>R_HI (kpc)</td><td>' + rHi + '</td></tr>';
    var src = massModelManuallyModified ? 'Manual' : 'Photometric';
    html += '<tr><td>Mass model source</td><td>' + src + '</td></tr>';
    if (pm) {
        var totalM = (pm.gas ? pm.gas.M : 0) + (pm.disk ? pm.disk.M : 0) + (pm.bulge ? pm.bulge.M : 0);
        html += '<tr><td>Total baryonic mass (M_sun)</td><td>' + totalM.toExponential(2) + '</td></tr>';
    }
    html += '</tbody></table></div></div>';

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

    var seriesConfig = [
        { idx: 3, id: 'observed', filename: 'observed.csv' },
        { idx: 1, id: 'gfd-photometric', filename: 'gfd_photometric.csv' },
        { idx: 13, id: 'gfd-topological', filename: 'gfd_topological.csv' },
        { idx: 0, id: 'newtonian', filename: 'newtonian.csv' },
        { idx: 2, id: 'mond', filename: 'mond.csv' },
        { idx: 7, id: 'cdm', filename: 'cdm.csv' }
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
                    var eStr = (errVal != null) ? errVal.toFixed(2) : '--';
                    html += '<tr><td>' + rx + '</td><td>' + vy + '</td><td>' + eStr + '</td></tr>';
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
// FONT SIZE CONTROLS
// =====================================================================

var _obsFontSizes = [0.75, 0.85, 1.0, 1.15, 1.3];
var _obsFontIndex = 2;

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

// =====================================================================
// MASS SLIDER EDIT STATE
// =====================================================================

function setMassSliderEditable(editable) {
    const massSliders = [bulgeMassSlider, diskMassSlider, gasMassSlider];

    massSliders.forEach(s => {
        s.disabled = !editable;
    });

    document.querySelectorAll('.mass-component .control-label').forEach((label, idx) => {
        if (idx % 2 === 0) {
            const labelText = label.querySelector('span:first-child');
            const valueText = label.querySelector('.control-value');
            if (!editable) {
                if (labelText) labelText.style.opacity = '0.4';
                if (valueText) valueText.style.opacity = '1';
            } else {
                if (labelText) labelText.style.opacity = '1';
                if (valueText) valueText.style.opacity = '1';
            }
        }
    });

    const scaleSliders = [bulgeScaleSlider, diskScaleSlider, gasScaleSlider];
    scaleSliders.forEach(s => {
        s.disabled = false;
        s.style.opacity = '1';
    });
}

// =====================================================================
// SECONDARY SLIDER LISTENERS
// =====================================================================

distanceSlider.addEventListener('input', () => {
    updateDisplays();
    if (!isLoadingExample && currentExample) {
        debouncedRefreshChart();
    }
});

massSlider.addEventListener('input', () => {
    updateDisplays();
});

velocitySlider.addEventListener('input', () => {
    updateDisplays();
});

accelSlider.addEventListener('input', () => {
    updateDisplays();
    if (!isLoadingExample && currentExample) {
        debouncedRefreshChart();
    }
});

// =====================================================================
// RESET TO PHOTOMETRIC
// =====================================================================

var resetToPhotometricBtn = document.getElementById('reset-to-photometric-btn');
if (resetToPhotometricBtn) {
    resetToPhotometricBtn.addEventListener('click', function() {
        resetToPhotometric();
    });
}

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
    if (isResizingMetrics) {
        isResizingMetrics = false;
        metricsResizeHandle.classList.remove('dragging');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        chart.resize();
    }
});

// =====================================================================
// RIGHT METRICS PANEL RESIZE + COLLAPSE
// =====================================================================

const metricsPanel = document.getElementById('metrics-panel');
const metricsResizeHandle = document.getElementById('metrics-resize-handle');
let isResizingMetrics = false;
let metricsWidthBeforeCollapse = 350;

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
    if (newWidth >= 350 && newWidth <= 600) {
        metricsPanel.style.width = newWidth + 'px';
        metricsWidthBeforeCollapse = newWidth;
    }
});

function toggleMetricsPanel() {
    var panel = metricsPanel;
    if (panel.classList.contains('collapsed')) {
        panel.classList.remove('collapsed');
        panel.style.width = metricsWidthBeforeCollapse + 'px';
    } else {
        metricsWidthBeforeCollapse = parseInt(panel.style.width) || 350;
        panel.classList.add('collapsed');
    }
    setTimeout(function() {
        chart.resize();
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

// =====================================================================
// DIAGNOSTICS PANE HELPERS
// =====================================================================

function hideAutoMapDiagnostics() {
    if (!metricsPaneTop) return;
    metricsPaneTop.style.display = 'none';
    if (metricsPaneDivider) metricsPaneDivider.style.display = 'none';
    metricsPaneTop.style.height = '';
    metricsPaneTop.style.flex = '';
    if (metricsPaneBottom) metricsPaneBottom.style.flex = '';
}

function blankRightPaneMetrics() {
    if (metricsPaneTop) {
        metricsPaneTop.style.display = 'none';
        if (metricsPaneDivider) metricsPaneDivider.style.display = 'none';
    }
}

// =====================================================================
// THEORY TOGGLE BAR
// =====================================================================

function isChipEnabled(seriesKey) {
    var chip = document.querySelector('.theory-chip[data-series="' + seriesKey + '"]');
    if (!chip) return true;
    var cb = chip.querySelector('input[type="checkbox"]');
    return cb ? cb.checked : true;
}

var theoryDatasetMap = {
    'observed':      [3],
    'newtonian':     [0],
    'gfd':           [1],
    'gfd_sigma_old': [8],
    'gfd_spline':    [10],
    'gfd_velocity':  [11],
    'gfd_sst_velocity_decode': [12],
    'gfd_symmetric': [9],
    'gfd_topological': [13],
    'mond':          [2],
    'cdm':           [7]
};

function forceChipOn(seriesKey) {
    var chip = document.querySelector('.theory-chip[data-series="' + seriesKey + '"]');
    if (!chip) return;
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

function unlockAllChips() {
    var chips = document.querySelectorAll('.theory-chip.locked');
    for (var i = 0; i < chips.length; i++) {
        chips[i].classList.remove('locked');
    }
}

function initTheoryToggles() {
    var chips = document.querySelectorAll('.theory-chip');
    for (var i = 0; i < chips.length; i++) {
        var chip = chips[i];
        var checkbox = chip.querySelector('input[type="checkbox"]');

        if (checkbox && checkbox.checked) {
            chip.classList.add('active');
        }

        chip.addEventListener('click', function(e) {
            e.preventDefault();
            var seriesKey = this.getAttribute('data-series');

            var cb = this.querySelector('input[type="checkbox"]');
            cb.checked = !cb.checked;

            var indices = theoryDatasetMap[seriesKey] || [];
            var hidden = !cb.checked;

            if (cb.checked) {
                this.classList.add('active');
            } else {
                this.classList.remove('active');
            }

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

var crosshairReadout = null;

function initCrosshairReadout() {
    var container = document.querySelector('.chart-container');
    if (!container) return;

    crosshairReadout = document.createElement('div');
    crosshairReadout.className = 'crosshair-readout';
    container.appendChild(crosshairReadout);

    var canvasEl = document.getElementById('gravityChart');
    if (!canvasEl) return;

    canvasEl.addEventListener('mousemove', function(e) {
        if (!crosshairReadout) return;

        var xScale = chart.scales.x;
        var yScale = chart.scales.y;
        if (!xScale || !yScale) return;

        var rect = canvasEl.getBoundingClientRect();
        var pixelX = e.clientX - rect.left;

        if (pixelX < xScale.left || pixelX > xScale.right) {
            crosshairReadout.classList.remove('visible');
            return;
        }

        var radius = xScale.getValueForPixel(pixelX);
        if (radius < 0) {
            crosshairReadout.classList.remove('visible');
            return;
        }

        var html = '<div style="margin-bottom:4px;color:#00E5A0;font-weight:600;">r = ' +
                   radius.toFixed(1) + ' kpc</div>';

        var seriesDefs = [
            {key: 'observed',  idx: 3, color: '#FFC107', label: 'Observed'},
            {key: 'newtonian', idx: 0, color: '#ff6b6b', label: 'Newtonian'},
            {key: 'gfd',       idx: 1, color: '#00E5A0', label: 'GFD'},
            {key: 'mond',      idx: 2, color: '#9966ff', label: 'MOND'},
            {key: 'cdm',       idx: 7, color: '#ffffff', label: 'CDM'}
        ];

        var hasValues = false;
        for (var i = 0; i < seriesDefs.length; i++) {
            var def = seriesDefs[i];
            var ds = chart.data.datasets[def.idx];
            if (!ds || ds.hidden || !ds.data || ds.data.length === 0) continue;

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
                if (nearest && nearestDist < 1.0) {
                    var eStr = nearest.err ? ' +/- ' + nearest.err.toFixed(1) : '';
                    html += '<div class="crosshair-readout-row">' +
                            '<span class="crosshair-dot" style="background:' + def.color + '"></span>' +
                            '<span class="crosshair-label">' + def.label + '</span>' +
                            '<span class="crosshair-value">' + nearest.y.toFixed(1) + eStr + ' km/s</span>' +
                            '</div>';
                    hasValues = true;
                }
                continue;
            }

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

        var containerRect = container.getBoundingClientRect();
        var readoutLeft = e.clientX - containerRect.left + 16;
        var readoutTop = e.clientY - containerRect.top - 20;

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

    initRightPanelTabs();
    initObsTabs();
    initObsFontControls();
    initTheoryToggles();
    initCrosshairReadout();
    initGalaxyBrowserSearch();

    try {
        galaxyCatalog = await fetchGalaxyCatalog();
    } catch (err) {
        console.error('Failed to load galaxy catalog:', err);
        galaxyCatalog = [];
    }

    renderGalaxyBrowser();

    if (galaxyCatalog.length > 0) {
        await selectGalaxyById(galaxyCatalog[0].id);
    } else {
        updateDisplays();
    }

    dismissSplash();
}

// =====================================================================
// WINDOW EXPORTS
// =====================================================================

window.selectGalaxyById = selectGalaxyById;
window.toggleMetricsSection = toggleMetricsSection;
window.toggleMetricsPanel = toggleMetricsPanel;
window.toggleChartDataSection = toggleChartDataSection;
window.downloadChartDataJSON = downloadChartDataJSON;
window.downloadCSV = downloadCSV;
window.tableToCSV = tableToCSV;
window.copyTableCSV = copyTableCSV;

init();
