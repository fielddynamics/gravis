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
// STATE
// =====================================================================
let currentMode = 'prediction';
let currentExample = null;
let isLoadingExample = false;
let galaxyCatalog = { prediction: [], inference: [] };

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
            currentExample = null;
            document.getElementById('examples-dropdown').value = '0';
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
                    boxHeight: 3
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
                callbacks: {
                    label: function(context) {
                        let label = context.dataset.label || '';
                        if (label) label += ': ';
                        if (context.parsed.y !== null) {
                            label += context.parsed.y.toFixed(1) + ' km/s';
                        }
                        if (context.datasetIndex === 3 && context.raw && context.raw.err) {
                            label += ' +/- ' + context.raw.err + ' km/s';
                        }
                        return label;
                    }
                }
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

async function fetchRotationCurve(maxRadius, accelRatio, massModel) {
    const resp = await fetch('/api/rotation-curve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            max_radius: maxRadius,
            num_points: 100,
            accel_ratio: accelRatio,
            mass_model: massModel
        })
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
            const data = await fetchRotationCurve(maxRadius, accelRatio, curveModel);
            renderCurves(data);

            // Show observation point only when an example is loaded
            if (currentExample) {
                chart.data.datasets[3].data = [{x: r_obs, y: v_obs}];
                chart.data.datasets[3].hidden = false;
                observedLegend.style.display = 'flex';
            } else {
                chart.data.datasets[3].data = [];
                chart.data.datasets[3].hidden = true;
                observedLegend.style.display = 'none';
            }
        } catch (err) {
            console.error('Inference API error:', err);
        }
    } else {
        // Prediction mode: use distributed mass model from sliders
        const massModel = getMassModelFromSliders();

        try {
            const data = await fetchRotationCurve(maxRadius, accelRatio, massModel);
            renderCurves(data);

            // Handle observed data
            if (currentExample && currentExample.observations) {
                chart.data.datasets[3].data = currentExample.observations.map(obs => ({
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

    for (let i = 0; i < data.radii.length; i++) {
        newtonianData.push({x: data.radii[i], y: data.newtonian[i]});
        dtgData.push({x: data.radii[i], y: data.dtg[i]});
        mondData.push({x: data.radii[i], y: data.mond[i]});
    }

    chart.data.labels = [];
    chart.data.datasets[0].data = newtonianData;
    chart.data.datasets[1].data = dtgData;
    chart.data.datasets[2].data = mondData;
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

    currentExample = example;

    distanceSlider.value = example.distance;
    accelSlider.value = example.accel;

    if (currentMode === 'prediction') {
        massSlider.value = example.mass || 11;
    } else {
        velocitySlider.value = example.velocity || 200;
    }

    // Set zoom limits based on observational data or distance
    if (example.observations && example.observations.length > 0) {
        const obsRadii = example.observations.map(obs => obs.r);
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
    currentMode = mode;
    currentExample = null;

    document.getElementById('mode-prediction').classList.toggle('active', mode === 'prediction');
    document.getElementById('mode-inference').classList.toggle('active', mode === 'inference');

    if (mode === 'prediction') {
        velocityControl.style.display = 'none';
        inferenceResult.classList.remove('visible');
        document.getElementById('mass-model-section').style.display = '';
        document.getElementById('mass-slider-group').style.display = 'none';
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
        document.getElementById('examples-dropdown').value = '0';
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
            document.getElementById('examples-dropdown').value = '0';
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
        document.getElementById('examples-dropdown').value = '0';
        updateDisplays();
        debouncedUpdateChart();
    }
});

accelSlider.addEventListener('input', () => {
    if (!isLoadingExample && currentExample) {
        currentExample = null;
        document.getElementById('examples-dropdown').value = '0';
    }
    if (!isLoadingExample) {
        updateDisplays();
        debouncedUpdateChart();
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
    if (!currentExample) return;
    const ex = currentExample;

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
        btn.style.display = (currentExample && currentExample.mass_model) ? 'block' : 'none';
    }
}

// Expose navigation function to global scope for onclick handlers
window.navigateToScreen = navigateToScreen;
window.setMode = setMode;
window.loadExample = loadExample;
window.openDataCard = openDataCard;
window.closeDataCard = closeDataCard;

init();
