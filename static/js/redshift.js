/**
 * GRAVIS - Redshift Dynamics Module
 * Frontend JavaScript: TF velocity evolution + Hubble Tension charts.
 *
 * Two modes:
 *   "tf"     - Tully-Fisher velocity evolution: v(z)/v(0) vs redshift
 *   "hubble" - Hubble Tension: H0 measurements whisker chart
 */

// =====================================================================
// HELPERS
// =====================================================================
function wrapText(text, maxChars) {
    if (!text || text.length <= maxChars) return [text];
    var words = text.split(' ');
    var lines = [];
    var line = '';
    for (var i = 0; i < words.length; i++) {
        var test = line ? line + ' ' + words[i] : words[i];
        if (test.length > maxChars && line) {
            lines.push(line);
            line = words[i];
        } else {
            line = test;
        }
    }
    if (line) lines.push(line);
    return lines;
}

var TOOLTIP_WRAP = 48;

function getOrCreateTooltipEl(chart) {
    var el = chart.canvas.parentNode.querySelector('.gravis-tooltip');
    if (!el) {
        el = document.createElement('div');
        el.className = 'gravis-tooltip';
        el.style.cssText = 'position:absolute;z-index:9999;pointer-events:none;'
            + 'background:#0a0a10;border:1px solid #555;border-radius:6px;'
            + 'padding:10px 12px;max-width:350px;font-family:sans-serif;'
            + 'opacity:0;transition:opacity 0.1s ease;';
        chart.canvas.parentNode.style.position = 'relative';
        chart.canvas.parentNode.appendChild(el);
    }
    return el;
}

function externalTooltipHandler(context) {
    var tooltip = context.tooltip;
    var el = getOrCreateTooltipEl(context.chart);

    if (tooltip.opacity === 0) {
        el.style.opacity = '0';
        return;
    }

    var html = '';
    if (tooltip.title && tooltip.title.length) {
        html += '<div style="font-size:13px;font-weight:600;color:#e8e8e8;margin-bottom:6px;">'
            + tooltip.title.join('<br>') + '</div>';
    }
    if (tooltip.body && tooltip.body.length) {
        tooltip.body.forEach(function(b, idx) {
            var color = '#ffe082';
            if (tooltip.labelTextColors && tooltip.labelTextColors[idx]) {
                color = tooltip.labelTextColors[idx];
            }
            var lines = b.lines || [];
            lines.forEach(function(line) {
                if (line === '') {
                    html += '<div style="height:5px;"></div>';
                } else {
                    html += '<div style="font-size:11.5px;color:' + color
                        + ';line-height:1.5;">' + line + '</div>';
                }
            });
        });
    }

    el.innerHTML = html;
    if (!html) {
        el.style.opacity = '0';
        return;
    }
    el.style.opacity = '1';

    var pos = context.chart.canvas.getBoundingClientRect();
    var tipW = el.offsetWidth;
    var tipH = el.offsetHeight;
    var canvasW = pos.width;
    var canvasH = pos.height;
    var left = tooltip.caretX + 12;
    var top = tooltip.caretY - tipH / 2;

    if (left + tipW > canvasW) left = tooltip.caretX - tipW - 12;
    if (left < 0) left = 4;
    if (top < 0) top = 4;
    if (top + tipH > canvasH) top = canvasH - tipH - 4;

    el.style.left = left + 'px';
    el.style.top = top + 'px';
}

// =====================================================================
// STATE
// =====================================================================
var currentMode = 'tf';
var currentExample = null;
var observedData = [];
var sinsHighlight = null;
var examplesCatalog = [];
var h0Measurements = null;
var tfChart = null;
var hubbleChart = null;

var updateTimer = null;
var DEBOUNCE_MS = 80;

// DOM element references (populated in init)
var redshiftSlider, redshiftValue;
var h0Slider, h0Value;
var lensSlider, lensValue;
var v0Slider, v0Value;
var resultZ, resultHz, resultRatio, resultVz, resultDeltaPct;
var resultDc, resultDl, resultDa, resultTlb;

// =====================================================================
// THEORY CHIP TOGGLES
// =====================================================================

// TF mode visibility
var seriesVisibility = {
    gfd: true,
    lcdm: true,
    observed: true,
    cage: true
};

// Hubble mode visibility
var hubbleVisibility = {
    'h-gfd': true,
    'h-mond': true,
    'h-obs': true,
    'h-slider': true
};

function initTheoryChips() {
    var chips = document.querySelectorAll('.theory-chip');
    chips.forEach(function(chip) {
        var checkbox = chip.querySelector('input[type="checkbox"]');
        var series = chip.getAttribute('data-series');
        if (checkbox.checked) {
            chip.classList.add('active');
        }
        chip.addEventListener('click', function(e) {
            e.preventDefault();
            checkbox.checked = !checkbox.checked;
            chip.classList.toggle('active', checkbox.checked);
            if (series.indexOf('h-') === 0) {
                hubbleVisibility[series] = checkbox.checked;
                if (hubbleChart) hubbleChart.update('none');
            } else {
                seriesVisibility[series] = checkbox.checked;
                updateChartVisibility();
            }
        });
    });
}

function updateChartVisibility() {
    if (currentMode === 'tf' && tfChart) {
        tfChart.data.datasets[0].hidden = !seriesVisibility.gfd;
        tfChart.data.datasets[1].hidden = !seriesVisibility.lcdm;
        tfChart.data.datasets[2].hidden = !seriesVisibility.cage;
        tfChart.data.datasets[3].hidden = !seriesVisibility.cage;
        tfChart.data.datasets[4].hidden = !seriesVisibility.observed;
        tfChart.data.datasets[5].hidden = !seriesVisibility.observed;
        tfChart.update('none');
    }
}

// =====================================================================
// SLIDER DISPLAY UPDATES
// =====================================================================
function updateSliderDisplays() {
    var z = parseFloat(redshiftSlider.value);
    redshiftValue.textContent = 'z = ' + z.toFixed(2);

    var h0 = parseFloat(h0Slider.value);
    h0Value.textContent = h0.toFixed(1) + ' km/s/Mpc';

    var lens = parseFloat(lensSlider.value);
    lensValue.textContent = '+/- ' + lens.toFixed(1) + '%';

    var v0 = parseFloat(v0Slider.value);
    v0Value.textContent = v0 + ' km/s';
}

// =====================================================================
// TF EVOLUTION CHART
// =====================================================================
function createTfChart() {
    var canvas = document.getElementById('tfCanvas');
    var ctx = canvas.getContext('2d');

    tfChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                // 0: GFD prediction curve
                {
                    label: 'GFD: [H(z)/H0]^0.17',
                    data: [],
                    borderColor: '#4da6ff',
                    borderWidth: 2.5,
                    pointRadius: 0,
                    tension: 0.3,
                    order: 2
                },
                // 1: Lambda-CDM (flat line)
                {
                    label: 'Lambda-CDM (no TF evolution)',
                    data: [],
                    borderColor: '#ffffff',
                    borderWidth: 2,
                    borderDash: [8, 4],
                    pointRadius: 0,
                    tension: 0,
                    order: 3
                },
                // 2: GFD uncertainty cage upper
                {
                    label: 'GFD +/- 6.2% cage',
                    data: [],
                    borderColor: 'rgba(77, 166, 255, 0.25)',
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.3,
                    fill: {target: 3, above: 'rgba(77, 166, 255, 0.08)', below: 'rgba(77, 166, 255, 0.08)'},
                    order: 4
                },
                // 3: GFD uncertainty cage lower
                {
                    label: 'GFD cage lower',
                    data: [],
                    borderColor: 'rgba(77, 166, 255, 0.25)',
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.3,
                    order: 4
                },
                // 4: Observed TF data points
                {
                    label: 'Observed TF Data',
                    data: [],
                    borderColor: '#FFC107',
                    backgroundColor: '#FFC107',
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    pointStyle: 'circle',
                    showLine: false,
                    order: 1
                },
                // 5: SINS highlight
                {
                    label: 'SINS z=2 (Cresci+2009)',
                    data: [],
                    borderColor: '#4caf50',
                    backgroundColor: '#4caf50',
                    pointRadius: 8,
                    pointHoverRadius: 10,
                    pointStyle: 'star',
                    showLine: false,
                    order: 0
                },
                // 6: Current redshift marker
                {
                    label: 'Selected z',
                    data: [],
                    borderColor: 'rgba(255, 255, 255, 0.3)',
                    borderWidth: 1,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    showLine: true,
                    order: 5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            interaction: { mode: 'nearest', intersect: false },
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'DFP Tully-Fisher Velocity Evolution',
                    color: '#e0e0e0',
                    font: { size: 16, weight: '500' },
                    padding: 20
                },
                tooltip: {
                    enabled: false,
                    external: externalTooltipHandler,
                    filter: function(item) {
                        var lbl = item.dataset.label;
                        return lbl !== 'GFD +/- 6.2% cage'
                            && lbl !== 'GFD cage lower'
                            && lbl !== 'Selected z';
                    },
                    callbacks: {
                        title: function(items) {
                            if (!items.length) return '';
                            var first = items[0];
                            // SINS highlight
                            if (first.datasetIndex === 5 && sinsHighlight) {
                                return 'SINS z = ' + sinsHighlight.z.toFixed(1) + '  |  Cresci+2009';
                            }
                            // Observed TF data
                            if (first.datasetIndex === 4 && observedData[first.dataIndex]) {
                                var obs = observedData[first.dataIndex];
                                return 'z = ' + obs.z.toFixed(1) + '  |  ' + obs.source;
                            }
                            return 'z = ' + first.parsed.x.toFixed(2);
                        },
                        label: function(ctx) {
                            var ds = ctx.dataset;
                            if (ds.label === 'GFD +/- 6.2% cage'
                                || ds.label === 'GFD cage lower'
                                || ds.label === 'Selected z') return null;
                            var ratio = ctx.parsed.y;
                            var pct = ((ratio - 1.0) * 100).toFixed(1);
                            var sign = pct >= 0 ? '+' : '';

                            // SINS highlight: rich tooltip
                            if (ctx.datasetIndex === 5 && sinsHighlight) {
                                var sRaw = [
                                    'v(z)/v(0) = ' + ratio.toFixed(3) + '  (' + sign + pct + '%)',
                                    'Published: ' + (sinsHighlight.year || 'N/A'),
                                    '',
                                    sinsHighlight.galaxy || '',
                                    sinsHighlight.telescope || '',
                                    sinsHighlight.instrument || '',
                                    sinsHighlight.band || '',
                                    'Resolution: ' + (sinsHighlight.resolution || 'N/A'),
                                    '',
                                    sinsHighlight.significance || '',
                                    sinsHighlight.note || ''
                                ];
                                var sLines = [];
                                for (var i = 0; i < sRaw.length; i++) {
                                    var w = wrapText(sRaw[i], TOOLTIP_WRAP);
                                    for (var j = 0; j < w.length; j++) sLines.push(w[j]);
                                }
                                return sLines;
                            }

                            // Observed TF data: enriched tooltip
                            if (ctx.datasetIndex === 4 && observedData[ctx.dataIndex]) {
                                var obs = observedData[ctx.dataIndex];
                                var raw = [
                                    'v(z)/v(0) = ' + ratio.toFixed(3) + '  (' + sign + pct + '%)  +/- ' + obs.err.toFixed(2),
                                    'Published: ' + (obs.year || 'N/A')
                                ];
                                if (obs.survey) raw.push('Survey: ' + obs.survey);
                                if (obs.telescope) raw.push('Telescope: ' + obs.telescope);
                                if (obs.instrument) raw.push('Instrument: ' + obs.instrument);
                                if (obs.note) {
                                    raw.push('');
                                    raw.push(obs.note);
                                }
                                var lines = [];
                                for (var i = 0; i < raw.length; i++) {
                                    var w = wrapText(raw[i], TOOLTIP_WRAP);
                                    for (var j = 0; j < w.length; j++) lines.push(w[j]);
                                }
                                return lines;
                            }

                            // Theory curves: simple label
                            return ds.label + ': ' + ratio.toFixed(3) + ' (' + sign + pct + '%)';
                        },
                        labelTextColor: function(ctx) {
                            // SINS and observed data get brighter text for readability
                            if (ctx.datasetIndex === 5) return '#a5d6a7';
                            if (ctx.datasetIndex === 4) return '#ffe082';
                            return '#b0b0b0';
                        }
                    }
                },
                zoom: {
                    pan: { enabled: true, mode: 'xy' },
                    zoom: {
                        wheel: { enabled: true },
                        pinch: { enabled: true },
                        mode: 'xy',
                        onZoomComplete: function() {
                            document.getElementById('reset-zoom-btn').style.display = 'block';
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Redshift z', color: '#808080', font: { size: 13 } },
                    min: 0, max: 4.2,
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: { color: '#808080' }
                },
                y: {
                    type: 'linear',
                    title: { display: true, text: 'v(z) / v(0)', color: '#808080', font: { size: 13 } },
                    min: 0.9, max: 1.5,
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: { color: '#808080' }
                }
            }
        }
    });
}

// =====================================================================
// HUBBLE TENSION CHART
// =====================================================================

function drawVLine(ctx, x, top, bottom, color, width, dash) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash || []);
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.restore();
}

function drawVLabel(ctx, text, x, y, color, font, align) {
    ctx.save();
    ctx.fillStyle = color;
    ctx.font = font || '11px sans-serif';
    ctx.textAlign = align || 'center';
    ctx.fillText(text, x, y);
    ctx.restore();
}

function createHubbleChart() {
    var canvas = document.getElementById('hubbleCanvas');
    var ctx = canvas.getContext('2d');

    hubbleChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'H0 Measurement',
                    data: [],
                    backgroundColor: [],
                    borderColor: [],
                    borderWidth: 1,
                    hoverBorderWidth: 2,
                    hoverBorderColor: '#ffffff',
                    barPercentage: 0.5,
                    categoryPercentage: 0.8
                },
                {
                    type: 'scatter',
                    label: 'Error bars',
                    data: [],
                    pointRadius: 0,
                    showLine: false
                }
            ]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 },
            layout: {
                padding: { top: 8, bottom: 8 }
            },
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'Hubble Tension: H0 Measurements',
                    color: '#e0e0e0',
                    font: { size: 16, weight: '500' },
                    padding: { top: 4, bottom: 16 }
                },
                tooltip: {
                    enabled: false,
                    external: externalTooltipHandler,
                    callbacks: {
                        title: function(items) {
                            if (!items.length) return '';
                            var ctx = items[0];
                            if (ctx.datasetIndex !== 0 || !h0Measurements) return '';
                            var mIdx = ctx.dataIndex - 1;
                            if (mIdx < 0) return '';
                            var m = h0Measurements.measurements[mIdx];
                            if (!m) return '';
                            return m.method + (m.year ? '  (' + m.year + ')' : '');
                        },
                        label: function(ctx) {
                            if (ctx.datasetIndex !== 0 || !h0Measurements) return null;
                            var mIdx = ctx.dataIndex - 1;
                            if (mIdx < 0) return null;
                            var m = h0Measurements.measurements[mIdx];
                            if (!m) return null;
                            var errStr;
                            if (m.err_low !== undefined && m.err_high !== undefined) {
                                errStr = '+' + m.err_high.toFixed(1) + ' / -' + m.err_low.toFixed(1);
                            } else {
                                errStr = '+/- ' + m.err.toFixed(1);
                            }
                            var raw = [
                                'H0 = ' + m.h0.toFixed(1) + '  ' + errStr + '  km/s/Mpc',
                                'Published: ' + (m.year || 'N/A')
                            ];
                            if (m.survey) raw.push('Survey: ' + m.survey);
                            if (m.telescope) raw.push('Telescope: ' + m.telescope);
                            if (m.instrument) raw.push('Instrument: ' + m.instrument);
                            if (m.note) {
                                raw.push('');
                                raw.push(m.note);
                            }
                            var lines = [];
                            for (var i = 0; i < raw.length; i++) {
                                var w = wrapText(raw[i], TOOLTIP_WRAP);
                                for (var j = 0; j < w.length; j++) lines.push(w[j]);
                            }
                            return lines;
                        },
                        labelTextColor: function() {
                            return '#ffe082';
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'H0 (km/s/Mpc)', color: '#808080', font: { size: 13 } },
                    min: 62, max: 82,
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: { color: '#808080' }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: {
                        color: '#b0b0b0',
                        font: { size: 12 },
                        callback: function(value, index) {
                            // Hide the spacer row label (index 0)
                            var label = this.getLabelForValue(value);
                            return label || '';
                        }
                    }
                }
            }
        },
        plugins: [{
            id: 'hubbleAnnotations',
            afterDraw: function(chart) {
                if (!h0Measurements) return;
                var ctx = chart.ctx;
                var xScale = chart.scales.x;
                var yScale = chart.scales.y;
                var area = chart.chartArea;

                // Spacer row y-center (index 0) for prediction labels
                var labelRowY = yScale.getPixelForValue(0);

                // Observed error bars (controlled by h-obs badge)
                // Measurements are at indices 1+ (index 0 is the spacer row)
                if (hubbleVisibility['h-obs']) {
                    var measurements = h0Measurements.measurements;
                    measurements.forEach(function(m, i) {
                        var xCenter = xScale.getPixelForValue(m.h0);
                        var errLow = m.err_low !== undefined ? m.err_low : m.err;
                        var errHigh = m.err_high !== undefined ? m.err_high : m.err;
                        var xLow = xScale.getPixelForValue(m.h0 - errLow);
                        var xHigh = xScale.getPixelForValue(m.h0 + errHigh);
                        var yCenter = yScale.getPixelForValue(i + 1);

                        ctx.save();
                        ctx.strokeStyle = '#e0e0e0';
                        ctx.lineWidth = 2;

                        ctx.beginPath();
                        ctx.moveTo(xLow, yCenter);
                        ctx.lineTo(xHigh, yCenter);
                        ctx.stroke();

                        ctx.beginPath();
                        ctx.moveTo(xLow, yCenter - 5);
                        ctx.lineTo(xLow, yCenter + 5);
                        ctx.stroke();

                        ctx.beginPath();
                        ctx.moveTo(xHigh, yCenter - 5);
                        ctx.lineTo(xHigh, yCenter + 5);
                        ctx.stroke();

                        ctx.fillStyle = '#e0e0e0';
                        ctx.beginPath();
                        ctx.arc(xCenter, yCenter, 4, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.restore();
                    });
                }

                // GFD predictions (controlled by h-gfd badge)
                if (hubbleVisibility['h-gfd']) {
                    var oneLoopVal = h0Measurements.gfd_one_loop;
                    var oneLoopX = xScale.getPixelForValue(oneLoopVal);

                    // Prediction band from lens throughput slider
                    var lensPct = parseFloat(document.getElementById('lens-slider').value) / 100.0;
                    var bandLow = xScale.getPixelForValue(oneLoopVal * (1 - lensPct));
                    var bandHigh = xScale.getPixelForValue(oneLoopVal * (1 + lensPct));
                    ctx.save();
                    ctx.fillStyle = 'rgba(77, 166, 255, 0.07)';
                    ctx.fillRect(bandLow, area.top, bandHigh - bandLow, area.bottom - area.top);
                    ctx.restore();

                    // One-loop: primary solid line, label to the RIGHT
                    drawVLine(ctx, oneLoopX, area.top, area.bottom,
                        '#4da6ff', 2.5, []);
                    drawVLabel(ctx, 'GFD: ' + oneLoopVal.toFixed(2),
                        oneLoopX + 6, labelRowY + 4, '#4da6ff', 'bold 11px sans-serif', 'left');
                }

                // MOND prediction (controlled by h-mond badge)
                if (hubbleVisibility['h-mond']) {
                    var mondVal = h0Measurements.mond_predicted;
                    var mondX = xScale.getPixelForValue(mondVal);

                    drawVLine(ctx, mondX, area.top, area.bottom,
                        '#9966ff', 2, [6, 4]);
                    drawVLabel(ctx, 'MOND: ' + mondVal.toFixed(2),
                        mondX - 6, labelRowY, '#9966ff', 'bold 11px sans-serif', 'right');
                }

                // H0 slider line (controlled by h-slider badge)
                if (hubbleVisibility['h-slider']) {
                    var sliderH0 = parseFloat(document.getElementById('h0-slider').value);
                    var sliderX = xScale.getPixelForValue(sliderH0);

                    drawVLine(ctx, sliderX, area.top, area.bottom,
                        '#FFC107', 1.5, [3, 3]);
                    drawVLabel(ctx, 'H0 = ' + sliderH0.toFixed(1),
                        sliderX - 6, labelRowY + 4, '#FFC107', 'bold 11px sans-serif', 'right');
                }
            }
        }]
    });
}

function populateHubbleChart() {
    if (!h0Measurements || !hubbleChart) return;

    var measurements = h0Measurements.measurements;
    // Prepend a spacer row for prediction labels
    var labels = [''].concat(measurements.map(function(m) { return m.method; }));
    var data = [null].concat(measurements.map(function(m) { return m.h0; }));
    var bgColors = ['rgba(0,0,0,0)'].concat(measurements.map(function() { return 'rgba(255, 255, 255, 0.0)'; }));
    var borderColors = ['rgba(0,0,0,0)'].concat(measurements.map(function() { return 'rgba(255, 255, 255, 0.0)'; }));
    var hoverBg = ['rgba(0,0,0,0)'].concat(measurements.map(function() { return 'rgba(255, 255, 255, 0.08)'; }));
    var hoverBorder = ['rgba(0,0,0,0)'].concat(measurements.map(function() { return 'rgba(255, 255, 255, 0.25)'; }));

    hubbleChart.data.labels = labels;
    hubbleChart.data.datasets[0].data = data;
    hubbleChart.data.datasets[0].backgroundColor = bgColors;
    hubbleChart.data.datasets[0].borderColor = borderColors;
    hubbleChart.data.datasets[0].hoverBackgroundColor = hoverBg;
    hubbleChart.data.datasets[0].hoverBorderColor = hoverBorder;

    hubbleChart.update('none');
}

// =====================================================================
// DATA FETCHING & CHART UPDATE
// =====================================================================
function getConfig() {
    return {
        z: parseFloat(redshiftSlider.value),
        H0: parseFloat(h0Slider.value),
        omega_m: 0.30,
        v0: parseFloat(v0Slider.value),
        lens_pct: parseFloat(lensSlider.value)
    };
}

function scheduleUpdate() {
    clearTimeout(updateTimer);
    updateTimer = setTimeout(function() {
        updateSliderDisplays();
        if (currentMode === 'tf') {
            fetchAndUpdateTf();
        } else {
            updateHubbleFromSlider();
        }
    }, DEBOUNCE_MS);
}

async function fetchAndUpdateTf() {
    var config = getConfig();

    var responses = await Promise.all([
        fetch('/api/redshift/curve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        }),
        fetch('/api/redshift/compute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        })
    ]);

    var curveData = await responses[0].json();
    var pointData = await responses[1].json();

    updateTfCurves(curveData);
    updateResults(pointData);
    updateRedshiftMarker(config.z);
    updateTfChartTitle();
    tfChart.update('none');
}

function updateTfCurves(data) {
    tfChart.data.datasets[0].data = data.z.map(function(z, i) { return { x: z, y: data.gfd[i] }; });
    tfChart.data.datasets[1].data = data.z.map(function(z, i) { return { x: z, y: data.lcdm[i] }; });
    tfChart.data.datasets[2].data = data.z.map(function(z, i) { return { x: z, y: data.gfd_upper[i] }; });
    tfChart.data.datasets[3].data = data.z.map(function(z, i) { return { x: z, y: data.gfd_lower[i] }; });
}

function updateResults(data) {
    resultZ.textContent = data.z.toFixed(2);
    resultHz.textContent = data.hz.toFixed(1);
    resultRatio.textContent = data.gfd_ratio.toFixed(4);
    resultVz.textContent = data.gfd_vz.toFixed(1);

    var pct = data.delta_pct;
    var sign = pct >= 0 ? '+' : '';
    resultDeltaPct.textContent = sign + pct.toFixed(1) + '%';
    resultDeltaPct.style.color = pct >= 0 ? '#4caf50' : '#ef5350';

    resultDc.textContent = data.comoving_distance_mpc.toFixed(1);
    resultDl.textContent = data.luminosity_distance_mpc.toFixed(1);
    resultDa.textContent = data.angular_diameter_distance_mpc.toFixed(1);
    resultTlb.textContent = data.lookback_time_gyr.toFixed(2);
}

function updateRedshiftMarker(z) {
    tfChart.data.datasets[6].data = [
        { x: z, y: 0.9 },
        { x: z, y: 1.5 }
    ];
}

function updateTfChartTitle() {
    if (currentExample) {
        var name = currentExample.name.replace(/\s*\(.*\)/, '');
        tfChart.options.plugins.title.text = name + ' : TF Velocity Evolution';
    } else {
        tfChart.options.plugins.title.text = 'DFP Tully-Fisher Velocity Evolution';
    }
}

function updateHubbleFromSlider() {
    if (hubbleChart) {
        hubbleChart.update('none');
    }
}

function loadObservedData() {
    fetch('/api/redshift/observations')
        .then(function(resp) { return resp.json(); })
        .then(function(data) {
            observedData = data.observations;
            sinsHighlight = data.sins_highlight;

            tfChart.data.datasets[4].data = observedData.map(function(obs) {
                return { x: obs.z, y: obs.ratio };
            });

            if (sinsHighlight) {
                tfChart.data.datasets[5].data = [
                    { x: sinsHighlight.z, y: sinsHighlight.ratio }
                ];
            }

            tfChart.update('none');
        });
}

function loadH0Measurements() {
    return fetch('/api/redshift/h0-measurements')
        .then(function(resp) { return resp.json(); })
        .then(function(data) {
            h0Measurements = data;
            populateHubbleChart();
        });
}

// =====================================================================
// MODE SWITCHING
// =====================================================================
function setMode(mode) {
    currentMode = mode;

    document.getElementById('mode-tf').classList.toggle('active', mode === 'tf');
    document.getElementById('mode-hubble').classList.toggle('active', mode === 'hubble');

    // Toggle chart canvases
    document.getElementById('tfCanvas').style.display = mode === 'tf' ? 'block' : 'none';
    document.getElementById('hubbleCanvas').style.display = mode === 'hubble' ? 'block' : 'none';

    // Toggle command bar elements
    document.getElementById('examples-group').style.display = mode === 'tf' ? '' : 'none';
    document.getElementById('redshift-slider-group').style.display = mode === 'tf' ? '' : 'none';

    // Toggle theory chips per mode
    document.getElementById('theory-toggles-tf').style.display = mode === 'tf' ? 'flex' : 'none';
    document.getElementById('theory-toggles-hubble').style.display = mode === 'hubble' ? 'flex' : 'none';

    // Toggle left panel sections
    document.getElementById('object-params-section').style.display = mode === 'tf' ? '' : 'none';
    document.getElementById('gfd-theory-section').style.display = mode === 'tf' ? '' : 'none';
    document.getElementById('computed-results').style.display = mode === 'tf' ? 'block' : 'none';
    // Close any open document viewer when switching modes
    if (typeof GravisDoc !== 'undefined' && GravisDoc.isOpen()) {
        GravisDoc.close();
    }

    if (mode === 'tf') {
        fetchAndUpdateTf();
    } else {
        if (hubbleChart) hubbleChart.update();
    }
}

// =====================================================================
// EXAMPLES DROPDOWN
// =====================================================================
function loadExamplesCatalog() {
    return fetch('/api/redshift/examples')
        .then(function(resp) { return resp.json(); })
        .then(function(examples) {
            examplesCatalog = examples;
            populateDropdown();

            var dropdown = document.getElementById('examples-dropdown');
            if (dropdown.options.length > 1) {
                dropdown.value = '1';
                loadExample();
            }
        });
}

function populateDropdown() {
    var dropdown = document.getElementById('examples-dropdown');
    dropdown.innerHTML = '';

    var placeholder = document.createElement('option');
    placeholder.value = '0';
    placeholder.textContent = 'Select an object...';
    dropdown.appendChild(placeholder);

    examplesCatalog.forEach(function(ex, i) {
        var opt = document.createElement('option');
        opt.value = String(i + 1);
        opt.textContent = ex.name;
        dropdown.appendChild(opt);
    });
}

function loadExample() {
    var dropdown = document.getElementById('examples-dropdown');
    var idx = parseInt(dropdown.value, 10);

    if (idx === 0) {
        currentExample = null;
        if (tfChart) {
            updateTfChartTitle();
            tfChart.update('none');
        }
        return;
    }

    var ex = examplesCatalog[idx - 1];
    if (!ex) return;

    currentExample = ex;

    redshiftSlider.value = ex.z;
    v0Slider.value = ex.v0;

    updateSliderDisplays();
    fetchAndUpdateTf();
}

// =====================================================================
// DATA SOURCES CARD
// =====================================================================
function openDataCard() {
    document.getElementById('chart-face').style.display = 'none';
    document.getElementById('data-face').style.display = 'block';

    var tbody = document.getElementById('dc-obs-tbody');
    tbody.innerHTML = '';
    observedData.forEach(function(obs) {
        var row = document.createElement('tr');
        row.innerHTML = '<td>' + obs.z.toFixed(2) + '</td>'
            + '<td>' + obs.ratio.toFixed(3) + ' +/- ' + obs.err.toFixed(3) + '</td>'
            + '<td>' + obs.source + '</td>';
        tbody.appendChild(row);
    });

    var refs = document.getElementById('dc-references');
    refs.innerHTML = '';
    var sources = [
        'Kassin+2007 ApJ 660 L35 (DEEP2/AEGIS TF kinematics)',
        'Puech+2008 A&A 484 173 (IMAGES VLT/FLAMES z~0.6)',
        'Cresci+2009 ApJ 697 115 (SINS VLT/SINFONI z~2)',
        'Miller+2011 ApJ 741 115 (DEEP2 Keck/DEIMOS z~1)',
        'Ubler+2017 ApJ 842 121 (KMOS3D z~0.6-2.6)',
        'Straatman+2017 ApJ 839 57 (ZFIRE z~2.2)',
    ];
    sources.forEach(function(ref) {
        var li = document.createElement('li');
        li.textContent = ref;
        refs.appendChild(li);
    });

    var summary = document.getElementById('dc-summary');
    summary.innerHTML = '<p style="color:#b0b0b0;font-size:0.9em;line-height:1.6;">'
        + 'Tully-Fisher velocity evolution measurements from integral field unit (IFU) '
        + 'surveys of high-redshift disk galaxies. Each data point represents the ratio '
        + 'v(z)/v(0) of rotation velocity at redshift z to the local TF calibration at '
        + 'matched stellar mass.'
        + '</p>';
}

function closeDataCard() {
    document.getElementById('data-face').style.display = 'none';
    document.getElementById('chart-face').style.display = 'block';
}

// =====================================================================
// RESIZE HANDLE
// =====================================================================
function initResizeHandle() {
    var handle = document.getElementById('resize-handle');
    var leftPanel = document.getElementById('left-panel');
    var isDragging = false;
    var startX, startWidth;

    handle.addEventListener('mousedown', function(e) {
        isDragging = true;
        startX = e.clientX;
        startWidth = leftPanel.offsetWidth;
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    document.addEventListener('mousemove', function(e) {
        if (!isDragging) return;
        var newWidth = startWidth + (e.clientX - startX);
        if (newWidth >= 200 && newWidth <= 500) {
            leftPanel.style.width = newWidth + 'px';
        }
    });

    document.addEventListener('mouseup', function() {
        if (isDragging) {
            isDragging = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            if (tfChart) tfChart.resize();
            if (hubbleChart) hubbleChart.resize();
        }
    });
}

// =====================================================================
// RESET ZOOM
// =====================================================================
function resetZoom() {
    if (currentMode === 'tf' && tfChart) {
        tfChart.resetZoom();
    }
    document.getElementById('reset-zoom-btn').style.display = 'none';
}

// =====================================================================
// INIT
// =====================================================================
function init() {
    // Grab DOM references
    redshiftSlider = document.getElementById('redshift-slider');
    redshiftValue = document.getElementById('redshift-value');
    h0Slider = document.getElementById('h0-slider');
    h0Value = document.getElementById('h0-value');
    lensSlider = document.getElementById('lens-slider');
    lensValue = document.getElementById('lens-value');
    v0Slider = document.getElementById('v0-slider');
    v0Value = document.getElementById('v0-value');

    resultZ = document.getElementById('result-z');
    resultHz = document.getElementById('result-hz');
    resultRatio = document.getElementById('result-ratio');
    resultVz = document.getElementById('result-vz');
    resultDeltaPct = document.getElementById('result-delta-pct');
    resultDc = document.getElementById('result-dc');
    resultDl = document.getElementById('result-dl');
    resultDa = document.getElementById('result-da');
    resultTlb = document.getElementById('result-tlb');

    // Create both charts
    createTfChart();
    createHubbleChart();

    // Hide hubble canvas initially
    document.getElementById('hubbleCanvas').style.display = 'none';

    initTheoryChips();
    initResizeHandle();

    // Wire up sliders
    [redshiftSlider, h0Slider, lensSlider, v0Slider].forEach(function(slider) {
        slider.addEventListener('input', scheduleUpdate);
    });

    document.getElementById('reset-zoom-btn').addEventListener('click', resetZoom);

    // Load data
    loadObservedData();
    loadH0Measurements();
    loadExamplesCatalog();
}

// Expose functions for inline onclick handlers
window.setMode = setMode;
window.loadExample = loadExample;
window.openDataCard = openDataCard;
window.closeDataCard = closeDataCard;

// Wait for Chart.js to load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(init, 100);
    });
} else {
    setTimeout(init, 100);
}
