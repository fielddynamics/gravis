/**
 * GRAVIS Inference Page
 *
 * Galaxy selector + Bayesian GFD base fit chart.
 * Plots observation data and the covariant completion (GFD base)
 * fitted via differential_evolution + L-BFGS-B.
 */

(function () {
    "use strict";

    let chart = null;
    let lastFitData = null;
    let currentGalaxyId = null;

    // ================================================================
    // GALAXY DROPDOWN
    // ================================================================

    async function loadGalaxies() {
        const dropdown = document.getElementById("galaxy-dropdown");
        try {
            const resp = await fetch("/api/rotation/galaxies");
            const data = await resp.json();

            for (const [group, galaxies] of Object.entries(data)) {
                if (!Array.isArray(galaxies)) continue;
                const validGalaxies = galaxies.filter(
                    g => g.observations && g.observations.length >= 3
                );
                if (validGalaxies.length === 0) continue;

                const optgroup = document.createElement("optgroup");
                optgroup.label = group;
                for (const g of validGalaxies) {
                    const opt = document.createElement("option");
                    opt.value = g.id;
                    opt.textContent = g.name || g.id;
                    optgroup.appendChild(opt);
                }
                dropdown.appendChild(optgroup);
            }
        } catch (e) {
            console.error("Failed to load galaxies:", e);
        }
    }

    // ================================================================
    // API CALL
    // ================================================================

    async function runBayesianFit(galaxyId, chartMaxR) {
        const loading = document.getElementById("loading-indicator");
        const results = document.getElementById("fit-results");
        const horizonPanel = document.getElementById("horizon-panel");
        loading.style.display = "block";
        results.style.display = "none";

        const payload = { galaxy_id: galaxyId };
        if (chartMaxR > 0) {
            // Request data 1% past the target so the curve always
            // reaches beyond the R_env line on the graph
            payload.chart_max_r = chartMaxR * 1.01;
        }

        try {
            const resp = await fetch("/api/sandbox/map_gfd_with_bayesian", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await resp.json();
            if (data.error) {
                loading.style.display = "none";
                alert("Error: " + data.error);
                return;
            }
            loading.style.display = "none";
            results.style.display = "block";
            horizonPanel.style.display = "block";
            lastFitData = data;

            // Set slider to SPARC galactic_radius on first load
            if (!chartMaxR) {
                const sparc = data.sparc_galactic_radius || 0;
                const derived = (data.field_geometry || {}).envelope_radius_kpc || 0;
                const rvis = (data.field_geometry || {}).visible_radius_kpc || 0;
                const maxObs = Math.max(...(data.residuals || []).map(r => r.r), 0);
                const defaultEnv = Math.max(sparc, derived, rvis, maxObs) * 1.1;
                const slider = document.getElementById("horizon-slider");
                slider.value = defaultEnv.toFixed(1);
                slider.max = Math.max(200, defaultEnv * 2);
                document.getElementById("horizon-value").textContent = defaultEnv.toFixed(1) + " kpc";
            }

            renderResults(data);
            renderChart(data);
        } catch (e) {
            loading.style.display = "none";
            console.error("Bayesian fit failed:", e);
            alert("Request failed: " + e.message);
        }
    }

    // ================================================================
    // HORIZON SLIDER CHANGE: re-fetch with new chart range
    // ================================================================

    function onHorizonChange() {
        const slider = document.getElementById("horizon-slider");
        const val = parseFloat(slider.value);
        document.getElementById("horizon-value").textContent = val.toFixed(1) + " kpc";

        if (!lastFitData) return;

        // Update the chart x-axis, R_env line, and R_t line from the slider
        const horizonR = val;
        const throatR = horizonR * 0.30;

        // Update the vertical lines and x-axis without re-fetching
        // (the fit result doesn't change, just the display range)
        renderChartWithHorizon(lastFitData, horizonR, throatR);
    }

    // ================================================================
    // RESULTS PANEL
    // ================================================================

    function fmtSci(v) {
        if (v === null || v === undefined) return "N/A";
        if (Math.abs(v) < 1e4) return v.toFixed(2);
        return v.toExponential(3);
    }

    function fmtDelta(d) {
        if (d === null || d === undefined) return "";
        const sign = d >= 0 ? "+" : "";
        const color = Math.abs(d) < 10 ? "#4da6ff" : (Math.abs(d) < 30 ? "#ffaa00" : "#ff6688");
        return "<span style='color:" + color + "'>" + sign + d.toFixed(1) + "%</span>";
    }

    function massRow(label, obsVal, bayVal, deltaVal, unit) {
        var html = "<div style='margin-bottom:2px'>" + label + "</div>";
        html += "<div style='display:flex;align-items:baseline;gap:6px;margin-left:10px;margin-bottom:6px'>";
        html += "<span style='color:#ffaa00;min-width:95px'>" + fmtSci(obsVal) + "</span>";
        html += "<span style='color:#555;font-size:0.85em'>vs</span>";
        html += "<span style='color:#4da6ff;min-width:95px'>" + fmtSci(bayVal) + "</span>";
        if (unit) html += "<span style='color:#555;font-size:0.8em'>" + unit + "</span>";
        html += "<span style='min-width:55px;text-align:right'>" + fmtDelta(deltaVal) + "</span>";
        html += "</div>";
        return html;
    }

    function renderResults(data) {
        const fitEl = document.getElementById("fit-info");
        var fitHtml =
            "RMS (Bayesian): <span style='color:#4da6ff'>" + data.rms + " km/s</span><br>" +
            "chi2/dof: " + data.chi2_dof + "<br>" +
            "Observations: " + data.n_obs;

        const vs = data.vortex_signal;
        if (vs) {
            fitHtml += "<div style='border-top:1px solid #333;margin-top:8px;padding-top:8px'>";
            fitHtml += "<div style='color:#ff44dd;font-weight:600'>Vortex Signal (MACD)</div>";
            fitHtml += "sigma (net): <b>" + vs.sigma_net + "</b><br>";
            fitHtml += "Boost energy: +" + vs.energy_boost.toLocaleString() + " km<sup>2</sup>/s<sup>2</sup><br>";
            fitHtml += "Suppress energy: " + vs.energy_suppress.toLocaleString() + " km<sup>2</sup>/s<sup>2</sup><br>";
            fitHtml += "Boost/Suppress: " + vs.energy_ratio;
            fitHtml += "</div>";
        }
        fitEl.innerHTML = fitHtml;

        const geomEl = document.getElementById("geom-info");
        const fg = data.field_geometry || {};
        const rt = fg.throat_radius_kpc;
        const renv = fg.envelope_radius_kpc;
        const tf = fg.throat_fraction;
        const rvis = fg.visible_radius_kpc;
        const rvis90 = fg.visible_radius_90_kpc || 0;
        const rvis99 = fg.visible_radius_99_kpc || 0;
        const maxObsR = Math.max(...(data.residuals || []).map(r => r.r), 0);
        var geomHtml =
            "R_t (throat): <span style='color:#00ff88'>" + (rt ? rt.toFixed(2) + " kpc" : "N/A") + "</span><br>" +
            "R_env (field horizon): <span style='color:#ff6688'>" + (renv ? renv.toFixed(2) + " kpc" : "N/A") + "</span><br>" +
            "R_vis (baryonic extent): <span style='color:#aa88ff'>" +
                (rvis90 && rvis99
                    ? rvis90.toFixed(1) + " to " + rvis99.toFixed(1) + " kpc <span style='font-size:0.85em;color:#888'>(90% to 99.5% enclosed)</span>"
                    : (rvis ? rvis.toFixed(2) + " kpc" : "N/A")) +
            "</span><br>" +
            "R_t / R_env: " + (tf ? tf.toFixed(4) : "N/A") + "<br>" +
            "Cycle: " + (fg.cycle || "?");
        if (renv && rvis99 && rvis99 > renv * 1.05) {
            geomHtml += "<div style='color:#aa88ff;margin-top:6px;font-size:0.9em'>" +
                "Baryonic matter extends " + (rvis99 / renv).toFixed(1) +
                "x beyond R_env (deep field)</div>";
        }
        geomEl.innerHTML = geomHtml;

        // Mass Distribution comparison panel
        const massEl = document.getElementById("mass-comparison");
        const mc = data.mass_comparison;
        if (!mc) {
            massEl.innerHTML = "<span style='color:#555'>No mass comparison available</span>";
            return;
        }

        var html = "";
        html += "<div style='display:flex;gap:6px;margin-bottom:8px;font-size:0.78em;color:#666'>";
        html += "<span style='min-width:10px'></span>";
        html += "<span style='min-width:95px;color:#ffaa00'>Observed</span>";
        html += "<span style='min-width:15px'></span>";
        html += "<span style='min-width:95px;color:#4da6ff'>Bayesian</span>";
        html += "<span></span>";
        html += "<span style='color:#888'>delta</span>";
        html += "</div>";

        // Gas Disk
        var g = mc.gas || {};
        html += "<div style='color:#aaa;font-weight:600;margin-top:6px'>Gas Disk (HI + H2 + He)</div>";
        html += massRow("Mass", g.obs_M, g.bay_M, g.delta_M, "M_sun");
        html += massRow("Scale length (R_d)", g.obs_Rd, g.bay_Rd, g.delta_Rd, "kpc");

        // Stellar Disk
        var d = mc.disk || {};
        html += "<div style='color:#aaa;font-weight:600;margin-top:8px'>Stellar Disk (Exponential)</div>";
        html += massRow("Mass", d.obs_M, d.bay_M, d.delta_M, "M_sun");
        html += massRow("Scale length (R_d)", d.obs_Rd, d.bay_Rd, d.delta_Rd, "kpc");

        // Stellar Bulge
        var b = mc.bulge || {};
        html += "<div style='color:#aaa;font-weight:600;margin-top:8px'>Stellar Bulge (Hernquist)</div>";
        html += massRow("Mass", b.obs_M, b.bay_M, b.delta_M, "M_sun");
        html += massRow("Scale length (a)", b.obs_a, b.bay_a, b.delta_a, "kpc");

        // Total
        var t = mc.total || {};
        html += "<div style='border-top:1px solid #333;margin-top:10px;padding-top:8px'>";
        html += massRow("Total Baryonic Mass", t.obs_M, t.bay_M, t.delta_M, "M_sun");
        html += "</div>";

        massEl.innerHTML = html;
    }

    // ================================================================
    // CHART RENDERING
    // ================================================================

    function renderChart(data) {
        renderChartCore(data);
    }

    function renderChartWithHorizon(data, horizonR, throatR) {
        renderChartCore(data, horizonR);
    }

    function renderChartCore(data, xOverride) {
        const canvas = document.getElementById("inferenceChart");
        const ctx = canvas.getContext("2d");

        if (chart) {
            chart.destroy();
            chart = null;
        }

        const residuals = data.residuals || [];
        const chartData = data.chart || {};
        const fg = data.field_geometry || {};

        // Derived topology
        const derivedRt = fg.throat_radius_kpc || 0;
        const derivedRenv = fg.envelope_radius_kpc || 0;
        const derivedRvis = fg.visible_radius_kpc || 0;
        const derivedRvis90 = fg.visible_radius_90_kpc || 0;
        const derivedRvis99 = fg.visible_radius_99_kpc || 0;
        const maxObsR = residuals.length > 0
            ? Math.max(...residuals.map(r => r.r)) : 0;

        // Observation data points
        const obsPoints = residuals.map(r => ({ x: r.r, y: r.v_obs }));
        const obsErrors = residuals.map(r => r.err);

        // GFD base curve from Bayesian fit
        const gfdRadii = chartData.radii || [];
        const gfdVels = chartData.gfd_base || [];
        const gfdPoints = gfdRadii.map((r, i) => ({ x: r, y: gfdVels[i] }));

        // GFD curve from photometric mass parameters (topology-corrected)
        const photoVels = chartData.gfd_photometric || [];
        const photoPoints = gfdRadii.map((r, i) => ({ x: r, y: photoVels[i] }));
        const hasPhoto = photoVels.length > 0;

        // GFD full covariant curve (photometric masses + fitted sigma)
        const covariantVels = chartData.gfd_covariant || [];
        const covariantPoints = gfdRadii.map((r, i) => ({ x: r, y: covariantVels[i] }));
        const hasCovariant = covariantVels.length > 0;

        // X-axis: cover derived topology, visible horizon, AND all observations
        const xMax = xOverride
            ? xOverride * 1.10
            : Math.max(derivedRenv, derivedRvis99 || derivedRvis, maxObsR) * 1.10;

        const yMin = 0;
        const obsMax = obsPoints.length > 0
            ? Math.max(...obsPoints.map(p => p.y)) : 0;
        const yMax = obsMax > 0 ? Math.ceil(obsMax * 1.2 / 10) * 10 : undefined;

        const datasets = [];

        // Filter curve data to only include points within visible x range
        // to prevent Chart.js interpolation artifacts at edges.
        const gfdVisible = gfdPoints.filter(p => p.x <= xMax);
        const photoVisible = photoPoints.filter(p => p.x <= xMax);
        const covariantVisible = covariantPoints.filter(p => p.x <= xMax);

        // 1. GFD Bayesian fit curve (blue, solid)
        datasets.push({
            label: "GFD Bayesian Fit",
            data: gfdVisible,
            borderColor: "rgba(77, 166, 255, 0.6)",
            backgroundColor: "transparent",
            borderWidth: 2,
            pointRadius: 0,
            showLine: true,
            tension: 0,
            order: 4,
        });

        // 2. GFD Photometric base curve (green dashed)
        if (hasPhoto) {
            datasets.push({
                label: "GFD Photometric (no vortex)",
                data: photoVisible,
                borderColor: "rgba(0, 229, 160, 0.45)",
                backgroundColor: "transparent",
                borderWidth: 2,
                borderDash: [8, 4],
                pointRadius: 0,
                showLine: true,
                tension: 0,
                order: 3,
            });
        }

        // 3. GFD Covariant = photometric + vortex delta (should match blue)
        if (hasCovariant) {
            datasets.push({
                label: "GFD Covariant (photo + vortex)",
                data: covariantVisible,
                borderColor: "#ff44dd",
                backgroundColor: "transparent",
                borderWidth: 2.5,
                pointRadius: 0,
                showLine: true,
                tension: 0,
                order: 2,
            });
        }

        // 4. Observation points with error bars
        datasets.push({
            label: "Observed Data",
            data: obsPoints,
            borderColor: "#ffaa00",
            backgroundColor: "#ffaa00",
            pointRadius: 5,
            pointHoverRadius: 7,
            pointBorderWidth: 1.5,
            pointBorderColor: "#ffaa00",
            pointBackgroundColor: "#ffaa00",
            showLine: false,
            order: 1,
            errorBars: obsErrors,
        });

        // Vertical lines from derived field geometry
        const vLines = [];
        if (derivedRt > 0) {
            vLines.push({
                x: derivedRt,
                color: "#00ff88",
                label: "R_t = " + derivedRt.toFixed(1) + " kpc",
            });
        }
        if (derivedRenv > 0) {
            vLines.push({
                x: derivedRenv,
                color: "#ff6688",
                label: "R_env = " + derivedRenv.toFixed(1) + " kpc",
            });
        }
        // R_vis band: shaded region showing baryonic mass extent (90% to ~100%)
        const vBands = [];
        if (derivedRvis90 > 0 && derivedRvis99 > 0) {
            vBands.push({
                x1: derivedRvis90,
                x2: derivedRvis99,
                color: "rgba(170,136,255,0.15)",
                label: "Baryonic extent (" + derivedRvis90.toFixed(0) + " to " + derivedRvis99.toFixed(0) + " kpc)",
                labelColor: "#aa88ff",
            });
        } else if (derivedRvis > 0) {
            vLines.push({
                x: derivedRvis,
                color: "#aa88ff",
                dash: [3, 3],
                label: "R_vis = " + derivedRvis.toFixed(1) + " kpc",
            });
        }

        chart = new Chart(ctx, {
            type: "scatter",
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                plugins: {
                    verticalLines: vLines,
                    verticalBands: vBands,
                    title: {
                        display: true,
                        text: (data.galaxy || "Galaxy") + " : GFD Velocity Curves",
                        color: "#e0e0e0",
                        font: { size: 16, weight: "normal" },
                        padding: { bottom: 16 },
                    },
                    legend: {
                        labels: { color: "#b0b0b0", usePointStyle: true },
                    },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                const ds = ctx.dataset;
                                const p = ctx.raw;
                                if (ds.errorBars) {
                                    const err = ds.errorBars[ctx.dataIndex];
                                    return ds.label + ": v=" + p.y.toFixed(1) +
                                        " +/- " + err.toFixed(1) + " km/s at r=" +
                                        p.x.toFixed(2) + " kpc";
                                }
                                if (p.y !== undefined) {
                                    return ds.label + ": " + p.y.toFixed(1) +
                                        " km/s at r=" + p.x.toFixed(2) + " kpc";
                                }
                                return ds.label;
                            },
                        },
                    },
                },
                scales: {
                    x: {
                        type: "linear",
                        title: {
                            display: true,
                            text: "Galactocentric Radius (kpc)",
                            color: "#909090",
                        },
                        ticks: { color: "#808080" },
                        grid: { color: "#333333" },
                        min: 0,
                        max: xMax,
                    },
                    y: {
                        type: "linear",
                        title: {
                            display: true,
                            text: "Circular Velocity v(r) [km/s]",
                            color: "#909090",
                        },
                        ticks: { color: "#808080" },
                        grid: { color: "#333333" },
                        min: yMin,
                        max: yMax,
                    },
                },
            },
            plugins: [errorBarPlugin, verticalLinePlugin],
        });
    }

    // ================================================================
    // VERTICAL LINE PLUGIN
    // ================================================================

    const verticalLinePlugin = {
        id: "verticalLines",
        afterDatasetsDraw(chart) {
            const ctx = chart.ctx;
            const xScale = chart.scales.x;
            const yScale = chart.scales.y;

            // Draw shaded bands first (behind lines)
            const bands = chart.options.plugins.verticalBands || [];
            bands.forEach(band => {
                const x1 = xScale.getPixelForValue(band.x1);
                const x2 = xScale.getPixelForValue(band.x2);
                if (x2 < xScale.left || x1 > xScale.right) return;
                const left = Math.max(x1, xScale.left);
                const right = Math.min(x2, xScale.right);

                ctx.save();
                ctx.fillStyle = band.color || "rgba(170,136,255,0.12)";
                ctx.fillRect(left, yScale.top, right - left, yScale.bottom - yScale.top);

                if (band.label) {
                    ctx.fillStyle = band.labelColor || "#aa88ff";
                    ctx.font = "11px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText(band.label, (left + right) / 2, yScale.top - 6);
                }
                ctx.restore();
            });

            const lines = chart.options.plugins.verticalLines || [];
            lines.forEach(line => {
                const xPx = xScale.getPixelForValue(line.x);
                if (xPx < xScale.left || xPx > xScale.right) return;

                ctx.save();
                ctx.strokeStyle = line.color || "#ffffff";
                ctx.lineWidth = line.width || 1.5;
                ctx.setLineDash(line.dash || [6, 4]);
                ctx.beginPath();
                ctx.moveTo(xPx, yScale.top);
                ctx.lineTo(xPx, yScale.bottom);
                ctx.stroke();

                if (line.label) {
                    ctx.setLineDash([]);
                    ctx.fillStyle = line.color || "#ffffff";
                    ctx.font = "11px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText(line.label, xPx, yScale.top - 6);
                }
                ctx.restore();
            });
        },
    };

    // ================================================================
    // ERROR BAR PLUGIN
    // ================================================================

    const errorBarPlugin = {
        id: "errorBars",
        afterDatasetsDraw(chart) {
            const ctx = chart.ctx;
            chart.data.datasets.forEach((ds, dsIndex) => {
                if (!ds.errorBars) return;
                const meta = chart.getDatasetMeta(dsIndex);
                if (meta.hidden) return;

                ctx.save();
                ctx.strokeStyle = ds.borderColor || "#ffaa00";
                ctx.lineWidth = 1.2;
                const capW = 4;

                meta.data.forEach((point, i) => {
                    const err = ds.errorBars[i];
                    if (!err || err <= 0) return;

                    const yScale = chart.scales.y;
                    const yVal = ds.data[i].y;
                    const yTop = yScale.getPixelForValue(yVal + err);
                    const yBot = yScale.getPixelForValue(yVal - err);
                    const x = point.x;

                    ctx.beginPath();
                    ctx.moveTo(x, yTop);
                    ctx.lineTo(x, yBot);
                    ctx.stroke();

                    ctx.beginPath();
                    ctx.moveTo(x - capW, yTop);
                    ctx.lineTo(x + capW, yTop);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(x - capW, yBot);
                    ctx.lineTo(x + capW, yBot);
                    ctx.stroke();
                });
                ctx.restore();
            });
        },
    };

    // ================================================================
    // EVENT WIRING
    // ================================================================

    document.addEventListener("DOMContentLoaded", function () {
        loadGalaxies();

        document.getElementById("galaxy-dropdown").addEventListener("change", function () {
            currentGalaxyId = this.value;
            if (currentGalaxyId) {
                runBayesianFit(currentGalaxyId, 0);
            }
        });

        const slider = document.getElementById("horizon-slider");
        let sliderTimeout = null;
        slider.addEventListener("input", function () {
            const val = parseFloat(this.value);
            document.getElementById("horizon-value").textContent = val.toFixed(1) + " kpc";

            // Immediate local re-render (just move lines and x-axis)
            if (lastFitData) {
                renderChartWithHorizon(lastFitData, val, val * 0.30);
            }
        });

        // On slider release, re-fetch with chart data extending 1% past the horizon
        slider.addEventListener("change", function () {
            if (currentGalaxyId) {
                const val = parseFloat(this.value);
                runBayesianFit(currentGalaxyId, val * 1.01);
            }
        });
    });
})();
