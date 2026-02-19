"""
Vortex-mode mirror with first-observer cut-off and optional GFD bridge.

Rule: The first observation point is the cut-off for GFD Photometric, MOND,
Newtonian, and CDM. When mirror (vortex) is set: mirror for r < -r_first_obs,
then either a gap (no bridge) or a GFD-anchored bridge through r = -1, 0, 1,
then real for r >= r_first_obs.

No unicode (Windows charmap).
"""

import numpy as np

BRIDGE_NUM_POINTS = 25


def _value_at(radii, series, r_abs):
    """Linear interpolation: value at r_abs from (radii, series)."""
    if not radii or not series or len(radii) != len(series):
        return None
    radii = np.asarray(radii)
    series = np.asarray(series)
    if r_abs <= radii[0]:
        return float(series[0])
    if r_abs >= radii[-1]:
        return float(series[-1])
    for i in range(len(radii) - 1):
        if radii[i] <= r_abs <= radii[i + 1]:
            t = (r_abs - radii[i]) / (radii[i + 1] - radii[i])
            return float(series[i] + t * (series[i + 1] - series[i]))
    return float(series[-1])


def truncate_to_first_obs(radii, series_dict, r_first_obs):
    """
    Truncate radii and series to r >= r_first_obs (first-principle cut-off).

    Theory curves (GFD Photometric, MOND, Newtonian, CDM) must not be
    shown inside the first observation radius.

    Parameters
    ----------
    radii : list of float
        Positive radii (kpc), sorted ascending.
    series_dict : dict[str, list]
        Series name -> list of values (same length as radii).
    r_first_obs : float
        Minimum observation radius (kpc). If <= 0, no truncation.

    Returns
    -------
    radii_out : list of float
        radii filtered to r >= r_first_obs.
    series_out : dict[str, list]
        Each series filtered to same indices.
    """
    if not radii or r_first_obs <= 0:
        return list(radii), {k: list(v) for k, v in series_dict.items()}

    radii = np.asarray(radii)
    mask = radii >= r_first_obs
    if not np.any(mask):
        return [], {k: [] for k in series_dict}

    radii_out = [round(float(r), 6) for r in radii[mask]]
    series_out = {}
    for key, series in series_dict.items():
        if not series or len(series) != len(radii):
            series_out[key] = []
            continue
        series = np.asarray(series)
        series_out[key] = [round(float(v), 4) for v in series[mask]]
    return radii_out, series_out


def _bridge_values_for_series(r_bridge, key_radii, key_vals):
    """Linear interpolation from key (radius, value) to each r in r_bridge."""
    key_radii = np.asarray(key_radii, dtype=float)
    key_vals = np.asarray(key_vals, dtype=float)
    return [round(float(np.interp(r, key_radii, key_vals)), 4) for r in r_bridge]


def mirror_curve_with_cutoff(radii, series_dict, r_first_obs, bridge_fv_percent=1.0,
                            gfd_bridge_velocities=None):
    """
    Vortex mode: mirror for r < -r_first_obs; then bridge or gap; then real for r >= r_first_obs.

    If gfd_bridge_velocities is (v_at_minus_1, v_at_0, v_at_1), a bridge segment
    is inserted that passes through those GFD anchor points and connects to mirror
    at -r_first_obs and real at r_first_obs (linear interpolation between key points).
    Otherwise a single gap point at r=0 (value None) is used so the chart line breaks.

    Parameters
    ----------
    radii : list of float
        Full positive radii (kpc), sorted ascending.
    series_dict : dict[str, list]
        Full series (same length as radii).
    r_first_obs : float
        First observation radius (kpc). Cut-off boundary.
    bridge_fv_percent : float
        Unused; kept for API compatibility.
    gfd_bridge_velocities : tuple (v_m1, v0, v1) or None
        GFD raw velocity at r = -1, 0, 1 kpc. If provided, bridge through these points.

    Returns
    -------
    radii_sym : list of float
        Radii: mirror segment + bridge (or gap) + real segment.
    series_sym : dict[str, list]
        Mirrored + bridge/gap + real values.
    """
    if not radii:
        return [], {k: [] for k in series_dict}

    radii = np.asarray(radii, dtype=float)
    r_first_obs = float(r_first_obs)
    if r_first_obs <= 0:
        r_first_obs = radii[0] if radii[0] > 0 else 0.01

    # Real side: only r >= r_first_obs
    mask_pos = radii >= r_first_obs
    if not np.any(mask_pos):
        return [], {k: [] for k in series_dict}

    radii_pos = radii[mask_pos]
    # Mirror side: r < -r_first_obs (use v(|r|) from full curve)
    radii_neg = sorted([-float(r) for r in radii if r > r_first_obs])

    use_bridge = (
        gfd_bridge_velocities is not None
        and len(gfd_bridge_velocities) == 3
        and all(x is not None for x in gfd_bridge_velocities)
    )
    if use_bridge:
        v_m1, v0, v1 = [float(x) for x in gfd_bridge_velocities]
        # Bridge radii from -r_first_obs to r_first_obs (include -1, 0, 1)
        r_bridge = np.linspace(-r_first_obs, r_first_obs, BRIDGE_NUM_POINTS).tolist()
        radii_sym = radii_neg + r_bridge + radii_pos.tolist()
        radii_sym = [round(r, 6) for r in radii_sym]

        # Key points: must be sorted for np.interp. At -1, 0, 1 use average of
        # theory endpoint-to-endpoint linear and GFD raw so the bridge is smooth
        # (avoids jagged edge when theory differs strongly from GFD at the core).
        key_radii = sorted([-r_first_obs, -1.0, 0.0, 1.0, r_first_obs])
        span = 2.0 * r_first_obs
        series_sym = {}
        for key, series in series_dict.items():
            if not series or len(series) != len(radii):
                series_sym[key] = []
                continue
            series = np.asarray(series, dtype=float)
            v_at_neg_cutoff = _value_at(radii.tolist(), series.tolist(), r_first_obs)
            v_at_pos_cutoff = float(series[mask_pos][0])
            if v_at_neg_cutoff is None:
                v_at_neg_cutoff = v_at_pos_cutoff
            # Theory value at r by linear interpolation between the two endpoints
            def v_theory_linear(r):
                t = (float(r) + r_first_obs) / span if span > 0 else 0.5
                return v_at_neg_cutoff + t * (v_at_pos_cutoff - v_at_neg_cutoff)
            # Anchor points: half theory, half GFD raw
            v_at_m1 = 0.5 * (v_theory_linear(-1.0) + v_m1)
            v_at_0 = 0.5 * (v_theory_linear(0.0) + v0)
            v_at_1 = 0.5 * (v_theory_linear(1.0) + v1)
            r_to_val = {
                -r_first_obs: v_at_neg_cutoff, -1.0: v_at_m1, 0.0: v_at_0, 1.0: v_at_1,
                r_first_obs: v_at_pos_cutoff
            }
            key_vals = [r_to_val[r] for r in key_radii]
            out = []
            for r in radii_neg:
                v = _value_at(radii.tolist(), series.tolist(), abs(r))
                out.append(round(v, 4) if v is not None else 0.0)
            out.extend(_bridge_values_for_series(r_bridge, key_radii, key_vals))
            for i in np.where(mask_pos)[0]:
                out.append(round(float(series[i]), 4))
            series_sym[key] = out
    else:
        # Gap: one point at r=0 with value None so chart draws no line
        radii_sym = radii_neg + [0.0] + radii_pos.tolist()
        radii_sym = [round(r, 6) for r in radii_sym]
        series_sym = {}
        for key, series in series_dict.items():
            if not series or len(series) != len(radii):
                series_sym[key] = []
                continue
            series = np.asarray(series, dtype=float)
            out = []
            for r in radii_neg:
                v = _value_at(radii.tolist(), series.tolist(), abs(r))
                out.append(round(v, 4) if v is not None else 0.0)
            out.append(None)
            for i in np.where(mask_pos)[0]:
                out.append(round(float(series[i]), 4))
            series_sym[key] = out

    return radii_sym, series_sym


def mirror_curve_symmetric(radii, series_dict):
    """
    Legacy: full symmetric mirror (no cut-off). Prefer mirror_curve_with_cutoff
    when observations are available.
    """
    if not radii:
        return [], {k: [] for k in series_dict}

    radii = list(radii)
    neg_radii = [-r for r in radii if r > 0]
    neg_radii.sort()
    radii_sym = neg_radii + radii

    series_sym = {}
    for key, series in series_dict.items():
        if not series or len(series) != len(radii):
            series_sym[key] = []
            continue
        out = []
        for r in radii_sym:
            v = _value_at(radii, series, abs(r))
            out.append(round(v, 4) if v is not None else 0.0)
        series_sym[key] = out

    radii_sym = [round(r, 6) for r in radii_sym]
    return radii_sym, series_sym
