"""
Field equation and solver integrity tests.

These verify mathematical properties of the covariant field equation
x^2/(1+x) = y_N that follows from the Lagrangian F(y), and confirm
that the analytic solver is correct, continuous, and consistent with
the theory's limiting behaviors.
"""

import math
import pytest

from physics.aqual import solve_x, velocity as dtg_velocity
from physics.mond import solve_x as mond_solve_x
from physics.constants import G, M_SUN, KPC_TO_M, A0


class TestSolveXInverse:
    """solve_x must be the exact inverse of the field equation."""

    @pytest.mark.parametrize("x", [
        0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
        50.0, 100.0, 1000.0, 1e6, 1e10,
    ])
    def test_solve_x_inverts_field_equation(self, x):
        """For known x, compute y_N = x^2/(1+x), solve back, recover x."""
        y_N = x * x / (1.0 + x)
        x_recovered = solve_x(y_N)
        assert x_recovered == pytest.approx(x, rel=1e-10), \
            f"x={x}: y_N={y_N}, recovered={x_recovered}"


class TestSolverContinuity:
    """solve_x must be continuous -- no jumps or discontinuities."""

    def test_small_perturbations_produce_small_changes(self):
        """Perturbing y_N by 0.01% should perturb x by a small amount."""
        base_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        for y_N in base_values:
            x1 = solve_x(y_N)
            x2 = solve_x(y_N * 1.0001)  # 0.01% perturbation
            # Change in x should be small relative to x
            rel_change = abs(x2 - x1) / max(x1, 1e-30)
            assert rel_change < 0.01, \
                f"y_N={y_N}: x jumped from {x1} to {x2} (rel={rel_change})"

    def test_monotonically_increasing(self):
        """solve_x(y_N) must be strictly increasing for y_N > 0."""
        y_values = [1e-10, 1e-5, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1e6]
        x_prev = 0.0
        for y_N in y_values:
            x = solve_x(y_N)
            assert x > x_prev, f"Not increasing at y_N={y_N}: x={x} <= {x_prev}"
            x_prev = x


class TestGFDvsMONDFieldEquations:
    """GFD and MOND have different field equations that diverge
    in the transition regime but agree in the limits."""

    def test_transition_point_x_equals_1(self):
        """At x=1: GFD gives y_N=0.5, MOND gives y_N=1/sqrt(2)~0.707."""
        # GFD: x^2/(1+x) at x=1 = 1/2 = 0.5
        gfd_y = 1.0 * 1.0 / (1.0 + 1.0)
        assert gfd_y == pytest.approx(0.5, abs=1e-15)

        # MOND: x^2/sqrt(1+x^2) at x=1 = 1/sqrt(2) ~ 0.7071
        mond_y = 1.0 * 1.0 / math.sqrt(1.0 + 1.0 * 1.0)
        assert mond_y == pytest.approx(1.0 / math.sqrt(2), abs=1e-15)

        # GFD enhances more: same x requires less y_N (less Newtonian mass)
        assert gfd_y < mond_y

    def test_gfd_always_enhances_more_than_mond(self):
        """For any given y_N, GFD produces a larger x than MOND.

        This means GFD predicts higher velocities, which is why the
        blue GFD curve always sits above the green MOND curve."""
        y_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
        for y_N in y_values:
            x_gfd = solve_x(y_N)
            x_mond = mond_solve_x(y_N)
            assert x_gfd >= x_mond, \
                f"y_N={y_N}: GFD x={x_gfd} < MOND x={x_mond}"

    def test_both_converge_in_deep_field_limit(self):
        """For y_N << 1, both give x ~ sqrt(y_N)."""
        for y_N in [1e-8, 1e-6, 1e-4]:
            x_gfd = solve_x(y_N)
            x_mond = mond_solve_x(y_N)
            expected = math.sqrt(y_N)
            assert x_gfd == pytest.approx(expected, rel=0.05)
            assert x_mond == pytest.approx(expected, rel=0.05)

    def test_btfr_from_field_equation(self):
        """In deep field: v^4 = G*M*a0 (Baryonic Tully-Fisher Relation).

        At large radius where g << a0, x ~ sqrt(y_N), so:
        g ~ a0*sqrt(g_N/a0) = sqrt(a0*g_N) = sqrt(a0*GM/r^2)
        v^2 = g*r = sqrt(a0*GM*r^2/r^2) = sqrt(a0*GM)
        v^4 = a0*GM   (the BTFR)
        """
        M = 5.0e10  # solar masses -- moderate galaxy
        r = 100.0   # kpc -- well into deep field
        v = dtg_velocity(r, M)

        # BTFR prediction
        v_btfr = ((G * A0) ** 0.25) * ((M * M_SUN) ** 0.25) / 1000.0
        # At r=100 kpc for a moderate galaxy, should be close
        assert v == pytest.approx(v_btfr, rel=0.05), \
            f"v_GFD={v:.2f}, v_BTFR={v_btfr:.2f}"
