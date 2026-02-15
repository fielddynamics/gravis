"""
a0 Derivation Service stub.

Will provide an interactive derivation of the characteristic
acceleration scale a0 = k^2 * G * m_e / r_e^2 from fundamental
constants, with sliders showing sensitivity to k.

Status: coming_soon
"""

from physics.services import GravisService


class A0DerivationService(GravisService):

    id = "a0_derivation"
    name = "a0 Derivation"
    description = "Deriving a0 from fundamental constants and k=4 topology"
    category = "quantum_nuclear"
    status = "coming_soon"
    route = "/a0-derivation"

    def validate(self, config):
        raise NotImplementedError(
            "a0 derivation service is not yet implemented")

    def compute(self, config):
        raise NotImplementedError(
            "a0 derivation service is not yet implemented")
