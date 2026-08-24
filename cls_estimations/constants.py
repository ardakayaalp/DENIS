"""Physical constants and unit conversions used across the toolkit.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Defines the speed of light, atomic-mass-unit-to-kilogram and
elementary-charge values, and the cm^-1-to-Hz conversion used by the
Doppler and isotope-shift calculations.

Depends on: standard library and third-party packages only.
"""
C_LIGHT = 299792458.0            # speed of light [m/s]
AMU_TO_KG = 1.66053906660e-27    # atomic mass unit [kg]
E_CHARGE = 1.602176634e-19       # elementary charge [C]
CM_TO_HZ = C_LIGHT * 100.0       # cm^-1 to Hz
