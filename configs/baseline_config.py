# Geometry-first baseline configuration
# Passive Radar Target Localisation

SEED = 42

# =========================================================
# Physical constant
# =========================================================

C = 299_792_458.0
# Speed of light, m/s


# =========================================================
# Signal settings
# =========================================================

FS = 2_000_000
# Sampling frequency = 2 MHz

DURATION = 0.02
# Signal duration = 20 ms

CARRIER_FREQUENCY = 650_000_000.0
# Carrier frequency = 650 MHz

BANDWIDTH = 500_000.0
# Baseband signal bandwidth = 500 kHz


# =========================================================
# Geometry
# =========================================================

TX_POSITION = (0.0, 0.0)

RX_POSITIONS = {
    "R1": (1000.0, 0.0),
    "R2": (0.0, 1200.0),
    "R3": (-800.0, 400.0),
}

TARGET_INITIAL_POSITION = (600.0, 800.0)

TARGET_VELOCITY = (60.0, -20.0)


# =========================================================
# Signal amplitudes
# =========================================================

DIRECT_PATH_AMPLITUDE = 1.0

TARGET_AMPLITUDE = 0.1

NOISE_STD = 0.05