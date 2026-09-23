# ============================================================
# Geometry-first Passive Radar Baseline Configuration
# ============================================================

# Random seed
SEED = 42


# ============================================================
# Physical Constant
# ============================================================

# Speed of light (m/s)
C = 299_792_458.0


# ============================================================
# Signal Settings
# ============================================================

# Sampling frequency = 2 MHz
FS = 2_000_000

# Signal duration = 20 ms
DURATION = 0.02

# Carrier frequency = 650 MHz
CARRIER_FREQUENCY = 650_000_000.0

# Baseband bandwidth = 500 kHz
BANDWIDTH = 500_000.0


# ============================================================
# Geometry
# All coordinates are in metres
# ============================================================

# Transmitter
TX_POSITION = (0.0, 0.0)

# Three receivers
RX_POSITIONS = {
    "R1": (1000.0, 0.0),
    "R2": (0.0, 1200.0),
    "R3": (-800.0, 400.0),
}


# ============================================================
# Target Ground Truth
# ============================================================

# Initial target position
TARGET_INITIAL_POSITION = (600.0, 800.0)

# Target velocity (vx, vy), m/s
TARGET_VELOCITY = (60.0, -20.0)


# ============================================================
# Simplified Signal Levels
# ============================================================

DIRECT_PATH_AMPLITUDE = 1.0

TARGET_AMPLITUDE = 0.1

NOISE_STD = 0.05