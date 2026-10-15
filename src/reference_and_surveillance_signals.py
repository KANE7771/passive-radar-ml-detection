
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.transmit_signal import generate_transmitted_baseband


# =====================================================
# 1. Simulation Parameters
# =====================================================

C = 299_792_458.0

FS = 2_000_000
DURATION = 0.02
BANDWIDTH = 500_000
SEED = 42

# Geometry configuration (metres)
TX_POSITION = np.array([0.0, 0.0])
RX_POSITION = np.array([1000.0, 0.0])
TARGET_POSITION = np.array([600.0, 800.0])

# Stationary target
TARGET_VELOCITY = np.array([0.0, 0.0])

# Signal amplitudes
REFERENCE_AMPLITUDE = 1.0
DIRECT_PATH_AMPLITUDE = 1.0
TARGET_ECHO_AMPLITUDE = 0.1
NOISE_STD = 0.05


# =====================================================
# 2. Fractional Delay Function
# =====================================================

def fractional_delay(signal, delay_samples):
    """
    Apply a positive fractional sample delay
    using linear interpolation.

    Out-of-range samples are zero.
    """

    sample_index = np.arange(len(signal))
    source_index = sample_index - delay_samples

    real_part = np.interp(
        source_index,
        sample_index,
        signal.real,
        left=0.0,
        right=0.0,
    )

    imag_part = np.interp(
        source_index,
        sample_index,
        signal.imag,
        left=0.0,
        right=0.0,
    )

    return real_part + 1j * imag_part


# =====================================================
# 3. Signal Propagation
# =====================================================

def propagate_signal(
    transmitted_baseband,
    path_distance,
    fs,
    carrier_frequency,
):
    """
    Simulate propagation using a complex-baseband
    equivalent of the transmitted RF signal.

    Includes:
      - Propagation delay
      - Carrier phase shift

    Path attenuation is applied separately.
    """

    propagation_delay = path_distance / C

    delay_samples = propagation_delay * fs

    delayed_signal = fractional_delay(
        transmitted_baseband,
        delay_samples,
    )

    carrier_phase = np.exp(
        -1j
        * 2
        * np.pi
        * carrier_frequency
        * propagation_delay
    )

    return delayed_signal * carrier_phase


# =====================================================
# 4. Calculate Bistatic Geometry
# =====================================================

def calculate_geometry():

    # Direct path: Tx -> Rx
    direct_distance = np.linalg.norm(
        RX_POSITION - TX_POSITION
    )

    # Reflected path: Tx -> Target -> Rx
    tx_target_distance = np.linalg.norm(
        TARGET_POSITION - TX_POSITION
    )

    target_rx_distance = np.linalg.norm(
        RX_POSITION - TARGET_POSITION
    )

    reflected_distance = (
        tx_target_distance + target_rx_distance
    )

    # Bistatic excess range
    bistatic_excess_range = (
        reflected_distance - direct_distance
    )

    # Propagation delays
    direct_delay = direct_distance / C

    reflected_delay = reflected_distance / C

    bistatic_delay = reflected_delay - direct_delay

    return {
        "direct_distance": direct_distance,
        "reflected_distance": reflected_distance,
        "bistatic_excess_range": bistatic_excess_range,
        "direct_delay": direct_delay,
        "reflected_delay": reflected_delay,
        "bistatic_delay": bistatic_delay,
    }


# =====================================================
# 5. Generate Reference Signal
# =====================================================

def generate_reference_signal(
    transmitted_baseband,
    carrier_frequency,
    geometry,
):
    """
    Reference signal is obtained from
    the transmitted signal through the direct path.
    """

    reference = propagate_signal(
        transmitted_baseband,
        geometry["direct_distance"],
        FS,
        carrier_frequency,
    )

    reference *= REFERENCE_AMPLITUDE

    return reference


# =====================================================
# 6. Generate Surveillance Signal
# =====================================================

def generate_surveillance_signal(
    transmitted_baseband,
    carrier_frequency,
    geometry,
):
    """
    Surveillance signal consists of:
      1. Direct-path component
      2. Stationary target echo
      3. Complex Gaussian noise
    """

    # 6.1 Direct-path component
    direct_path = propagate_signal(
        transmitted_baseband,
        geometry["direct_distance"],
        FS,
        carrier_frequency,
    )

    direct_path *= DIRECT_PATH_AMPLITUDE

    # 6.2 Reflected target echo
    target_echo = propagate_signal(
        transmitted_baseband,
        geometry["reflected_distance"],
        FS,
        carrier_frequency,
    )

    target_echo *= TARGET_ECHO_AMPLITUDE

    # Target is stationary:
    # velocity = (0, 0) m/s
    # Doppler shift = 0 Hz
    # No additional Doppler modulation is applied.

    # 6.3 Complex Gaussian noise
    rng = np.random.default_rng(SEED + 1000)

    noise = NOISE_STD * (
        rng.standard_normal(len(transmitted_baseband))
        + 1j * rng.standard_normal(len(transmitted_baseband))
    )

    # 6.4 Full surveillance signal
    surveillance = direct_path + target_echo + noise

    return surveillance, direct_path, target_echo, noise


# =====================================================
# 7. Run Reference and Surveillance Simulation
# =====================================================

def run_reference_and_surveillance_simulation():

    # Step 1 + Step 2:
    # Random signal -> transmitted signal
    transmitted_baseband, fc = generate_transmitted_baseband(
        fs=FS,
        duration=DURATION,
        bandwidth=BANDWIDTH,
        seed=SEED,
    )

    # Calculate geometry
    geometry = calculate_geometry()

    # Derive reference from transmitted signal
    reference = generate_reference_signal(
        transmitted_baseband,
        fc,
        geometry,
    )

    # Derive surveillance from transmitted signal
    surveillance, direct_path, target_echo, noise = (
        generate_surveillance_signal(
            transmitted_baseband,
            fc,
            geometry,
        )
    )

    return {
        "transmitted_baseband": transmitted_baseband,
        "carrier_frequency": fc,
        "reference": reference,
        "surveillance": surveillance,
        "direct_path": direct_path,
        "target_echo": target_echo,
        "noise": noise,
        "geometry": geometry,
        "target_velocity": TARGET_VELOCITY.copy(),
        "doppler_hz": 0.0,
    }


# =====================================================
# 8. Plot Signals and Display Results
# =====================================================

def main():

    data = run_reference_and_surveillance_simulation()

    reference = data["reference"]
    surveillance = data["surveillance"]
    target_echo = data["target_echo"]
    direct_path = data["direct_path"]
    geometry = data["geometry"]

    n_view = 150
    sample_index = np.arange(n_view)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    # Figure 1: Reference signal
    axes[0].plot(
        sample_index,
        reference[:n_view].real,
        label="Reference",
    )

    axes[0].set_title(
        "Reference Signal (Direct-Path Observation)"
    )

    axes[0].legend()

    # Figure 2: Stationary target echo
    axes[1].plot(
        sample_index,
        target_echo[:n_view].real,
        label="Target Echo",
    )

    axes[1].set_title(
        "Stationary Target Echo (Reflected Component)"
    )

    axes[1].legend()

    # Figure 3: Surveillance signal
    axes[2].plot(
        sample_index,
        surveillance[:n_view].real,
        label="Surveillance Signal",
    )

    axes[2].plot(
        sample_index,
        direct_path[:n_view].real,
        label="Direct-Path Component",
        alpha=0.65,
    )

    axes[2].set_title(
        "Surveillance Signal = Direct Path + Echo + Noise"
    )

    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Real Amplitude")
        ax.grid(True)

    fig.tight_layout()

    # Save figure
    output_dir = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "figures"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir
        / "03_reference_and_surveillance_signals.png"
    )

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    # Display results
    print("\n====================================")
    print("REFERENCE & SURVEILLANCE SIMULATION")
    print("====================================")

    print(
        "Carrier Frequency:",
        data["carrier_frequency"] / 1e6,
        "MHz",
    )

    print("Target Velocity:", data["target_velocity"])
    print("Doppler Shift:", data["doppler_hz"], "Hz")

    print("\n--- Geometry ---")

    print(
        "Direct-Path Distance:",
        geometry["direct_distance"],
        "m",
    )

    print(
        "Reflected-Path Distance:",
        geometry["reflected_distance"],
        "m",
    )

    print(
        "Bistatic Excess Range:",
        geometry["bistatic_excess_range"],
        "m",
    )

    print("\n--- Propagation Delay ---")

    print(
        "Direct-Path Delay:",
        geometry["direct_delay"] * FS,
        "samples",
    )

    print(
        "Reflected-Path Delay:",
        geometry["reflected_delay"] * FS,
        "samples",
    )

    print(
        "Relative Bistatic Delay:",
        geometry["bistatic_delay"] * FS,
        "samples",
    )

    print("\nSimulation completed.")
    print("Figure saved to:", output_path)


if __name__ == "__main__":
    main()
