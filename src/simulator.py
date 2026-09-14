import json
from pathlib import Path

import numpy as np

from configs.baseline_config import (
    SEED,
    C,
    FS,
    DURATION,
    CARRIER_FREQUENCY,
    BANDWIDTH,
    TX_POSITION,
    RX_POSITIONS,
    TARGET_INITIAL_POSITION,
    TARGET_VELOCITY,
    DIRECT_PATH_AMPLITUDE,
    TARGET_AMPLITUDE,
    NOISE_STD,
)


# =========================================================
# 1. Generate reference signal
# =========================================================

def generate_reference_signal(
        fs,
        duration,
        bandwidth,
        seed):

    rng = np.random.default_rng(seed)

    n_samples = int(
        round(fs * duration)
    )

    spectrum = (
        rng.normal(size=n_samples)
        +
        1j * rng.normal(size=n_samples)
    )

    frequencies = np.fft.fftfreq(
        n_samples,
        d=1.0 / fs
    )

    spectrum[
        np.abs(frequencies) > bandwidth / 2.0
    ] = 0.0

    reference = np.fft.ifft(
        spectrum
    )

    power = np.mean(
        np.abs(reference) ** 2
    )

    reference = (
        reference / np.sqrt(power)
    )

    return reference
# =========================================================
# 2. Calculate bistatic geometry
# =========================================================

def calculate_bistatic_truth(
        tx_position,
        rx_position,
        target_position,
        target_velocity,
        carrier_frequency,
        fs):

    tx = np.asarray(
        tx_position,
        dtype=float
    )

    rx = np.asarray(
        rx_position,
        dtype=float
    )

    target = np.asarray(
        target_position,
        dtype=float
    )

    velocity = np.asarray(
        target_velocity,
        dtype=float
    )


    # Tx → Target
    d_tx_target = np.linalg.norm(
        target - tx
    )


    # Target → Rx
    d_target_rx = np.linalg.norm(
        target - rx
    )


    # Tx → Rx direct path
    d_tx_rx = np.linalg.norm(
        rx - tx
    )


    # Bistatic excess range
    bistatic_excess_range = (
        d_tx_target
        +
        d_target_rx
        -
        d_tx_rx
    )


    # Convert range difference to time delay
    bistatic_delay_seconds = (
        bistatic_excess_range / C
    )


    # Convert seconds to samples
    delay_samples = (
        bistatic_delay_seconds * fs
    )


    # =====================================================
    # Doppler
    # =====================================================

    wavelength = (
        C / carrier_frequency
    )


    # Unit vector from Tx to Target
    u_tx = (
        target - tx
    ) / d_tx_target


    # Unit vector from Rx to Target
    u_rx = (
        target - rx
    ) / d_target_rx


    total_path_rate = np.dot(
        u_tx + u_rx,
        velocity
    )


    doppler_hz = (
        -total_path_rate / wavelength
    )


    return {
        "tx_to_target_m":
            float(d_tx_target),

        "target_to_rx_m":
            float(d_target_rx),

        "tx_to_rx_m":
            float(d_tx_rx),

        "bistatic_excess_range_m":
            float(bistatic_excess_range),

        "bistatic_delay_s":
            float(bistatic_delay_seconds),

        "delay_samples":
            float(delay_samples),

        "doppler_hz":
            float(doppler_hz),
    }
# =========================================================
# 3. Apply fractional delay
# =========================================================

def apply_fractional_delay(
        signal,
        delay_samples):

    n = np.arange(
        len(signal),
        dtype=float
    )

    source_index = (
        n - delay_samples
    )


    delayed_real = np.interp(
        source_index,
        n,
        signal.real,
        left=0.0,
        right=0.0
    )


    delayed_imag = np.interp(
        source_index,
        n,
        signal.imag,
        left=0.0,
        right=0.0
    )


    delayed_signal = (
        delayed_real
        +
        1j * delayed_imag
    )

    return delayed_signal

# =========================================================
# 4. Apply Doppler
# =========================================================

def apply_doppler(
        signal,
        doppler_hz,
        fs):

    time = (
        np.arange(
            len(signal),
            dtype=float
        )
        / fs
    )


    phase_rotation = np.exp(
        1j
        * 2.0
        * np.pi
        * doppler_hz
        * time
    )


    return (
        signal * phase_rotation
    )
# =========================================================
# 5. Generate one receiver channel
# =========================================================

def generate_receiver_channels(
        reference,
        rx_position,
        receiver_index):


    truth = calculate_bistatic_truth(
        TX_POSITION,
        rx_position,
        TARGET_INITIAL_POSITION,
        TARGET_VELOCITY,
        CARRIER_FREQUENCY,
        FS
    )


    # Apply geometry-derived delay
    target_echo = apply_fractional_delay(
        reference,
        truth["delay_samples"]
    )


    # Apply geometry-derived Doppler
    target_echo = apply_doppler(
        target_echo,
        truth["doppler_hz"],
        FS
    )


    # Reproducible noise
    rng = np.random.default_rng(
        SEED + 1000 + receiver_index
    )


    noise = (
        rng.normal(
            scale=NOISE_STD,
            size=len(reference)
        )
        +
        1j * rng.normal(
            scale=NOISE_STD,
            size=len(reference)
        )
    )


    surveillance = (
        DIRECT_PATH_AMPLITUDE
        * reference

        +

        TARGET_AMPLITUDE
        * target_echo

        +

        noise
    )


    return (
        surveillance,
        target_echo,
        noise,
        truth
    )
# =========================================================
# 6. Run baseline simulation
# =========================================================

if __name__ == "__main__":


    reference = generate_reference_signal(
        FS,
        DURATION,
        BANDWIDTH,
        SEED
    )


    all_truth = {}


    print(
        "=== Geometry-first Passive Radar Simulation ==="
    )

    print(
        "Tx position:",
        TX_POSITION
    )

    print(
        "Target position:",
        TARGET_INITIAL_POSITION
    )

    print(
        "Target velocity:",
        TARGET_VELOCITY
    )

    print()


    for index, (
        rx_name,
        rx_position
    ) in enumerate(
        RX_POSITIONS.items()
    ):


        surveillance, \
        target_echo, \
        noise, \
        truth = generate_receiver_channels(
            reference,
            rx_position,
            index
        )


        all_truth[rx_name] = truth


        print(
            rx_name,
            "position:",
            rx_position
        )


        print(
            "  Bistatic excess range:",
            round(
                truth[
                    "bistatic_excess_range_m"
                ],
                3
            ),
            "m"
        )


        print(
            "  Delay:",
            round(
                truth[
                    "bistatic_delay_s"
                ] * 1e6,
                3
            ),
            "us"
        )


        print(
            "  Delay samples:",
            round(
                truth[
                    "delay_samples"
                ],
                3
            )
        )


        print(
            "  Doppler:",
            round(
                truth[
                    "doppler_hz"
                ],
                3
            ),
            "Hz"
        )


        print()


    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )


    results_dir = (
        project_root
        / "results"
    )


    results_dir.mkdir(
        exist_ok=True
    )


    truth_record = {
        "seed":
            SEED,

        "tx_position_m":
            list(
                TX_POSITION
            ),

        "rx_positions_m": {
            name:
            list(position)

            for name, position
            in RX_POSITIONS.items()
        },

        "target_position_m":
            list(
                TARGET_INITIAL_POSITION
            ),

        "target_velocity_mps":
            list(
                TARGET_VELOCITY
            ),

        "receivers":
            all_truth
    }


    with open(
        results_dir / "truth.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            truth_record,
            file,
            indent=4
        )
        python - m
        src.simulator