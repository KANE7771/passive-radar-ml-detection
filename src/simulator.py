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


# ============================================================
# 1. Generate Reference Signal
# ============================================================

def generate_reference_signal(
        fs,
        duration,
        bandwidth,
        seed):

    """
    Generate a reproducible band-limited
    complex baseband reference signal.
    """

    rng = np.random.default_rng(seed)

    n_samples = int(
        round(fs * duration)
    )

    # Generate a random complex spectrum
    spectrum = (
        rng.normal(size=n_samples)
        +
        1j * rng.normal(size=n_samples)
    )

    frequencies = np.fft.fftfreq(
        n_samples,
        d=1.0 / fs
    )

    # Keep only frequencies inside selected bandwidth
    spectrum[
        np.abs(frequencies) > bandwidth / 2
    ] = 0.0

    # Convert to time domain
    reference = np.fft.ifft(
        spectrum
    )

    # Normalise average power to 1
    power = np.mean(
        np.abs(reference) ** 2
    )

    reference = (
        reference / np.sqrt(power)
    )

    return reference


# ============================================================
# 2. Calculate Bistatic Geometry
# ============================================================

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


    # Tx -> Target
    d_tx_target = np.linalg.norm(
        target - tx
    )


    # Target -> Rx
    d_target_rx = np.linalg.norm(
        target - rx
    )


    # Tx -> Rx direct path
    d_tx_rx = np.linalg.norm(
        rx - tx
    )


    # --------------------------------------------------------
    # Bistatic excess range
    #
    # Rb =
    # |Tx -> Target|
    # +
    # |Target -> Rx|
    # -
    # |Tx -> Rx|
    # --------------------------------------------------------

    bistatic_excess_range = (
        d_tx_target
        +
        d_target_rx
        -
        d_tx_rx
    )


    # --------------------------------------------------------
    # Distance -> Time Delay
    #
    # tau = Rb / c
    # --------------------------------------------------------

    bistatic_delay_seconds = (
        bistatic_excess_range
        /
        C
    )


    # --------------------------------------------------------
    # Seconds -> Samples
    # --------------------------------------------------------

    delay_samples = (
        bistatic_delay_seconds
        *
        fs
    )


    # ========================================================
    # Doppler
    # ========================================================

    wavelength = (
        C
        /
        carrier_frequency
    )


    # Direction from Tx to Target
    u_tx = (
        target - tx
    ) / d_tx_target


    # Direction from Rx to Target
    u_rx = (
        target - rx
    ) / d_target_rx


    # Rate of change of the bistatic path
    total_path_rate = np.dot(
        u_tx + u_rx,
        velocity
    )


    # Bistatic Doppler
    doppler_hz = (
        -total_path_rate
        /
        wavelength
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


# ============================================================
# 3. Fractional Delay
# ============================================================

def apply_fractional_delay(
        signal,
        delay_samples):

    """
    Delay signal by a non-integer number of samples.
    """

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


    return (
        delayed_real
        +
        1j * delayed_imag
    )


# ============================================================
# 4. Apply Doppler
# ============================================================

def apply_doppler(
        signal,
        doppler_hz,
        fs):

    time = (
        np.arange(
            len(signal)
        )
        /
        fs
    )


    phase_rotation = np.exp(
        1j
        *
        2
        *
        np.pi
        *
        doppler_hz
        *
        time
    )


    return (
        signal
        *
        phase_rotation
    )


# ============================================================
# 5. Generate One Receiver Channel
# ============================================================

def generate_receiver_channels(
        reference,
        rx_position,
        receiver_index):


    # Calculate physical truth first
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

        truth[
            "delay_samples"
        ]
    )


    # Apply geometry-derived Doppler
    target_echo = apply_doppler(

        target_echo,

        truth[
            "doppler_hz"
        ],

        FS
    )


    # Generate reproducible receiver noise
    rng = np.random.default_rng(
        SEED
        +
        1000
        +
        receiver_index
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


    # Simplified surveillance channel
    surveillance = (

        DIRECT_PATH_AMPLITUDE
        *
        reference

        +

        TARGET_AMPLITUDE
        *
        target_echo

        +

        noise
    )


    return {

        "reference":
            reference.copy(),

        "surveillance":
            surveillance,

        "target_echo":
            target_echo,

        "noise":
            noise,

        "truth":
            truth,
    }


# ============================================================
# 6. Run Baseline Simulation
# ============================================================

def run_baseline_simulation():

    reference = generate_reference_signal(

        FS,

        DURATION,

        BANDWIDTH,

        SEED
    )


    receiver_data = {}


    for (
        receiver_index,
        (rx_name, rx_position)
    ) in enumerate(
        RX_POSITIONS.items()
    ):


        receiver_data[
            rx_name
        ] = generate_receiver_channels(

            reference,

            rx_position,

            receiver_index
        )


    # --------------------------------------------------------
    # Save Ground Truth
    # --------------------------------------------------------

    truth_record = {

        "seed":
            SEED,

        "sampling_frequency_hz":
            FS,

        "carrier_frequency_hz":
            CARRIER_FREQUENCY,

        "bandwidth_hz":
            BANDWIDTH,

        "tx_position_m":
            list(TX_POSITION),

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

        "receivers": {

            name:
                data["truth"]

            for name, data
            in receiver_data.items()
        },
    }


    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )


    results_dir = (
        project_root
        /
        "results"
    )


    results_dir.mkdir(
        exist_ok=True
    )


    truth_path = (
        results_dir
        /
        "truth.json"
    )


    with truth_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            truth_record,
            file,
            indent=4
        )


    return (
        receiver_data,
        truth_record,
        truth_path
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    (
        receiver_data,
        truth_record,
        truth_path
    ) = run_baseline_simulation()


    print(
        "=== Geometry-first Passive Radar Simulation ==="
    )


    print(
        "Tx:",
        TX_POSITION
    )


    print(
        "Target:",
        TARGET_INITIAL_POSITION
    )


    print(
        "Velocity:",
        TARGET_VELOCITY
    )


    print()


    for (
        rx_name,
        rx_position
    ) in RX_POSITIONS.items():


        truth = (
            truth_record[
                "receivers"
            ][
                rx_name
            ]
        )


        print(
            f"{rx_name}: {rx_position}"
        )


        print(
            "Tx -> Target:",
            round(
                truth[
                    "tx_to_target_m"
                ],
                3
            ),
            "m"
        )


        print(
            "Target -> Rx:",
            round(
                truth[
                    "target_to_rx_m"
                ],
                3
            ),
            "m"
        )


        print(
            "Tx -> Rx:",
            round(
                truth[
                    "tx_to_rx_m"
                ],
                3
            ),
            "m"
        )


        print(
            "Bistatic excess range:",
            round(
                truth[
                    "bistatic_excess_range_m"
                ],
                3
            ),
            "m"
        )


        print(
            "Delay:",
            round(
                truth[
                    "bistatic_delay_s"
                ]
                *
                1e6,
                3
            ),
            "us"
        )


        print(
            "Delay samples:",
            round(
                truth[
                    "delay_samples"
                ],
                3
            )
        )


        print(
            "Doppler:",
            round(
                truth[
                    "doppler_hz"
                ],
                3
            ),
            "Hz"
        )


        print()


    print(
        "Truth saved to:",
        truth_path
    )