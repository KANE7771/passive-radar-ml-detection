import numpy as np

from configs.baseline_config import (
    C,
    FS,
    DURATION,
    BANDWIDTH,
    SEED,
    CARRIER_FREQUENCY,
    TX_POSITION,
    RX_POSITIONS,
    TARGET_INITIAL_POSITION,
    TARGET_VELOCITY,
)

from src.simulator import (
    generate_reference_signal,
    calculate_bistatic_truth,
)


# ============================================================
# Test 1 - Reproducibility
# ============================================================

def test_reference_reproducibility():

    signal_1 = generate_reference_signal(
        FS,
        DURATION,
        BANDWIDTH,
        SEED
    )


    signal_2 = generate_reference_signal(
        FS,
        DURATION,
        BANDWIDTH,
        SEED
    )


    assert np.allclose(
        signal_1,
        signal_2
    )


# ============================================================
# Test 2 - Range / Delay Consistency
# ============================================================

def test_geometry_delay_consistency():

    for (
        rx_name,
        rx_position
    ) in RX_POSITIONS.items():


        truth = calculate_bistatic_truth(

            TX_POSITION,

            rx_position,

            TARGET_INITIAL_POSITION,

            TARGET_VELOCITY,

            CARRIER_FREQUENCY,

            FS
        )


        calculated_range = (

            C

            *

            truth[
                "bistatic_delay_s"
            ]
        )


        print(
            rx_name,
            truth[
                "bistatic_excess_range_m"
            ],
            calculated_range
        )


        assert np.isclose(

            calculated_range,

            truth[
                "bistatic_excess_range_m"
            ]
        )


# ============================================================
# Test 3 - Stationary Target = Zero Doppler
# ============================================================

def test_stationary_target_has_zero_doppler():

    stationary_velocity = (
        0.0,
        0.0
    )


    for rx_position \
        in RX_POSITIONS.values():


        truth = calculate_bistatic_truth(

            TX_POSITION,

            rx_position,

            TARGET_INITIAL_POSITION,

            stationary_velocity,

            CARRIER_FREQUENCY,

            FS
        )


        assert np.isclose(

            truth[
                "doppler_hz"
            ],

            0.0
        )


# ============================================================
# Test 4 - Reverse Motion = Reverse Doppler
# ============================================================

def test_reverse_velocity_reverses_doppler():

    reverse_velocity = tuple(

        -value

        for value
        in TARGET_VELOCITY
    )


    for rx_position \
        in RX_POSITIONS.values():


        forward = calculate_bistatic_truth(

            TX_POSITION,

            rx_position,

            TARGET_INITIAL_POSITION,

            TARGET_VELOCITY,

            CARRIER_FREQUENCY,

            FS
        )


        reverse = calculate_bistatic_truth(

            TX_POSITION,

            rx_position,

            TARGET_INITIAL_POSITION,

            reverse_velocity,

            CARRIER_FREQUENCY,

            FS
        )


        assert np.isclose(

            forward[
                "doppler_hz"
            ],

            -reverse[
                "doppler_hz"
            ]
        )