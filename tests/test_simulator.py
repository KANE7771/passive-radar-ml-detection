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


# =========================================================
# Test 1: Reproducibility
# =========================================================

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


    are_identical = np.allclose(
        signal_1,
        signal_2
    )


    print(
        "\nReproducibility Test"
    )

    print(
        "Seed:",
        SEED
    )

    print(
        "Signals identical:",
        are_identical
    )


    assert are_identical


# =========================================================
# Test 2: Geometry-derived delay
# =========================================================

def test_geometry_delay_consistency():

    print(
        "\nGeometry-derived Delay Test"
    )


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


        reconstructed_range = (
            C
            *
            truth[
                "bistatic_delay_s"
            ]
        )


        print(
            rx_name,
            "Range:",
            round(
                truth[
                    "bistatic_excess_range_m"
                ],
                3
            ),
            "m"
        )


        assert (
            truth[
                "bistatic_excess_range_m"
            ]
            >= 0
        )


        assert np.isclose(
            reconstructed_range,
            truth[
                "bistatic_excess_range_m"
            ]
        )


# =========================================================
# Test 3: Zero velocity → zero Doppler
# =========================================================

def test_stationary_target_has_zero_doppler():

    zero_velocity = (
        0.0,
        0.0
    )


    for rx_position \
        in RX_POSITIONS.values():


        truth = calculate_bistatic_truth(
            TX_POSITION,
            rx_position,
            TARGET_INITIAL_POSITION,
            zero_velocity,
            CARRIER_FREQUENCY,
            FS
        )


        assert np.isclose(
            truth[
                "doppler_hz"
            ],
            0.0
        )