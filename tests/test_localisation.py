import numpy as np

from configs.baseline_config import (
    TX_POSITION,
    RX_POSITIONS,
    TARGET_INITIAL_POSITION,
)

from src.simulator import (
    run_baseline_simulation,
)

from src.localisation import (
    localise_from_bistatic_ranges,
)


def test_localisation_with_ground_truth_ranges():

    (
        receiver_data,
        _,
        _
    ) = run_baseline_simulation()


    # --------------------------------------------------------
    # Use PERFECT bistatic ranges
    # --------------------------------------------------------

    measured_ranges = {

        name:
            data[
                "truth"
            ][
                "bistatic_excess_range_m"
            ]

        for (
            name,
            data
        )
        in receiver_data.items()

    }


    # --------------------------------------------------------
    # Estimate target location
    # --------------------------------------------------------

    (
        estimated_position,
        result
    ) = localise_from_bistatic_ranges(

        measured_ranges,

        TX_POSITION,

        RX_POSITIONS
    )


    true_position = np.asarray(
        TARGET_INITIAL_POSITION,
        dtype=float
    )


    position_error = np.linalg.norm(

        estimated_position

        -

        true_position
    )


    print(
        "\n=== Ground Truth Localisation Test ==="
    )

    print(
        "True position:",
        true_position
    )

    print(
        "Estimated position:",
        estimated_position
    )

    print(
        "Position error:",
        position_error,
        "m"
    )


    assert (
        position_error
        <
        1e-3
    )