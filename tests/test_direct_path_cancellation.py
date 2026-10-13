import numpy as np

from src.simulator import (
    run_baseline_simulation,
)

from src.direct_path_cancellation import (
    cancel_direct_path,
)


def test_direct_path_cancellation():

    (
        receiver_data,
        _,
        _
    ) = run_baseline_simulation()


    data = receiver_data["R1"]


    reference = data["reference"]

    surveillance = data["surveillance"]


    # --------------------------------------------------------
    # Apply cancellation
    # --------------------------------------------------------

    cleaned, alpha = cancel_direct_path(
        reference,
        surveillance
    )


    # --------------------------------------------------------
    # Measure correlation with reference
    # before cancellation
    # --------------------------------------------------------

    correlation_before = abs(
        np.vdot(
            reference,
            surveillance
        )
    )


    # --------------------------------------------------------
    # Measure correlation after cancellation
    # --------------------------------------------------------

    correlation_after = abs(
        np.vdot(
            reference,
            cleaned
        )
    )


    print(
        "\n=== Direct Path Cancellation Test ==="
    )

    print(
        "Correlation before:",
        correlation_before
    )

    print(
        "Correlation after:",
        correlation_after
    )

    print(
        "Estimated alpha:",
        alpha
    )


    # --------------------------------------------------------
    # Cancellation should reduce direct-path correlation
    # --------------------------------------------------------

    assert (
        correlation_after
        <
        correlation_before
    )