import numpy as np

from scipy.optimize import (
    least_squares,
)


def calculate_predicted_bistatic_range(
        target_position,
        tx_position,
        rx_position):

    """
    Calculate predicted bistatic excess range
    for one receiver.
    """

    target = np.asarray(
        target_position,
        dtype=float
    )

    tx = np.asarray(
        tx_position,
        dtype=float
    )

    rx = np.asarray(
        rx_position,
        dtype=float
    )


    tx_target = np.linalg.norm(
        target - tx
    )

    target_rx = np.linalg.norm(
        target - rx
    )

    tx_rx = np.linalg.norm(
        tx - rx
    )


    return (
        tx_target
        +
        target_rx
        -
        tx_rx
    )


def localise_from_bistatic_ranges(
        measured_ranges,
        tx_position,
        rx_positions,
        initial_guess=None):

    """
    Estimate 2D target position using nonlinear
    least-squares fitting of bistatic ranges.
    """

    receiver_names = list(
        measured_ranges.keys()
    )


    if len(receiver_names) < 3:

        raise ValueError(
            "At least three bistatic measurements "
            "are required for this 2D baseline."
        )


    tx = np.asarray(
        tx_position,
        dtype=float
    )


    # --------------------------------------------------------
    # Initial guess
    #
    # Do NOT use the true target position.
    # Use the centroid of Tx and receivers.
    # --------------------------------------------------------

    if initial_guess is None:

        anchor_positions = [
            tx
        ]

        for name in receiver_names:

            anchor_positions.append(
                np.asarray(
                    rx_positions[
                        name
                    ],
                    dtype=float
                )
            )


        initial_guess = np.mean(
            np.vstack(
                anchor_positions
            ),
            axis=0
        )


    # --------------------------------------------------------
    # Residual function
    # --------------------------------------------------------

    def residuals(
            candidate_position):

        errors = []


        for name in receiver_names:

            predicted = (
                calculate_predicted_bistatic_range(
                    candidate_position,
                    tx_position,
                    rx_positions[
                        name
                    ]
                )
            )


            measured = (
                measured_ranges[
                    name
                ]
            )


            errors.append(
                predicted
                -
                measured
            )


        return np.asarray(
            errors
        )


    # --------------------------------------------------------
    # Nonlinear least squares
    # --------------------------------------------------------

    result = least_squares(
        residuals,
        initial_guess
    )


    estimated_position = (
        result.x
    )


    return (
        estimated_position,
        result
    )