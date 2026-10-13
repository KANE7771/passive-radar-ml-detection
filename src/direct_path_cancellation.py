import numpy as np


def cancel_direct_path(
        reference,
        surveillance):

    """
    Estimate the strongest direct-path component
    that is correlated with the reference signal,
    then subtract it from the surveillance signal.

    This is a simple baseline direct-path cancellation
    method, not a full clutter-cancellation algorithm.
    """

    reference = np.asarray(
        reference,
        dtype=complex
    )

    surveillance = np.asarray(
        surveillance,
        dtype=complex
    )


    # --------------------------------------------------------
    # Estimate complex direct-path coefficient
    #
    # alpha tells us how much of the reference signal
    # is present in the surveillance channel.
    # --------------------------------------------------------

    alpha = (
        np.vdot(
            reference,
            surveillance
        )
        /
        np.vdot(
            reference,
            reference
        )
    )


    # --------------------------------------------------------
    # Estimate direct-path signal
    # --------------------------------------------------------

    direct_path_estimate = (
        alpha
        *
        reference
    )


    # --------------------------------------------------------
    # Remove estimated direct path
    # --------------------------------------------------------

    cleaned = (
        surveillance
        -
        direct_path_estimate
    )


    return (
        cleaned,
        alpha
    )