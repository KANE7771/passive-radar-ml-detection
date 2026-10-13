import json
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from configs.baseline_config import (
    TX_POSITION,
    RX_POSITIONS,
    TARGET_INITIAL_POSITION,
)

from src.localisation import (
    localise_from_bistatic_ranges,
)


def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )


    measurement_path = (
        project_root
        /
        "results"
        /
        "measurements.json"
    )


    # --------------------------------------------------------
    # Load estimated measurements
    # --------------------------------------------------------

    with measurement_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        measurements = json.load(
            file
        )


    measured_ranges = {

        receiver_name:
            values[
                "estimated_bistatic_excess_range_m"
            ]

        for (
            receiver_name,
            values
        )
        in measurements.items()

    }


    # --------------------------------------------------------
    # Run localisation
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


    print()
    print(
        "================================="
    )

    print(
        "2D PASSIVE RADAR LOCALISATION"
    )

    print(
        "================================="
    )


    print(
        "True target position:",
        true_position
    )


    print(
        "Estimated target position:",
        estimated_position
    )


    print(
        "Position error:",
        position_error,
        "m"
    )


    # ========================================================
    # Plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )


    # Tx
    ax.scatter(
        TX_POSITION[0],
        TX_POSITION[1],
        marker="^",
        s=150,
        label="Tx"
    )


    # Receivers
    for (
        name,
        position
    ) in RX_POSITIONS.items():

        ax.scatter(
            position[0],
            position[1],
            marker="s",
            s=100,
            label=name
        )


    # True Target
    ax.scatter(
        true_position[0],
        true_position[1],
        marker="*",
        s=220,
        label="True Target"
    )


    # Estimated Target
    ax.scatter(
        estimated_position[0],
        estimated_position[1],
        marker="o",
        s=150,
        facecolors="none",
        label="Estimated Target"
    )


    # Error connection line
    ax.plot(
        [
            true_position[0],
            estimated_position[0]
        ],
        [
            true_position[1],
            estimated_position[1]
        ],
        linestyle="--"
    )


    ax.set_title(
        (
            "Passive Radar 2D Localisation\n"
            f"Position error = "
            f"{position_error:.2f} m"
        )
    )


    ax.set_xlabel(
        "x position (m)"
    )


    ax.set_ylabel(
        "y position (m)"
    )


    ax.grid(
        True
    )


    ax.axis(
        "equal"
    )


    ax.legend()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = (
        project_root
        /
        "results"
        /
        "figures"
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    output_path = (
        output_dir
        /
        "localisation_result.png"
    )


    fig.tight_layout()


    fig.savefig(
        output_path,
        dpi=200
    )


    plt.show()


    print(
        "Figure saved to:",
        output_path
    )


if __name__ == "__main__":

    main()