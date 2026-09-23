from pathlib import Path

import matplotlib.pyplot as plt

from configs.baseline_config import (
    TX_POSITION,
    RX_POSITIONS,
    TARGET_INITIAL_POSITION,
    TARGET_VELOCITY,
    CARRIER_FREQUENCY,
    FS,
)

from src.simulator import (
    calculate_bistatic_truth,
)


def plot_geometry():

    tx_x, tx_y = TX_POSITION

    target_x, target_y = (
        TARGET_INITIAL_POSITION
    )


    fig, ax = plt.subplots(
        figsize=(9, 7)
    )


    # Tx
    ax.scatter(
        tx_x,
        tx_y,
        marker="^",
        s=140,
        label="Tx"
    )


    # Target
    ax.scatter(
        target_x,
        target_y,
        marker="*",
        s=220,
        label="Target"
    )


    ax.text(
        tx_x,
        tx_y,
        " Tx"
    )


    ax.text(
        target_x,
        target_y,
        " Target"
    )


    for (
        rx_name,
        rx_position
    ) in RX_POSITIONS.items():


        rx_x, rx_y = (
            rx_position
        )


        ax.scatter(
            rx_x,
            rx_y,
            marker="s",
            s=100,
            label=rx_name
        )


        ax.text(
            rx_x,
            rx_y,
            f" {rx_name}"
        )


        # Direct path
        Tx_Rx_x = [
            tx_x,
            rx_x
        ]

        Tx_Rx_y = [
            tx_y,
            rx_y
        ]


        ax.plot(
            Tx_Rx_x,
            Tx_Rx_y,
            linestyle="--"
        )


        # Target reflected path
        ax.plot(

            [
                tx_x,
                target_x,
                rx_x
            ],

            [
                tx_y,
                target_y,
                rx_y
            ]
        )


        truth = calculate_bistatic_truth(

            TX_POSITION,

            rx_position,

            TARGET_INITIAL_POSITION,

            TARGET_VELOCITY,

            CARRIER_FREQUENCY,

            FS
        )


        mid_x = (
            target_x
            +
            rx_x
        ) / 2


        mid_y = (
            target_y
            +
            rx_y
        ) / 2


        ax.text(

            mid_x,

            mid_y,

            (
                f"{rx_name}\n"
                f"Delay={truth['delay_samples']:.2f} samples\n"
                f"Doppler={truth['doppler_hz']:.1f} Hz"
            ),

            fontsize=8
        )


    ax.set_title(
        "Baseline Passive Radar Geometry"
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


    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )


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

        "geometry.png"

    )


    fig.tight_layout()


    fig.savefig(
        output_path,
        dpi=200
    )


    plt.show()


    print(
        "Saved:",
        output_path
    )


if __name__ == "__main__":

    plot_geometry()