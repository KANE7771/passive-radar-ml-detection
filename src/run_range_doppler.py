from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from configs.baseline_config import (
    FS,
)

from src.simulator import (
    run_baseline_simulation,
)

from src.ambiguity_func import (
    cross_ambiguity_fft,
    find_ambiguity_peak,
)


def main():

    (
        receiver_data,
        truth_record,
        _
    ) = run_baseline_simulation()


    data = receiver_data[
        "R1"
    ]


    reference = data[
        "reference"
    ]


    # First validation:
    # use clean target echo
    target_echo = data[
        "target_echo"
    ]


    truth = data[
        "truth"
    ]


    max_delay = (

        int(
            np.ceil(
                truth[
                    "delay_samples"
                ]
            )
        )

        +

        5
    )


    (
        power,
        delay_axis,
        doppler_axis
    ) = cross_ambiguity_fft(

        reference,

        target_echo,

        FS,

        max_delay
    )


    peak = find_ambiguity_peak(

        power,

        delay_axis,

        doppler_axis,

        -500,

        500
    )


    print(
        "True delay:",
        truth[
            "delay_samples"
        ]
    )


    print(
        "Estimated delay:",
        peak[
            "delay_samples"
        ]
    )


    print(
        "True Doppler:",
        truth[
            "doppler_hz"
        ]
    )


    print(
        "Estimated Doppler:",
        peak[
            "doppler_hz"
        ]
    )


    # Only display +/- 500 Hz
    mask = (

        (
            doppler_axis
            >=
            -500
        )

        &

        (
            doppler_axis
            <=
            500
        )
    )


    display_power = (
        power[
            :,
            mask
        ]
    )


    display_doppler = (
        doppler_axis[
            mask
        ]
    )


    power_db = (

        10

        *

        np.log10(

            display_power

            /

            np.max(
                display_power
            )

            +

            1e-12

        )
    )


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    image = ax.imshow(

        power_db.T,

        origin="lower",

        aspect="auto",

        extent=[

            delay_axis[0],

            delay_axis[-1],

            display_doppler[0],

            display_doppler[-1]

        ]
    )


    fig.colorbar(

        image,

        ax=ax,

        label="Relative Power (dB)"
    )


    # Ground Truth
    ax.scatter(

        truth[
            "delay_samples"
        ],

        truth[
            "doppler_hz"
        ],

        marker="x",

        s=120,

        label="Ground Truth"
    )


    # Estimated Peak
    ax.scatter(

        peak[
            "delay_samples"
        ],

        peak[
            "doppler_hz"
        ],

        marker="o",

        facecolors="none",

        s=120,

        label="Estimated Peak"
    )


    ax.set_title(
        "R1 Range-Doppler Validation"
    )


    ax.set_xlabel(
        "Delay (samples)"
    )


    ax.set_ylabel(
        "Doppler Frequency (Hz)"
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

        "range_doppler_R1.png"
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

    main()