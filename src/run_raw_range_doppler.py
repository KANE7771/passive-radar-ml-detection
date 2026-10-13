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

    # ========================================================
    # 1. Run the geometry-first simulator
    # ========================================================

    (
        receiver_data,
        truth_record,
        _
    ) = run_baseline_simulation()


    # ========================================================
    # 2. Select Receiver R1
    # ========================================================

    data = receiver_data["R1"]

    reference = data["reference"]

    # IMPORTANT:
    # Use the FULL surveillance signal here.
    #
    # surveillance =
    # direct path + target echo + noise
    #
    # This is different from the previous clean experiment,
    # which used data["target_echo"] only.
    received = data["surveillance"]

    truth = data["truth"]


    # ========================================================
    # 3. Define delay search range
    # ========================================================

    max_truth_delay = max(

        receiver["delay_samples"]

        for receiver
        in truth_record["receivers"].values()
    )


    max_delay = (
        int(
            np.ceil(
                max_truth_delay
            )
        )
        +
        6
    )


    # ========================================================
    # 4. Calculate Raw Range-Doppler Map
    # ========================================================

    (
        power,
        delay_axis,
        doppler_axis
    ) = cross_ambiguity_fft(

        reference,

        received,

        FS,

        max_delay
    )


    # ========================================================
    # 5. Find strongest peak
    # ========================================================

    peak = find_ambiguity_peak(

        power,

        delay_axis,

        doppler_axis,

        -500,

        500
    )


    # ========================================================
    # 6. Print Ground Truth and Strongest Raw Peak
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "RAW SURVEILLANCE RANGE-DOPPLER - R1"
    )

    print(
        "========================================"
    )

    print()

    print(
        "True target delay:",
        truth["delay_samples"],
        "samples"
    )

    print(
        "Strongest raw peak delay:",
        peak["delay_samples"],
        "samples"
    )

    print()

    print(
        "True target Doppler:",
        truth["doppler_hz"],
        "Hz"
    )

    print(
        "Strongest raw peak Doppler:",
        peak["doppler_hz"],
        "Hz"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "If the strongest peak is near delay = 0 "
        "and Doppler = 0 Hz, this is expected."
    )

    print(
        "It indicates that the direct-path component "
        "is dominating the raw surveillance signal."
    )


    # ========================================================
    # 7. Limit displayed Doppler region
    # ========================================================

    doppler_limit = 500.0

    doppler_mask = (

        (
            doppler_axis
            >=
            -doppler_limit
        )

        &

        (
            doppler_axis
            <=
            doppler_limit
        )
    )


    display_power = power[
        :,
        doppler_mask
    ]


    display_doppler = doppler_axis[
        doppler_mask
    ]


    # ========================================================
    # 8. Convert power to relative dB
    # ========================================================

    power_db = (

        10.0

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


    # ========================================================
    # 9. Plot Raw Range-Doppler Map
    # ========================================================

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

        label="Relative Ambiguity Power (dB)"
    )


    # ========================================================
    # 10. Mark true target location
    # ========================================================

    ax.scatter(

        truth["delay_samples"],

        truth["doppler_hz"],

        marker="x",

        s=120,

        label="True Target"
    )


    # ========================================================
    # 11. Mark strongest raw peak
    # ========================================================

    ax.scatter(

        peak["delay_samples"],

        peak["doppler_hz"],

        marker="o",

        facecolors="none",

        s=120,

        label="Strongest Raw Peak"
    )


    # ========================================================
    # 12. Figure settings
    # ========================================================

    ax.set_title(
        "R1 Raw Surveillance Range-Doppler Map"
    )


    ax.set_xlabel(
        "Delay (samples)"
    )


    ax.set_ylabel(
        "Doppler Frequency (Hz)"
    )


    ax.legend()


    # ========================================================
    # 13. Save Figure
    # ========================================================

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

        "range_doppler_R1_raw.png"

    )


    fig.tight_layout()


    fig.savefig(

        output_path,

        dpi=200

    )


    plt.show()


    print()

    print(
        "Raw Range-Doppler figure saved to:",
        output_path
    )


if __name__ == "__main__":

    main()