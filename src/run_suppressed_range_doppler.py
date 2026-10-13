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

from src.direct_path_cancellation import (
    cancel_direct_path,
)


def main():

    # ========================================================
    # 1. Run geometry-first simulator
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

    surveillance = data["surveillance"]

    truth = data["truth"]


    # ========================================================
    # 3. Apply Direct-Path Cancellation
    # ========================================================

    cleaned_signal, alpha = cancel_direct_path(
        reference,
        surveillance
    )


    print()

    print(
        "=========================================="
    )

    print(
        "DIRECT-PATH CANCELLATION - R1"
    )

    print(
        "=========================================="
    )

    print()

    print(
        "Estimated direct-path coefficient alpha:"
    )

    print(
        alpha
    )


    # ========================================================
    # 4. Compare direct-path correlation
    #    before and after suppression
    # ========================================================

    correlation_before = abs(
        np.vdot(
            reference,
            surveillance
        )
    )


    correlation_after = abs(
        np.vdot(
            reference,
            cleaned_signal
        )
    )


    print()

    print(
        "Correlation before suppression:",
        correlation_before
    )

    print(
        "Correlation after suppression:",
        correlation_after
    )


    # ========================================================
    # 5. Define Delay Search Range
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
    # 6. Run Range-Doppler Processing
    #    on the SUPPRESSED signal
    # ========================================================

    (
        power,
        delay_axis,
        doppler_axis
    ) = cross_ambiguity_fft(

        reference,

        cleaned_signal,

        FS,

        max_delay
    )


    # ========================================================
    # 7. Find strongest peak after suppression
    # ========================================================

    peak = find_ambiguity_peak(

        power,

        delay_axis,

        doppler_axis,

        -500,

        500
    )


    # ========================================================
    # 8. Print Target Truth vs Estimated Peak
    # ========================================================

    print()

    print(
        "=========================================="
    )

    print(
        "SUPPRESSED RANGE-DOPPLER RESULT - R1"
    )

    print(
        "=========================================="
    )

    print()

    print(
        "True target delay:",
        truth["delay_samples"],
        "samples"
    )

    print(
        "Estimated peak delay:",
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
        "Estimated peak Doppler:",
        peak["doppler_hz"],
        "Hz"
    )


    # ========================================================
    # 9. Calculate errors
    # ========================================================

    delay_error = abs(
        peak["delay_samples"]
        -
        truth["delay_samples"]
    )


    doppler_error = abs(
        peak["doppler_hz"]
        -
        truth["doppler_hz"]
    )


    print()

    print(
        "Delay error:",
        delay_error,
        "samples"
    )

    print(
        "Doppler error:",
        doppler_error,
        "Hz"
    )


    # ========================================================
    # 10. Limit displayed Doppler region
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
    # 11. Convert power to relative dB
    # ========================================================

    maximum_power = np.max(
        display_power
    )


    power_db = (

        10.0

        *

        np.log10(

            display_power

            /

            maximum_power

            +

            1e-12
        )
    )


    # ========================================================
    # 12. Plot Suppressed Range-Doppler Map
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
    # 13. Mark Ground-Truth Target
    # ========================================================

    ax.scatter(

        truth["delay_samples"],

        truth["doppler_hz"],

        marker="x",

        s=120,

        label="True Target"
    )


    # ========================================================
    # 14. Mark Estimated Peak
    # ========================================================

    ax.scatter(

        peak["delay_samples"],

        peak["doppler_hz"],

        marker="o",

        facecolors="none",

        s=120,

        label="Estimated Peak"
    )


    # ========================================================
    # 15. Figure Settings
    # ========================================================

    ax.set_title(
        "R1 Range-Doppler Map After Direct-Path Suppression"
    )


    ax.set_xlabel(
        "Delay (samples)"
    )


    ax.set_ylabel(
        "Doppler Frequency (Hz)"
    )


    ax.legend()


    # ========================================================
    # 16. Save Figure
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

        "range_doppler_R1_suppressed.png"

    )


    fig.tight_layout()


    fig.savefig(

        output_path,

        dpi=200
    )


    plt.show()


    print()

    print(
        "Suppressed Range-Doppler figure saved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":

    main()