import json
from pathlib import Path

import numpy as np

from configs.baseline_config import (
    C,
    FS,
)

from src.simulator import (
    run_baseline_simulation,
)

from src.direct_path_cancellation import (
    cancel_direct_path,
)

from src.ambiguity_func import (
    cross_ambiguity_fft,
    find_ambiguity_peak,
)


def main():

    # --------------------------------------------------------
    # Generate baseline data
    # --------------------------------------------------------

    (
        receiver_data,
        truth_record,
        _
    ) = run_baseline_simulation()


    # --------------------------------------------------------
    # Use common delay-search limit
    # --------------------------------------------------------

    max_truth_delay = max(

        receiver[
            "delay_samples"
        ]

        for receiver
        in truth_record[
            "receivers"
        ].values()

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


    measurements = {}


    # ========================================================
    # Process R1, R2 and R3
    # ========================================================

    for (
        receiver_name,
        data
    ) in receiver_data.items():


        reference = data[
            "reference"
        ]

        surveillance = data[
            "surveillance"
        ]

        truth = data[
            "truth"
        ]


        # ----------------------------------------------------
        # Direct-path cancellation
        # ----------------------------------------------------

        cleaned, alpha = cancel_direct_path(
            reference,
            surveillance
        )


        # ----------------------------------------------------
        # Range-Doppler processing
        # ----------------------------------------------------

        (
            power,
            delay_axis,
            doppler_axis
        ) = cross_ambiguity_fft(

            reference,

            cleaned,

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


        # ----------------------------------------------------
        # Estimated delay -> seconds
        # ----------------------------------------------------

        estimated_delay_seconds = (
            peak[
                "delay_samples"
            ]
            /
            FS
        )


        # ----------------------------------------------------
        # Delay -> bistatic excess range
        # ----------------------------------------------------

        estimated_range = (
            C
            *
            estimated_delay_seconds
        )


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        measurements[
            receiver_name
        ] = {

            "true_delay_samples":
                truth[
                    "delay_samples"
                ],

            "estimated_delay_samples":
                peak[
                    "delay_samples"
                ],

            "true_doppler_hz":
                truth[
                    "doppler_hz"
                ],

            "estimated_doppler_hz":
                peak[
                    "doppler_hz"
                ],

            "true_bistatic_excess_range_m":
                truth[
                    "bistatic_excess_range_m"
                ],

            "estimated_bistatic_excess_range_m":
                estimated_range,
        }


        print()
        print(
            "================================"
        )

        print(
            receiver_name
        )

        print(
            "================================"
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

        print(
            "True bistatic range:",
            truth[
                "bistatic_excess_range_m"
            ],
            "m"
        )

        print(
            "Estimated bistatic range:",
            estimated_range,
            "m"
        )


    # ========================================================
    # Save measurements
    # ========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )


    output_path = (
        project_root
        /
        "results"
        /
        "measurements.json"
    )


    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            measurements,
            file,
            indent=4
        )


    print()

    print(
        "Measurements saved to:",
        output_path
    )


if __name__ == "__main__":

    main()