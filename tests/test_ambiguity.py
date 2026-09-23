import numpy as np

from configs.baseline_config import (
    FS,
)

from src.simulator import (
    generate_reference_signal,
    apply_fractional_delay,
    apply_doppler,
    run_baseline_simulation,
)

from src.ambiguity_func import (
    cross_ambiguity_fft,
    find_ambiguity_peak,
)


# ============================================================
# Test 1
# Known Positive Delay + Positive Doppler
# 已知正 delay + 正 Doppler
# ============================================================

def test_known_delay_and_doppler():

    # --------------------------------------------------------
    # Simple software test configuration
    # --------------------------------------------------------

    fs = 10_000

    duration = 0.1

    bandwidth = 4_000

    seed = 7


    # --------------------------------------------------------
    # Known ground truth
    # --------------------------------------------------------

    true_delay = 5

    true_doppler = 100.0


    # --------------------------------------------------------
    # Generate reference signal
    # --------------------------------------------------------

    reference = generate_reference_signal(
        fs,
        duration,
        bandwidth,
        seed
    )


    # --------------------------------------------------------
    # Apply known delay
    # --------------------------------------------------------

    echo = apply_fractional_delay(
        reference,
        true_delay
    )


    # --------------------------------------------------------
    # Apply known Doppler
    # --------------------------------------------------------

    echo = apply_doppler(
        echo,
        true_doppler,
        fs
    )


    # --------------------------------------------------------
    # Calculate Range-Doppler map
    # --------------------------------------------------------

    (
        power,
        delay_axis,
        doppler_axis
    ) = cross_ambiguity_fft(
        reference,
        echo,
        fs,
        max_delay_samples=10
    )


    # --------------------------------------------------------
    # Find strongest peak
    # --------------------------------------------------------

    peak = find_ambiguity_peak(
        power,
        delay_axis,
        doppler_axis,
        -300,
        300
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n=== Test 1: Known Delay and Doppler ==="
    )

    print(
        "True delay:",
        true_delay
    )

    print(
        "Estimated delay:",
        peak["delay_samples"]
    )

    print(
        "True Doppler:",
        true_doppler
    )

    print(
        "Estimated Doppler:",
        peak["doppler_hz"]
    )


    # --------------------------------------------------------
    # Check result
    # --------------------------------------------------------

    assert (
        peak["delay_samples"]
        ==
        true_delay
    )


    assert abs(
        peak["doppler_hz"]
        -
        true_doppler
    ) < 15


# ============================================================
# Test 2
# Negative Doppler Sign
# 负 Doppler 测试
# ============================================================

def test_negative_doppler():

    # --------------------------------------------------------
    # Simple software test configuration
    # --------------------------------------------------------

    fs = 10_000

    duration = 0.1

    bandwidth = 4_000

    seed = 8


    # --------------------------------------------------------
    # Known ground truth
    # --------------------------------------------------------

    true_delay = 4

    true_doppler = -100.0


    # --------------------------------------------------------
    # Generate reference
    # --------------------------------------------------------

    reference = generate_reference_signal(
        fs,
        duration,
        bandwidth,
        seed
    )


    # --------------------------------------------------------
    # Apply known delay
    # --------------------------------------------------------

    echo = apply_fractional_delay(
        reference,
        true_delay
    )


    # --------------------------------------------------------
    # Apply negative Doppler
    # --------------------------------------------------------

    echo = apply_doppler(
        echo,
        true_doppler,
        fs
    )


    # --------------------------------------------------------
    # Calculate Range-Doppler map
    # --------------------------------------------------------

    (
        power,
        delay_axis,
        doppler_axis
    ) = cross_ambiguity_fft(
        reference,
        echo,
        fs,
        max_delay_samples=10
    )


    # --------------------------------------------------------
    # Find strongest peak
    # --------------------------------------------------------

    peak = find_ambiguity_peak(
        power,
        delay_axis,
        doppler_axis,
        -300,
        300
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n=== Test 2: Negative Doppler ==="
    )

    print(
        "True delay:",
        true_delay
    )

    print(
        "Estimated delay:",
        peak["delay_samples"]
    )

    print(
        "True Doppler:",
        true_doppler
    )

    print(
        "Estimated Doppler:",
        peak["doppler_hz"]
    )


    # --------------------------------------------------------
    # Check result
    # --------------------------------------------------------

    assert (
        peak["delay_samples"]
        ==
        true_delay
    )


    assert abs(
        peak["doppler_hz"]
        -
        true_doppler
    ) < 15


# ============================================================
# Test 3
# Geometry-derived R1 Target Echo
# 使用真实 geometry 自动产生的 R1 delay / Doppler
# ============================================================

def test_geometry_r1():

    # --------------------------------------------------------
    # Run geometry-first simulator
    # --------------------------------------------------------

    (
        receiver_data,
        truth_record,
        _
    ) = run_baseline_simulation()


    # --------------------------------------------------------
    # Select Receiver 1
    # --------------------------------------------------------

    data = receiver_data[
        "R1"
    ]


    # Reference signal
    reference = data[
        "reference"
    ]


    # Clean target echo
    # 暂时不使用 direct path + noise
    clean_target_echo = data[
        "target_echo"
    ]


    # Geometry-derived ground truth
    truth = data[
        "truth"
    ]


    # --------------------------------------------------------
    # Define delay search range
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Calculate Range-Doppler map
    # --------------------------------------------------------

    (
        power,
        delay_axis,
        doppler_axis
    ) = cross_ambiguity_fft(
        reference,
        clean_target_echo,
        FS,
        max_delay
    )


    # --------------------------------------------------------
    # Find estimated target peak
    # --------------------------------------------------------

    peak = find_ambiguity_peak(
        power,
        delay_axis,
        doppler_axis,
        -500,
        500
    )


    # --------------------------------------------------------
    # Calculate Doppler-bin spacing
    # --------------------------------------------------------

    doppler_bin_spacing = abs(
        doppler_axis[1]
        -
        doppler_axis[0]
    )


    # --------------------------------------------------------
    # Print Ground Truth vs Estimate
    # --------------------------------------------------------

    print(
        "\n=== Test 3: Geometry-derived R1 ==="
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
        "Doppler bin spacing:",
        doppler_bin_spacing
    )


    # --------------------------------------------------------
    # Delay validation
    #
    # Current ambiguity map searches integer delay bins,
    # while geometry-derived delay is fractional.
    # Therefore allow error <= 1 sample.
    # --------------------------------------------------------

    assert abs(
        peak[
            "delay_samples"
        ]
        -
        truth[
            "delay_samples"
        ]
    ) <= 1.0


    # --------------------------------------------------------
    # Doppler validation
    #
    # Doppler is quantised by FFT resolution.
    # Therefore allow error within one Doppler bin.
    # --------------------------------------------------------

    assert abs(
        peak[
            "doppler_hz"
        ]
        -
        truth[
            "doppler_hz"
        ]
    ) <= doppler_bin_spacing