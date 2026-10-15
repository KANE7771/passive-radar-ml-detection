
import numpy as np

from src.stationary_target_caf import (
    run_stationary_target_caf,
)


# =====================================================
# Test 1: Stationary target has zero Doppler
# =====================================================

def test_stationary_doppler_truth():

    result = run_stationary_target_caf(
        make_plot=False
    )

    assert np.isclose(
        result["true_doppler_hz"],
        0.0,
    )


# =====================================================
# Test 2: CAF estimates the target time delay
# =====================================================

def test_target_delay_estimation():

    result = run_stationary_target_caf(
        make_plot=False
    )

    true_delay = result["true_delay_samples"]
    estimated_delay = result["target_delay_samples"]

    print("\nTrue delay:", true_delay)
    print("Estimated delay:", estimated_delay)

    # Integer-delay estimator: allow 1 sample error
    assert abs(
        estimated_delay - true_delay
    ) <= 1.0


# =====================================================
# Test 3: CAF estimates near-zero Doppler
# =====================================================

def test_zero_doppler_estimation():

    result = run_stationary_target_caf(
        make_plot=False
    )

    estimated_doppler = result["target_doppler_hz"]
    doppler_spacing = result["doppler_bin_spacing_hz"]

    print("\nTrue Doppler: 0 Hz")
    print("Estimated Doppler:", estimated_doppler)
    print("Doppler bin spacing:", doppler_spacing)

    assert abs(estimated_doppler) <= doppler_spacing


# =====================================================
# Test 4: Full surveillance dominated by direct path
# =====================================================

def test_full_surveillance_peak():

    result = run_stationary_target_caf(
        make_plot=False
    )

    peak_delay = result["surveillance_peak_delay"]
    peak_doppler = result["surveillance_peak_doppler"]

    print("\nFull surveillance peak delay:", peak_delay)
    print("Full surveillance peak Doppler:", peak_doppler)

    # Expected for this fixed, direct-path-dominated
    # baseline scenario, not a general radar rule.
    assert peak_delay == 0.0

    assert abs(peak_doppler) <= (
        result["doppler_bin_spacing_hz"]
    )
