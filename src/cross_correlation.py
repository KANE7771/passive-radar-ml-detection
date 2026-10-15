
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

from src.reference_and_surveillance_signals import (
    run_reference_and_surveillance_simulation,
    FS,
)


def calculate_cross_correlation(reference, received, max_lag=20):
    """
    Positive lag means the received signal
    is delayed relative to the reference signal.
    """

    correlation = correlate(
        received,
        reference,
        mode="full",
        method="fft"
    )

    lags = correlation_lags(
        len(received),
        len(reference),
        mode="full"
    )

    mask = (lags >= 0) & (lags <= max_lag)

    selected_lags = lags[mask]
    selected_correlation = np.abs(correlation[mask])

    peak_index = np.argmax(selected_correlation)
    peak_delay = int(selected_lags[peak_index])

    return selected_lags, selected_correlation, peak_delay


def main():

    # Step 3: Generate reference and surveillance
    data = run_reference_and_surveillance_simulation()

    reference = data["reference"]
    surveillance = data["surveillance"]
    target_echo = data["target_echo"]

    true_delay = data["geometry"]["bistatic_delay"] * FS

    # Experiment A: Reference vs isolated target echo
    lags_a, corr_a, peak_a = calculate_cross_correlation(
        reference,
        target_echo
    )

    # Experiment B: Reference vs full surveillance
    lags_b, corr_b, peak_b = calculate_cross_correlation(
        reference,
        surveillance
    )

    print("\n=== STEP 4: CROSS-CORRELATION ===")

    print(f"True target delay: {true_delay:.6f} samples")
    print(f"Target-echo correlation peak: {peak_a} samples")
    print(f"Full-surveillance peak: {peak_b} samples")
    print(f"Target delay error: {peak_a - true_delay:.6f} samples")

    print("\nTarget velocity:", data["target_velocity"])
    print("True Doppler shift:", data["doppler_hz"], "Hz")

    # Plot correlation results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(lags_a, corr_a)
    axes[0].axvline(
        true_delay,
        color="red",
        linestyle="--",
        label=f"True delay: {true_delay:.3f}"
    )
    axes[0].axvline(
        peak_a,
        color="green",
        linestyle=":",
        label=f"Estimated delay: {peak_a}"
    )
    axes[0].set_title(
        "Cross-Correlation: Reference vs Target Echo"
    )
    axes[0].legend()

    axes[1].plot(lags_b, corr_b)
    axes[1].axvline(
        0,
        color="red",
        linestyle="--",
        label="Direct path: 0 samples"
    )
    axes[1].axvline(
        true_delay,
        color="green",
        linestyle=":",
        label="True target delay"
    )
    axes[1].set_title(
        "Cross-Correlation: Reference vs Full Surveillance"
    )
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel("Relative Delay (samples)")
        ax.set_ylabel("Correlation Magnitude")
        ax.grid(True)

    fig.tight_layout()

    output_dir = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "figures"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "04_cross_correlation.png"

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print("\nFigure saved to:", output_path)


if __name__ == "__main__":
    main()
