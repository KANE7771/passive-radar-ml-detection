
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.reference_and_surveillance_signals import (
    run_reference_and_surveillance_simulation,
    FS,
)

from src.ambiguity_func import (
    cross_ambiguity_fft,
    find_ambiguity_peak,
)


# =====================================================
# STEP 5: STATIONARY TARGET CAF EXPERIMENT
# =====================================================

MAX_DELAY_SAMPLES = 20
DOPPLER_MIN_HZ = -500
DOPPLER_MAX_HZ = 500


# =====================================================
# 1. Calculate CAF and estimate its peak
# =====================================================

def analyse_caf(reference, received):

    power, delay_axis, doppler_axis = cross_ambiguity_fft(
        reference,
        received,
        FS,
        MAX_DELAY_SAMPLES,
    )

    power = np.asarray(power)
    delay_axis = np.asarray(delay_axis)
    doppler_axis = np.asarray(doppler_axis)

    # Check CAF matrix dimensions
    expected_shape = (
        len(delay_axis),
        len(doppler_axis),
    )

    if power.shape != expected_shape:
        raise ValueError(
            f"CAF shape {power.shape} "
            f"does not match {expected_shape}"
        )

    # Find strongest peak in selected Doppler window
    peak = find_ambiguity_peak(
        power,
        delay_axis,
        doppler_axis,
        DOPPLER_MIN_HZ,
        DOPPLER_MAX_HZ,
    )

    return {
        "power": power,
        "delay_axis": delay_axis,
        "doppler_axis": doppler_axis,
        "delay_samples": float(peak["delay_samples"]),
        "doppler_hz": float(peak["doppler_hz"]),
    }


# =====================================================
# 2. Plot one CAF map
# =====================================================

def plot_caf_map(
    ax,
    result,
    title,
    true_delay_samples,
):

    power = result["power"]
    delay_axis = result["delay_axis"]
    doppler_axis = result["doppler_axis"]

    # Select Doppler frequencies for display
    mask = (
        (doppler_axis >= DOPPLER_MIN_HZ)
        & (doppler_axis <= DOPPLER_MAX_HZ)
    )

    displayed_doppler = doppler_axis[mask]
    displayed_power = power[:, mask]

    if len(displayed_doppler) < 2:
        raise ValueError(
            "Not enough Doppler bins in display window."
        )

    if len(delay_axis) < 2:
        raise ValueError(
            "Not enough delay bins."
        )

    # Normalise CAF power to its own strongest peak
    max_power = np.max(displayed_power)

    if max_power <= 0:
        raise ValueError("CAF has zero power.")

    power_db = 10 * np.log10(
        np.maximum(displayed_power / max_power, 1e-12)
    )

    delay_step = delay_axis[1] - delay_axis[0]
    doppler_step = (
        displayed_doppler[1] - displayed_doppler[0]
    )

    # Draw 2D Delay-Doppler map
    image = ax.imshow(
        power_db.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[
            delay_axis[0] - delay_step / 2,
            delay_axis[-1] + delay_step / 2,
            displayed_doppler[0] - doppler_step / 2,
            displayed_doppler[-1] + doppler_step / 2,
        ],
        cmap="viridis",
        vmin=-50,
        vmax=0,
    )

    # Geometry-derived target position
    ax.scatter(
        true_delay_samples,
        0.0,
        marker="x",
        color="red",
        s=110,
        linewidths=2,
        label="Target truth",
        zorder=5,
    )

    # Estimated maximum CAF peak
    ax.scatter(
        result["delay_samples"],
        result["doppler_hz"],
        marker="o",
        facecolors="none",
        edgecolors="white",
        s=125,
        linewidths=2,
        label="Strongest CAF peak",
        zorder=6,
    )

    ax.set_title(title)
    ax.set_xlabel("Relative Delay (samples)")
    ax.set_ylabel("Doppler Frequency (Hz)")
    ax.legend(loc="upper right")

    return image


# =====================================================
# 3. Run complete stationary-target CAF experiment
# =====================================================

def run_stationary_target_caf(make_plot=True):

    # Step 3: Generate reference and surveillance
    data = run_reference_and_surveillance_simulation()

    reference = data["reference"]
    target_echo = data["target_echo"]
    surveillance = data["surveillance"]

    # Ground truth comes from known geometry
    true_delay_samples = (
        data["geometry"]["bistatic_delay"] * FS
    )

    true_doppler_hz = float(data["doppler_hz"])

    # ---------------------------------------------
    # Experiment A: Isolated target echo
    # ---------------------------------------------

    target_result = analyse_caf(
        reference,
        target_echo,
    )

    # ---------------------------------------------
    # Experiment B: Full surveillance
    # ---------------------------------------------

    surveillance_result = analyse_caf(
        reference,
        surveillance,
    )

    # ---------------------------------------------
    # Numerical comparison
    # ---------------------------------------------

    target_delay_error = (
        target_result["delay_samples"]
        - true_delay_samples
    )

    target_doppler_error = (
        target_result["doppler_hz"]
        - true_doppler_hz
    )

    print("\n========================================")
    print("STEP 5: STATIONARY TARGET CAF")
    print("========================================")

    print("\n--- Ground Truth ---")
    print(
        f"True relative delay: "
        f"{true_delay_samples:.6f} samples"
    )
    print(
        f"True Doppler: "
        f"{true_doppler_hz:.6f} Hz"
    )

    print("\n--- Experiment A: Target Echo ---")
    print(
        f"CAF estimated delay: "
        f"{target_result['delay_samples']:.6f} samples"
    )
    print(
        f"CAF estimated Doppler: "
        f"{target_result['doppler_hz']:.6f} Hz"
    )
    print(
        f"Delay error: "
        f"{target_delay_error:.6f} samples"
    )
    print(
        f"Doppler error: "
        f"{target_doppler_error:.6f} Hz"
    )

    print("\n--- Experiment B: Full Surveillance ---")
    print(
        f"Strongest CAF delay: "
        f"{surveillance_result['delay_samples']:.6f} samples"
    )
    print(
        f"Strongest CAF Doppler: "
        f"{surveillance_result['doppler_hz']:.6f} Hz"
    )

    doppler_axis = np.sort(target_result["doppler_axis"])
    doppler_spacing = float(
        np.min(np.diff(doppler_axis))
    )

    print(
        f"\nDoppler bin spacing: "
        f"{doppler_spacing:.6f} Hz"
    )

    # ---------------------------------------------
    # Plot two CAF results
    # ---------------------------------------------

    if make_plot:

        fig, axes = plt.subplots(
            2, 1,
            figsize=(11, 11),
        )

        image_a = plot_caf_map(
            axes[0],
            target_result,
            "A: Reference vs Isolated Stationary Target Echo",
            true_delay_samples,
        )

        plot_caf_map(
            axes[1],
            surveillance_result,
            "B: Reference vs Full Surveillance",
            true_delay_samples,
        )

        # Mark direct-path position in Experiment B
        axes[1].scatter(
            0.0,
            0.0,
            marker="+",
            color="cyan",
            s=160,
            linewidths=2,
            label="Direct-path truth",
            zorder=7,
        )

        axes[1].legend(loc="upper right")

        fig.colorbar(
            image_a,
            ax=axes,
            label="Normalised CAF Power (dB)",
            fraction=0.025,
            pad=0.02,
        )

        fig.subplots_adjust(
            left=0.10,
            right=0.88,
            top=0.94,
            bottom=0.07,
            hspace=0.30,
        )

        # Save results
        output_dir = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "figures"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir / "05_stationary_target_caf.png"
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        print("\nFigure saved to:", output_path)

    return {
        "true_delay_samples": true_delay_samples,
        "true_doppler_hz": true_doppler_hz,
        "target_delay_samples": target_result["delay_samples"],
        "target_doppler_hz": target_result["doppler_hz"],
        "target_delay_error": target_delay_error,
        "target_doppler_error": target_doppler_error,
        "surveillance_peak_delay": (
            surveillance_result["delay_samples"]
        ),
        "surveillance_peak_doppler": (
            surveillance_result["doppler_hz"]
        ),
        "doppler_bin_spacing_hz": doppler_spacing,
    }


if __name__ == "__main__":
    run_stationary_target_caf(make_plot=True)
