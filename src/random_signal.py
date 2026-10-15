
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def generate_random_signal(
    fs=2_000_000,
    duration=0.02,
    bandwidth=500_000,
    seed=42,
):
    """
    Generate a reproducible band-limited complex
    random baseband signal.
    """

    if not (0 < bandwidth < fs):
        raise ValueError("Bandwidth must be between 0 and fs.")

    n_samples = int(round(fs * duration))
    if n_samples < 2:
        raise ValueError("Signal duration is too short.")

    rng = np.random.default_rng(seed)

    # Step 1: Random complex frequency components
    random_spectrum = (
        rng.standard_normal(n_samples)
        + 1j * rng.standard_normal(n_samples)
    )

    # Step 2: Frequency bins
    frequencies = np.fft.fftfreq(n_samples, d=1 / fs)

    # Step 3: Band-limit around 0 Hz
    frequency_mask = np.abs(frequencies) <= bandwidth / 2
    random_spectrum[~frequency_mask] = 0

    # Step 4: Convert to time-domain samples
    signal = np.fft.ifft(random_spectrum)

    # Step 5: Normalise average power to 1
    power = np.mean(np.abs(signal) ** 2)
    signal = signal / np.sqrt(power)

    return signal


def main():

    fs = 2_000_000
    duration = 0.02
    bandwidth = 500_000
    seed = 42

    signal = generate_random_signal(
        fs=fs,
        duration=duration,
        bandwidth=bandwidth,
        seed=seed,
    )

    n_samples = len(signal)
    time = np.arange(n_samples) / fs

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(n_samples, d=1 / fs)
    )

    spectrum = np.abs(
        np.fft.fftshift(np.fft.fft(signal))
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # Time-domain waveform
    axes[0].plot(
        time[:150] * 1e6,
        signal[:150].real,
        label="Real part",
    )

    axes[0].plot(
        time[:150] * 1e6,
        signal[:150].imag,
        label="Imaginary part",
        alpha=0.7,
    )

    axes[0].set_title("Random Complex Baseband Signal")
    axes[0].set_xlabel("Time (microseconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True)

    # Frequency spectrum
    axes[1].plot(
        frequencies / 1e6,
        spectrum,
    )
    axes[1].set_xlim(-1, 1)
    axes[1].set_title("Random Baseband Spectrum")
    axes[1].set_xlabel("Frequency (MHz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True)

    fig.tight_layout()

    output_dir = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "01_random_signal.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print("=== STEP 1: RANDOM SIGNAL ===")
    print(f"Sampling frequency: {fs / 1e6} MHz")
    print(f"Bandwidth: {bandwidth / 1e3} kHz")
    print(f"Duration: {duration} s")
    print(f"Samples: {n_samples}")
    print(f"Average power: {np.mean(np.abs(signal)**2):.6f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
