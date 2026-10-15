
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.random_signal import generate_random_signal


CARRIER_FREQUENCY = 650_000_000  # 650 MHz


def generate_transmitted_baseband(
    fs=2_000_000,
    duration=0.02,
    bandwidth=500_000,
    seed=42,
):
    """
    Complex-baseband representation of the
    transmitted waveform.

    The physical carrier frequency is stored
    separately and used in the propagation model.
    """

    baseband = generate_random_signal(
        fs=fs,
        duration=duration,
        bandwidth=bandwidth,
        seed=seed,
    )

    return baseband, CARRIER_FREQUENCY


def upconvert_to_rf(
    baseband,
    sampling_frequency,
    carrier_frequency,
):
    """
    Create a real RF waveform:

    s_RF(t) = Re{s_BB(t) * exp(j*2*pi*fc*t)}

    RF sampling frequency must satisfy Nyquist.
    """

    time = np.arange(len(baseband)) / sampling_frequency

    carrier = np.exp(
        1j * 2 * np.pi * carrier_frequency * time
    )

    transmitted_rf = np.real(baseband * carrier)

    return transmitted_rf


def main():

    # High sampling rate ONLY for short RF demonstration
    fs_rf = 1_600_000_000  # 1.6 GHz
    fc = CARRIER_FREQUENCY
    bandwidth = 500_000

    # Short signal to avoid large memory usage
    n_samples = 65536
    duration_rf = n_samples / fs_rf

    # Call the random-signal module
    random_baseband = generate_random_signal(
        fs=fs_rf,
        duration=duration_rf,
        bandwidth=bandwidth,
        seed=42,
    )

    # Frequency upconversion
    transmitted_rf = upconvert_to_rf(
        random_baseband,
        fs_rf,
        fc,
    )

    # Baseband frequency spectrum
    bb_frequency = np.fft.fftshift(
        np.fft.fftfreq(n_samples, d=1 / fs_rf)
    )

    bb_spectrum = np.abs(
        np.fft.fftshift(np.fft.fft(random_baseband))
    )

    # Positive-frequency spectrum of real RF signal
    rf_frequency = np.fft.rfftfreq(
        n_samples,
        d=1 / fs_rf,
    )

    rf_spectrum = np.abs(
        np.fft.rfft(transmitted_rf)
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    axes[0].plot(
        bb_frequency / 1e6,
        bb_spectrum,
    )
    axes[0].set_xlim(-1, 1)
    axes[0].set_title("Before Upconversion: Random Baseband")
    axes[0].set_xlabel("Frequency (MHz)")
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(True)

    axes[1].plot(
        rf_frequency / 1e6,
        rf_spectrum,
    )
    axes[1].axvline(
        fc / 1e6,
        color="red",
        linestyle="--",
        label="650 MHz carrier",
    )
    axes[1].set_xlim(649, 651)
    axes[1].set_title(
        "After Upconversion: Transmitted RF Signal"
    )
    axes[1].set_xlabel("Frequency (MHz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    output_dir = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "02_transmitted_signal.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print("=== STEP 2: TRANSMITTED SIGNAL ===")
    print(f"Carrier frequency: {fc / 1e6} MHz")
    print(f"RF sampling rate: {fs_rf / 1e9} GHz")
    print(f"RF demonstration duration: {duration_rf:.8f} s")
    print(f"RF samples: {len(transmitted_rf)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
