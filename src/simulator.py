import numpy as np
import json
import os
import matplotlib.pyplot as plt


# =========================================================
# 1. Generate reference signal
# =========================================================

def generate_reference_signal(fs, duration, seed):

    rng = np.random.default_rng(seed)

    n_samples = int(fs * duration)

    reference = (
        rng.normal(size=n_samples)
        + 1j * rng.normal(size=n_samples)
    )

    # Normalise signal power
    reference = reference / np.sqrt(
        np.mean(np.abs(reference) ** 2)
    )

    return reference


# =========================================================
# 2. Apply time delay
# =========================================================

def apply_delay(signal, delay_samples):

    delayed = np.zeros_like(signal)

    if delay_samples == 0:
        return signal.copy()

    delayed[delay_samples:] = signal[:-delay_samples]

    return delayed


# =========================================================
# 3. Apply Doppler shift
# =========================================================

def apply_doppler(signal, doppler_hz, fs):

    n = np.arange(len(signal))

    doppler_phase = np.exp(
        1j * 2 * np.pi * doppler_hz * n / fs
    )

    return signal * doppler_phase


# =========================================================
# 4. Generate surveillance signal
# =========================================================

def generate_surveillance(
        reference,
        delay_samples,
        doppler_hz,
        fs,
        direct_path_amplitude,
        target_amplitude,
        noise_std,
        seed):

    target_echo = apply_delay(
        reference,
        delay_samples
    )

    target_echo = apply_doppler(
        target_echo,
        doppler_hz,
        fs
    )

    rng = np.random.default_rng(seed + 1)

    noise = (
        rng.normal(
            scale=noise_std,
            size=len(reference)
        )
        +
        1j * rng.normal(
            scale=noise_std,
            size=len(reference)
        )
    )

    surveillance = (
        direct_path_amplitude * reference
        +
        target_amplitude * target_echo
        +
        noise
    )

    return surveillance, target_echo, noise
# =========================================================
# 5. Run baseline experiment
# =========================================================

if __name__ == "__main__":

    # Baseline configuration
    SEED = 42
    FS = 10000
    DURATION = 1.0

    TARGET_DELAY_SAMPLES = 20
    TARGET_DOPPLER_HZ = 50

    DIRECT_PATH_AMPLITUDE = 1.0
    TARGET_AMPLITUDE = 0.1
    NOISE_STD = 0.05

    # Generate reference signal
    reference = generate_reference_signal(
        FS,
        DURATION,
        SEED
    )

    # Generate surveillance signal
    surveillance, target_echo, noise = generate_surveillance(
        reference,
        TARGET_DELAY_SAMPLES,
        TARGET_DOPPLER_HZ,
        FS,
        DIRECT_PATH_AMPLITUDE,
        TARGET_AMPLITUDE,
        NOISE_STD,
        SEED
    )

    print("Number of samples:", len(reference))

    print(
        "Reference average power:",
        np.mean(np.abs(reference) ** 2)
    )

    print(
        "True target delay:",
        TARGET_DELAY_SAMPLES,
        "samples"
    )

    print(
        "True target Doppler:",
        TARGET_DOPPLER_HZ,
        "Hz"
    )
    # Create results folder
    os.makedirs(
        "results/figures",
        exist_ok=True
    )

    # Store simulation ground truth
    truth = {
        "seed": SEED,
        "delay_samples": TARGET_DELAY_SAMPLES,
        "doppler_hz": TARGET_DOPPLER_HZ,
        "target_amplitude": TARGET_AMPLITUDE
    }

    with open(
            "../results/figures/truth.json",
        "w"
    ) as file:

        json.dump(
            truth,
            file,
            indent=4
        )