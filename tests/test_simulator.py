import numpy as np

from src.simulator import generate_reference_signal


def test_reproducibility():

    signal_1 = generate_reference_signal(
        fs=10000,
        duration=1.0,
        seed=42
    )

    signal_2 = generate_reference_signal(
        fs=10000,
        duration=1.0,
        seed=42
    )

    are_identical = np.allclose(
        signal_1,
        signal_2
    )

    print("\nReproducibility Test")
    print("Seed: 42")
    print("Number of samples:", len(signal_1))
    print("Signals identical:", are_identical)

    assert are_identical