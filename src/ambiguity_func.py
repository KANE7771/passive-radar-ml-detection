import numpy as np

def cross_ambiguity_fft(reference, received, fs, max_delay_samples):
    """
    Search across delay and Doppler.

    Output:
        ambiguity power
        delay axis
        Doppler axis
    """
    reference = np.asarray(reference, dtype=complex)
    received = np.asarray(received, dtype=complex)
    max_delay_samples = int(max_delay_samples)
    n_valid = len(reference) - max_delay_samples
    reference_segment = reference[:n_valid]
    window = np.hanning(n_valid)
    delay_axis = np.arange(max_delay_samples + 1)
    ambiguity = np.zeros((len(delay_axis), n_valid), dtype=complex)
    for row, delay in enumerate(delay_axis):
        received_segment = received[delay:delay + n_valid]
        product = received_segment * np.conj(reference_segment) * window
        ambiguity[row, :] = np.fft.fftshift(np.fft.fft(product))
    doppler_axis = np.fft.fftshift(np.fft.fftfreq(n_valid, d=1.0 / fs))
    ambiguity_power = np.abs(ambiguity) ** 2
    return (ambiguity_power, delay_axis, doppler_axis)

def find_ambiguity_peak(ambiguity_power, delay_axis, doppler_axis, doppler_min=-500, doppler_max=500):
    mask = (doppler_axis >= doppler_min) & (doppler_axis <= doppler_max)
    restricted_power = ambiguity_power[:, mask]
    flat_index = np.argmax(restricted_power)
    delay_index, doppler_sub_index = np.unravel_index(flat_index, restricted_power.shape)
    valid_doppler_indices = np.where(mask)[0]
    doppler_index = valid_doppler_indices[doppler_sub_index]
    return {'delay_samples': float(delay_axis[delay_index]), 'doppler_hz': float(doppler_axis[doppler_index])}
