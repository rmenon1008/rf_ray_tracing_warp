import numpy as np
import math

LIGHT_SPEED = 299792458

def watts_to_dbm(watts):
    return 10 * np.log10(watts) + 30

def dbm_to_watts(dbm):
    return 10**((dbm - 30) / 10)

def bounce_amplitude(angle_between):
    theta = (np.pi / 2) - angle_between / 2

    n_1 = 12.0
    n_2 = 1.0
    theta_i = np.arcsin((n_2 * math.sin(theta)) / n_1)
    num = n_2 * math.cos(theta_i) - n_1 * math.cos(theta)
    denom = n_2 * math.cos(theta_i) + n_1 * math.cos(theta)
    amp = -(num / denom)**2

    return amp

def paths_to_phase_amp(paths, tx_num_rays, tx_power_watts, freq_hz, noise_dbm=-45):
    combined_signal = 0
    for path in paths:
        path_len = 0
        amplitude = tx_power_watts / tx_num_rays
        for p in range(len(path) - 2):
            seg1 = path[p + 1] - path[p]
            seg2 = path[p + 2] - path[p + 1]
            angle_between = np.arccos(np.dot(seg1, seg2) / (np.linalg.norm(seg1) * np.linalg.norm(seg2)))
            amplitude *= bounce_amplitude(angle_between)
            path_len += np.linalg.norm(seg1)
        path_len += np.linalg.norm(path[-2] - path[-1])
        path_len_wrapped_rad = (path_len / LIGHT_SPEED) * freq_hz * 2 * np.pi
        new_signal = amplitude * np.exp(-1j * 2 * np.pi * freq_hz) * np.exp(-1j * path_len_wrapped_rad)
        if not np.isnan(new_signal):
            combined_signal += new_signal

    # Add AWGN
    noise = np.random.normal(0, dbm_to_watts(noise_dbm))
    combined_signal += noise
    
    # Return the amplitude and phase of the combined signal
    return np.abs(combined_signal), np.angle(combined_signal)

def paths_to_csi(paths, tx_num_rays, tx_power_watts, channel_center_freq=2.401e9, noise_dbm=-45):
    subcarriers = np.linspace(channel_center_freq - 20e6, channel_center_freq + 20e6, 64)
    csi_mag = np.zeros_like(subcarriers)
    csi_phase = np.zeros_like(subcarriers)

    for i in range(len(subcarriers)):
        csi_mag[i], csi_phase[i] = paths_to_phase_amp(paths, tx_num_rays, tx_power_watts, subcarriers[i], noise_dbm=noise_dbm)

    return csi_mag, csi_phase
