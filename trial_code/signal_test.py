import numpy as np
import torch
import matplotlib.pyplot as plt

# Sampling parameters
sampling_rate = 1000  # Hz
duration = 1  # seconds
t = np.linspace(0, duration, int(sampling_rate * duration))

# Frequency components
frequency_1 = 50  # Hz
frequency_2 = 150  # Hz

# Exponential decay factors
decay_1 = 0.02
decay_2 = 0.1

# Generate exponential components
exp_component_1 = np.exp(-decay_1 * t)
exp_component_2 = np.exp(-decay_2 * t)

# Generate RF signal by summing the exponential components
rf_signal = np.sin(2 * np.pi * frequency_1 * t) * exp_component_1 + \
            np.sin(2 * np.pi * frequency_2 * t) * exp_component_2

# Radio signal
Gt = 1 # Trasmitter Gain
Gr = 1 # Receiver Gain
lambd = # 
l0 = # path length
signal =
R = torch.sqrt(Gt * Gr) * lambd / (4 * torch.pi) * (signal * exp_component_1)

# Plot the RF signal
plt.figure(figsize=(10, 6))
plt.plot(t, rf_signal)
plt.title("RF Signal with Exponential Components")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
