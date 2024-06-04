import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

parser = argparse.ArgumentParser()

parser.add_argument("file_name", type=str, help="Name of file to generate dynamic spectra from")

args = parser.parse_args()

def median_filter(spectra, margin=1):
    filtered = np.zeros_like(spectra)
    rows, cols = spectra.shape

    for i in range(rows):
        for j in range(cols):
            i_min = max(i - margin, 0)
            i_max = min(i + margin + 1, rows)
            j_min = max(j - margin, 0)
            j_max = min(j + margin + 1, cols)

            subarray = spectra[i_min:i_max, j_min:j_max]
            filtered[i, j] = np.median(subarray)

    return filtered


def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


file_path = args.file_name
timeMin = 0
timeMax = 0
freqMin = 0
freqMax = 0

x = 0
y = 0
update = False
getMinValues = True
peak = 1
data = [1, 1]
with open(file_path, 'r') as file:
    for line in file:
        if line.startswith("# File"):
            nsub = re.search(r'Nsub:\s*(\d+)', line)
            nch = re.search(r'Nch:\s*(\d+)', line)
            dynamic_spectr = np.zeros((int(nsub.group(1)), int(nch.group(1))))
            continue

        elif line.startswith("#"):
            stdev = np.std(data)
            noise = stdev * 3
            if noise == 0 or peak == 0:
                snr = 0
            else:
                snr = peak / noise
            dynamic_spectr[x][y] = snr

            update = True
            peak = 0
            data = []

            timeMax = re.search(r'MJD\(mid\):\s*([\d.]+)', line)
            freqMax = re.search(r'Freq:\s*([\d.]+)', line)
            if getMinValues:
                timeMin = re.search(r'MJD\(mid\):\s*([\d.]+)', line)
                freqMin = re.search(r'Freq:\s*([\d.]+)', line)
                getMinValues = False
            continue

        if update:
            columns = line.split()
            x = int(columns[0])
            y = int(columns[1])
            update = False
        # getting peak value
        columns = line.split()
        value = float(columns[3])
        if value > peak:
            peak = value
        data.append(value)

freqMax = float(freqMax.group(1))
freqMin = float(freqMin.group(1))
timeMax = float(timeMax.group(1))
timeMin = float(timeMin.group(1))
timeDiff_seconds = (timeMax - timeMin) * 24 * 60 * 60

dynamic_spectr = np.flipud(dynamic_spectr.T)

filtered_spectra = median_filter(dynamic_spectr, 3)
#filtered_spectra = gaussian_filter(dynamic_spectr, sigma=2)


# FT to obtain ACF
padded_shape = (len(filtered_spectra) * 2 - 1, len(filtered_spectra[1]) * 2 - 1)
padded_spectrum = np.zeros(padded_shape)
padded_spectrum[:filtered_spectra.shape[0], :filtered_spectra.shape[1]] = filtered_spectra

fft_result = np.fft.fft2(padded_spectrum)

squared = fft_result * fft_result

acf_2d = np.fft.ifft2(squared)
acf_2d = np.abs(acf_2d)

# Normalize
norm = np.linalg.norm(filtered_spectra)
acf_2d = acf_2d / (norm ** 2)

# Crossections & Interpolation
middle_row_index = acf_2d.shape[0] // 2
middle_col_index = acf_2d.shape[1] // 2

acf_2d[middle_row_index, :] = (acf_2d[middle_row_index - 1, :] + acf_2d[middle_row_index + 1, :]) / 2
acf_2d[:, middle_col_index] = (acf_2d[:, middle_col_index - 1] + acf_2d[:, middle_col_index + 1]) / 2

horizontal_cross_section = acf_2d[middle_row_index, :]
vertical_cross_section = acf_2d[:, middle_col_index]

# Fit Gaussian
x_horizontal = np.arange(len(horizontal_cross_section))
x_vertical = np.arange(len(vertical_cross_section))

initial_guess = [np.max(horizontal_cross_section), len(horizontal_cross_section), 1.0]  # Initial guess  A, mu, sigma
params_horizontal, _ = curve_fit(gaussian, x_horizontal, horizontal_cross_section, p0=initial_guess, maxfev=100000)

initial_guess2 = [np.max(vertical_cross_section), 350, 1.0]
params_vertical, _ = curve_fit(gaussian, x_vertical, vertical_cross_section, p0=initial_guess2, maxfev=100000)

# Calculate FWHM

fwhm_horizontal = 2.35482 * params_horizontal[2]
fwhm_vertical = 2.35482 * params_vertical[2]

fwhmHalf_horizontal = params_horizontal[0] / 2
fwhmHalf_vertical = params_vertical[0] / 2

gaussian_horizontal = gaussian(x_horizontal, *params_horizontal)
gaussian_vertical = gaussian(x_vertical, *params_vertical)

aboveFWHM_horizontal = np.where(gaussian_horizontal > fwhmHalf_horizontal)[0]
aboveFWHM_vertical = np.where(gaussian_vertical > fwhmHalf_vertical)[0]

# Plotting
fig, ax = plt.subplots(2, 3, figsize=(8, 4))
gs = GridSpec(2, 3, figure=fig, wspace=0.3)

#Plot Dynamic Spectra
fig.delaxes(ax[0, 0])
fig.delaxes(ax[1, 0])
fig.delaxes(ax[0, 2])
newax = fig.add_subplot(gs[0:, 0])

newax.imshow(filtered_spectra, aspect='auto', extent=[0, timeDiff_seconds, freqMin, freqMax], vmin=np.percentile(filtered_spectra, 1), vmax=np.percentile(filtered_spectra, 98))
newax.set_xlabel('Time [s]')
newax.set_ylabel('Frequency [MHz]')
newax.set_title("J0814+7429")

# Plot Horizontal Crossection
ax[0, 1].plot(x_horizontal, horizontal_cross_section, label="Horizontal Cross-Section")
ax[0, 1].plot(aboveFWHM_horizontal, gaussian_horizontal[aboveFWHM_horizontal], label="Fitted Gaussian", linestyle="--", color='r')
ax[0, 1].set_title(f'Horizontal Cross Section, FWHM = {fwhm_horizontal:.2f}')
ax[0, 1].set_xticks([])

# Plot ACF
ax[1, 1].imshow(acf_2d, aspect='auto', extent=[-1 * timeDiff_seconds // 2, timeDiff_seconds // 2, (freqMin - freqMax) / 2, (freqMax - freqMin) / 2])
ax[1, 1].set_xlabel('Time lag [s]')
ax[1, 1].set_ylabel('Frequency lag [MHz]')

# Plot Vertical Crossection
ax[1, 2].plot(vertical_cross_section, x_vertical, label="Vertical Cross-Section")
ax[1, 2].plot(gaussian_vertical[aboveFWHM_vertical], aboveFWHM_vertical, label="Fitted Gaussian", linestyle="--", color='r')
ax[1, 2].set_title(f'Vertical Cross Section, FWHM = {fwhm_vertical:.2f}')
ax[1, 2].set_yticks([])

fig.subplots_adjust(top=0.962, bottom=0.060, left=0.037, right=0.96, hspace=0.0, wspace=0.0)
plt.show()