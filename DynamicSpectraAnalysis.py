import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import ndimage
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter
from scipy import signal
from scipy import datasets
import scipy.fft
import re
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser()

parser.add_argument("file_name", type=str, help="Name of file to generate dynamic spectra from")
# parser.add_argument("signal_peak", type=int, help="Value where the signal peaks")

args = parser.parse_args()

"""def median_filter(spectra):
    margin = 1
    filtered = np.zeros_like(spectra)
    for i in range(len(filtered)):
        for j in range(len(filtered[0])):
            if (i == 0 and j == 0):
                selectTopLeftCorner = dynamic_spectr[i:i + margin + 1, j:j + margin + 1]
                filtered[i, j] = np.median(selectTopLeftCorner)
                continue
            if (i == 0 and j == len(filtered[0]) - 1):
                selectTopRightCorner = dynamic_spectr[i:i + margin + 1, j - margin:j + margin]
                filtered[i, j] = np.median(selectTopRightCorner)
                continue
            if (i == len(filtered) - 1 and j == 0):
                selectBottomLeftCorner = dynamic_spectr[i - margin:i + margin, j:j + margin + 1]
                filtered[i, j] = np.median(selectBottomLeftCorner)
                continue
            if (i == len(filtered) - 1 and j == len(filtered[0]) - 1):
                selectBottomRightCorner = dynamic_spectr[i - margin:i + margin, j - margin:j + margin]
                filtered[i, j] = np.median(selectBottomRightCorner)
                continue
            if (j == 0):
                selectionLeftSide = dynamic_spectr[i - margin:i + margin + 1, j:j + margin + 1]
                filtered[i, j] = np.median(selectionLeftSide)
                continue
            if (j == len(filtered[0]) - 1):
                selectionRightSide = dynamic_spectr[i - margin:i + margin + 1, j - margin:j + margin]
                filtered[i, j] = np.median(selectionRightSide)
                continue
            if (i == 0):
                selectionTopSide = dynamic_spectr[i:i + margin + 1, j - margin:j + margin + 1]
                filtered[i, j] = np.median(selectionTopSide)
                continue
            if (i == len(filtered) - 1):
                selectionBottomSide = dynamic_spectr[i - margin:i + margin, j - margin:j + margin + 1]
                filtered[i, j] = np.median(selectionBottomSide)
                continue
            selection = dynamic_spectr[i - margin:i + margin + 1, j - margin:j + margin + 1]
            filtered[i, j] = np.median(selection)
    return filtered"""


def median_filter(spectra, margin=7):
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
# bin_min = args.signal_peak - 50
# bin_max = args.signal_peak + 50
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
        # if bin_min < int(columns[2]) < bin_max:
        #    continue
        data.append(value)

freqMax = float(freqMax.group(1))
freqMin = float(freqMin.group(1))
timeMax = float(timeMax.group(1))
timeMin = float(timeMin.group(1))
timeDiff_seconds = (timeMax - timeMin) * 24 * 60 * 60

dynamic_spectr = np.flipud(dynamic_spectr.T)

filtered_spectra = median_filter(dynamic_spectr, 3)
# filtered_spectra = gaussian_filter(dynamic_spectr, sigma=1)


# FT to obtain ACF
# acf_2d = signal.correlate2d(filtered_spectra, filtered_spectra, mode='full')

padded_shape = (len(filtered_spectra) * 2 - 1, len(filtered_spectra[1]) * 2 - 1)
padded_spectrum = np.zeros(padded_shape)
padded_spectrum[:filtered_spectra.shape[0], :filtered_spectra.shape[1]] = filtered_spectra

fft_result = np.fft.fft2(padded_spectrum)

squared = fft_result * fft_result
# squared = np.fft.fftshift(squared)

acf_2d = np.fft.ifft2(squared)
# acf_2d = np.fft.fftshift(acf_2d)
acf_2d = np.abs(acf_2d)

# Normalize
norm = np.linalg.norm(filtered_spectra)
acf_2d = acf_2d / (norm ** 2)

# Crossections
middle_row_index = acf_2d.shape[0] // 2
middle_col_index = acf_2d.shape[1] // 2

# Interpolation
acf_2d[middle_row_index, :] = (acf_2d[middle_row_index - 1, :] + acf_2d[middle_row_index + 1, :]) / 2
acf_2d[:, middle_col_index] = (acf_2d[:, middle_col_index - 1] + acf_2d[:, middle_col_index + 1]) / 2

horizontal_cross_section = acf_2d[middle_row_index, :]
vertical_cross_section = acf_2d[:, middle_col_index]

# Generate x values for fitting
x_horizontal = np.arange(len(horizontal_cross_section))
x_vertical = np.arange(len(vertical_cross_section))

# Fit Gaussian
initial_guess = [np.max(horizontal_cross_section), len(horizontal_cross_section), 1.0]  # Initial guess  A, mu, sigma
params_horizontal, _ = curve_fit(gaussian, x_horizontal, horizontal_cross_section, p0=initial_guess, maxfev=100000)

# initial_guess2 = [np.max(vertical_cross_section), len(vertical_cross_section), 1.0]
initial_guess2 = [np.max(vertical_cross_section), 350, 1.0]
params_vertical, _ = curve_fit(gaussian, x_vertical, vertical_cross_section, p0=initial_guess2, maxfev=100000)

# Calculate FWHM

fwhm_horizontal = 2.35482 * params_horizontal[2]
fwhm_vertical = 2.35482 * params_vertical[2]
print("FWHM horizontal - ", fwhm_horizontal)
print("FWHM vertical - ", fwhm_vertical)

fwhmHalf_horizontal = params_horizontal[0] / 2
fwhmHalf_vertical = params_vertical[0] / 2

gaussian_horizontal = gaussian(x_horizontal, *params_horizontal)
gaussian_vertical = gaussian(x_vertical, *params_vertical)

aboveFWHM_horizontal = np.where(gaussian_horizontal > fwhmHalf_horizontal)[0]
aboveFWHM_vertical = np.where(gaussian_vertical > fwhmHalf_vertical)[0]

# Plotting
fig, ax = plt.subplots(2, 3, figsize=(8, 4))
gs = ax[1, 2].get_gridspec()

fig.delaxes(ax[0, 0])
fig.delaxes(ax[1, 0])
fig.delaxes(ax[0, 2])
newax = fig.add_subplot(gs[0:, 0])

newax.imshow(filtered_spectra, aspect='auto', extent=[0, timeDiff_seconds, freqMin, freqMax],
             vmin=np.percentile(filtered_spectra, 1), vmax=np.percentile(filtered_spectra, 98))
newax.set_xlabel('Time [s]')
newax.set_ylabel('Frequency [MHz]')
newax.set_title("J1509+5531 Median filter")
# Plot in the second subplot
ax[0, 1].plot(x_horizontal, horizontal_cross_section, label="Horizontal Cross-Section")
# ax[0, 1].plot(x_horizontal, gaussian(x_horizontal, *params_horizontal), label="Fitted Gaussian", linestyle="--")
ax[0, 1].plot(aboveFWHM_horizontal, gaussian_horizontal[aboveFWHM_horizontal], label="Fitted Gaussian", linestyle="--",
              color='r')
ax[0, 1].set_title(f'Horizontal Cross-Section, FWHM = {fwhm_horizontal}')
ax[0, 1].set_xticks([])

# Plot in the first subplot
ax[1, 1].imshow(acf_2d, aspect='auto',
                extent=[-1 * timeDiff_seconds // 2, timeDiff_seconds // 2, freqMin - freqMax, freqMax - freqMin])
ax[1, 1].set_xlabel('Time lag [s]')
# ax[1, 1].set_ylabel('Frequency lag [MHz]')
ax[1, 1].set_yticks([])

# Plot in the third subplot
ax[1, 2].plot(vertical_cross_section, x_vertical, label="Vertical Cross-Section")
# ax[1, 2].plot(gaussian(x_vertical, *params_vertical), x_vertical, label="Fitted Gaussian", linestyle="--")
ax[1, 2].plot(gaussian_vertical[aboveFWHM_vertical], aboveFWHM_vertical, label="Fitted Gaussian", linestyle="--",
              color='r')
ax[1, 2].set_title(f'Vertical Cross-Section, FWHM = {fwhm_vertical}')
ax[1, 2].set_yticks([])
# ax[1, 2].yaxis.set_label_position("right")
# ax[1, 2].set_ylabel('Frequency lag [MHz]')

fig.subplots_adjust(top=0.962, bottom=0.060, left=0.037, right=0.96, hspace=0.0, wspace=0.0)
plt.show()