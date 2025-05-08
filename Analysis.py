import csv
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import detrend


rows = []


def p2f(x):
    return float(x.strip('%'))


with open('data/CPI-data.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # skipping over the headers
    next(csv_reader)

    dates = []
    all_items_change = []
    energy_change = []
    gas_change = []
    food_change = []
    shelter_change = []

    for line in csv_reader:
        dates.append(line[0])
        all_items_change.append(p2f(line[1]))
        energy_change.append(p2f(line[5]))
        gas_change.append(p2f(line[8]))
        food_change.append(p2f(line[2]))
        shelter_change.append(p2f(line[15]))


# print(dates)
print(max(all_items_change))

indices = np.arange(0, len(dates), step=24)
# print(indices)
x_dates = []
for i in indices:
    x_dates.append(dates[i])

# extracting the Great recession dates from the list of dates

# start of recession
gr_start = dates.index('Dec 2007')

# end of recession
gr_end = dates.index('June 2009')

# extracting dates of COVID-19 recession
cr_start = dates.index('Feb 2020')

cr_end = dates.index('Apr 2020')


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_all_item_change():
    x = np.arange(len(dates))
    indices = np.arange(0, len(dates), step=24)
    # print(indices)
    x_dates = []
    for i in indices:
        x_dates.append(dates[i])
    # print(x)
    print(x_dates)

    y = all_items_change

    plt.plot(y)

    # to plot the smoothed data we use the savgol-filter from scipy
    yhat = savgol_filter(y, 15, 3)

    # the line below draws the smoothed out value of the all_item change category
    plt.plot(yhat, 'r-', lw=2)

    plt.axvspan(gr_start, gr_end, color='grey', alpha=0.5)
    plt.axvspan(cr_start, cr_end, color='grey', alpha=0.5)

    plt.ylim(-5, 10)

    plt.xticks(np.arange(0, len(dates), step=24), x_dates)
    # plt.xticks(x, x_ticks_labels, size='small')


def shelter():
    x_start = dates.index('Jan 2007')
    x_end = dates.index('Dec 2009')

    x_dates = ['Jan 2007', 'Jan 2008', 'Jan 2009', 'Jan 2010']

    y = shelter_change[x_start: x_end + 1]
    print(len(y))

    plt.plot(y, label="original data")

    plt.xticks(np.arange(0, 36, step=9), x_dates, rotation=45)
    plt.title('Change in Shelter CPI during recession')
    plt.legend()
    # plt.ylim(-50, 50)


def energy():
    x = np.arange(len(dates))
    indices = np.arange(0, len(dates), step=24)
    # print(indices)
    x_dates = []
    for i in indices:
        x_dates.append(dates[i])

    # smoothing out the energy data
    energy_hat = savgol_filter(energy_change, 15, 3)

    plt.plot(energy_change, label="original data")

    # the plot below draws a smoothed out version of the energy data above
    # the current data
    plt.plot(energy_hat, 'r-', lw=2, label="smoothed data")

    plt.axvspan(gr_start, gr_end, color='grey', alpha=0.5)
    plt.axvspan(cr_start, cr_end, color='grey', alpha=0.5)

    plt.xticks(np.arange(0, len(dates), step=24), x_dates, rotation=45)
    plt.title('Change in Energy CPI in 20 years')
    plt.legend()
    plt.ylim(-50, 50)


def natural_gas():
    x = np.arange(len(dates))
    indices = np.arange(0, len(dates), step=24)
    # print(indices)
    x_dates = []
    for i in indices:
        x_dates.append(dates[i])

    # smoothing out the energy data
    gas_hat = savgol_filter(gas_change, 15, 3)

    plt.plot(gas_change, label="Original data")

    plt.plot(gas_hat, 'r-', lw=2, label="Smoothed data")

    plt.axvspan(gr_start, gr_end, color='grey', alpha=0.5)
    plt.axvspan(cr_start, cr_end, color='grey', alpha=0.5)

    plt.xticks(np.arange(0, len(dates), step=24), x_dates, rotation=45)
    plt.title('Change in Natural gas CPI in 20 years')
    plt.legend()
    plt.ylim(-50, 50)


def energy_and_gas_relation():
    # code below is from lab 4 and uses fft to compute the cross-correlation of
    # the unfiltered time series
    energy_data = np.array(energy_change).flatten()
    gas_data = np.array(gas_change).flatten()

    n = len(energy_data)

    # padding arrays with zeros
    energy_padded = np.pad(energy_data, (0, n-1), 'constant', constant_values=(0, 0))
    gas_padded = np.pad(gas_data, (0, n-1), 'constant', constant_values=(0, 0))

    # FFT of both signals
    fft_energy = fft(energy_padded)
    fft_gas = fft(gas_padded)

    # Cross-correlation using the convolution theorem
    cc_fft = ifft(fft_energy.conjugate() * fft_gas)

    # using fft shift to center the time shift/lag axis at 0
    cc_val = fftshift(cc_fft)
    n = len(energy_change)
    dt = 1
    lags = np.arange(-n + 1, n, dt)

    # cc_val = np.correlate(energy_change, gas_change, 'full')

    # in order to draw a vertical line at the max cc value
    max_index = np.argmax(np.abs(cc_val))
    max_x = lags[max_index]

    plt.plot(lags, cc_val)
    plt.axvline(x=max_x, color='r', linestyle='--')
    plt.ylabel("Cross-correlation value")
    plt.xlabel("Time lag (months)")
    plt.title("Cross-correlation between gas and energy")

    # plt.plot(res)
    # plt.axvspan(gr_start, gr_end, color='grey', alpha=0.5)
    # plt.axvspan(cr_start, cr_end, color='grey', alpha=0.5)
    #
    # plt.xticks(np.arange(0, len(dates), step=24), x_dates)
    # plt.ylim(-50, 50)


# The code above wraps the discussion of relationships between energy and gas prices

def food_vals():
    start_date = dates.index('Jan 2010')
    end_date = dates.index('Jan 2020')

    food_data = food_change[start_date: end_date]
    y = food_data

    xvals = []
    for i in range(start_date, end_date, 12):
        xvals.append(dates[i])

    plt.plot(y)
    plt.ylabel("CPI % change")
    plt.title("Change in food CPI")

    plt.xticks(np.arange(0, 120, step=12), xvals, rotation=45)


def food_trends():
    """
    This code attempts to analyze the seasonal trends in the prices of food
    over the span of nearly a decade
    """
    start_date = dates.index('Jan 2010')
    end_date = dates.index('Jan 2020')

    # we set up an array that sets up the food values between january 2011
    # and jan 2020
    food_data = food_change[start_date: end_date]

    food_data = np.array(food_data)
    food_data = detrend(food_data)

    # we now perform DFT on the food_data in order to detect if there is any
    # seasonality trends in the food prices

    Fs = 12     # sampling frequency per year
    n = len(food_data)      # aka the number of samples
    dt = 1 / Fs    # The rate at which data is sampled in a year

    yf = fft(food_data, n)
    xf = fftfreq(n, dt)
    # trend_mag = np.abs(yf) / n

    return yf, xf


def draw_food_fft():
    yf, xf = food_trends()
    Fs = 12     # sampling frequency per year
    n = len(yf)      # aka the number of samples
    dt = 1 / Fs
    if n % 2 == 0:
        positive_freqs = xf[:n//2]
        positive_fft = yf[:n//2]
    else:
        # in this case, the number of entries is odd
        positive_freqs = xf[:(n//2 + 1)]
        positive_fft = yf[:(n//2 + 1)]

    # in order to normalize the fft data, we consider only the positive
    # frequencies and multiply them by 2/N. multiply by 2 in order to preserve
    # energy that is distributed across the fft and divide by N in order to
    # normalize that result since the fft result is scaled by the number of
    # points in the original data set

    plt.stem(positive_freqs, 2/n * np.abs(positive_fft), use_line_collection=True)

    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Fourier Transform - Frequency Domain')
    plt.show()


def seasonal_patterns():
    start_date = dates.index('Jan 2010')
    end_date = dates.index('Jan 2020')

    xvals = []
    for i in range(start_date, end_date, 12):
        xvals.append(dates[i])

    # xvals now contain the dates that we would like to display on the xaxis

    yf, xf = food_trends()
    # this will generate a graph for the fourier transform as well, along with
    # returning the array of the fft values

    # the code below generates an array of boolean values and the indexes
    # that contain the value true are those that have a value of
    seasonal_freqs = (xf > 1) & (xf < 2)
    # print(seasonal_freqs)

    # we create a frequency mask that zeros out all frequencies that are
    # not between the range 1 and 2 as we are only interested in the
    # annual and bi-annual patterns of seasonality
    freq_mask = np.zeros_like(yf)
    freq_mask[seasonal_freqs] = yf[seasonal_freqs]

    # freq_mask now contains the indexes of the peaks corresponding to seasonal
    # patterns thereby isolating the frequency components associated with
    # seasonality

    # we now apply ifft to extract the seasonal pattern in the time domain
    seasonal_pattern = ifft(freq_mask)

    # taking the real part of the ifft output
    seasonal_pattern_real = np.real(seasonal_pattern)

    # print(seasonal_pattern_real)
    plt.plot(seasonal_pattern_real)
    plt.title("Seasonal patterns in the Food CPI")

    xticks = np.arange(0, 120, step=12)

    plt.xticks(xticks, xvals, rotation=45)


if __name__ == "__main__":
    # plot_all_item_change()
    # print(dates)
    # energy()
    # natural_gas()
    # energy_and_gas_relation()
    # shelter()
    # food_vals()
    # draw_food_fft()
    seasonal_patterns()
