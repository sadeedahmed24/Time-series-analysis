import csv
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftshift


def p2f(x):
    return float(x.strip('%'))


with open('data/Unemployment-Data.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # skipping over the headers
    next(csv_reader)

    dates = []
    unemployment_rate = []

    for line in csv_reader:
        dates.append(line[0])
        unemployment_rate.append(p2f(line[1]))


# print(dates)
print(max(unemployment_rate))


# recession_1 window; We consider the times before the start of the recession
# and after the end of the recession to capture a fuller image of the
# recession's economic impact
gr_start = dates.index('Jan 2007')

gr_end = dates.index('Oct 2009')


def shelter_impulse_response(t, decay_rate=0.5):
    """This function generates a hypothetical impulse response for the Shelter CPI"""
    return np.exp(-decay_rate * t) * (t > 0)


def recession_impulse():
    """The function below returns the recession impulse array"""
    y = unemployment_rate[gr_start: gr_end + 1]


    # print(y)
    y = np.array(y)
    # this normalizes the y - values

    y_n = 10  # Highest impact point of the recession impulse
    n = len(y)
    m = n + 46  # Extend by 46 units
    x = np.arange(n, m+1)

    a = (1 - y_n) / ((m - n) ** 2)

    padded_vals = a * (x - n) ** 2 + y_n

    y = np.append(y, padded_vals)

    y = y / np.abs(max(y))

    # plt.plot(y)
    # plt.title("Onset of recession (unemployment rate)")

    # x = np.arange(len(dates))
    indices = np.arange(34, len(y), step=12)
    # print(indices)
    x_dates = []
    for i in indices:
        x_dates.append(dates[i])

    plt.xticks(np.arange(0, 80, step=20), x_dates, rotation=45)

    # generating a time series for the impulse response
    time = np.linspace(0, 80, 80)

    response = shelter_impulse_response(time)
    # print(response)

    # using the recession impulse we perform a convolution to study the
    # impulse response of the shelter CPI
    convoluted_response = np.convolve(y, response, mode='full')[:len(time)]

    # print(convoluted_response)
    plt.plot(time, convoluted_response, label='Modeled Shelter CPI Response')
    plt.title("Impulse response of shelter CPI to the recession Impulse")

    return y


if __name__ == "__main__":
    recession_impulse()




