# US CPI Time Series Analysis 🧮📈

This repository contains the code and final report for my PHY408 project at the University of Toronto, titled **"Time Series Analysis of the U.S. Consumer Price Index (CPI)"**. The goal of this project was to explore relationships and patterns in various CPI components using signal processing techniques, statistical modeling, and Python-based data analysis.

## 📄 Report

The full analysis and results are documented in [`Ahmed_Sadeed_PHY408Report.pdf`](Ahmed_Sadeed_PHY408Report.pdf).

## 📊 Project Summary

Using publicly available data from the U.S. Bureau of Labor Statistics, this project investigates three major questions:

1. **Correlation between Natural Gas and Energy CPI**  
   - Uses smoothing (Savitzky-Golay), stationarity testing, and FFT-based cross-correlation.
   - Finding: Gas CPI leads Energy CPI with a lag of ~2 months.

2. **Impulse Response of Shelter CPI to Recession**  
   - Models recessions using unemployment rates as proxy impulses.
   - Convolution-based response shows housing price sensitivity to economic downturns.

3. **Seasonality in Food CPI**  
   - Detrends data and uses DFT to extract periodic patterns.
   - Clear seasonal highs in summer and lows at year-end are identified.

## 🧪 Technologies Used

- Python
- NumPy, SciPy, Matplotlib
- Signal Processing (Convolution, FFT)
- Web scraping (for BLS data extraction)
- Time series and economic data modeling

## 📁 File Structure
- analysis.py (Gas-Energy correlation and Food CPI seasonality analysis)
- recession_impact.py (Impulse response modeling of Shelter CPI during recessions)
- Ahmed_Sadeed_PHY408Report.pdf (Final project report (PDF))
- data/ (Directory for CPI and unemployment datasets (not included))

## 📚 Data Sources

- [CPI by Category (BLS)](https://www.bls.gov/charts/consumer-price-index/consumer-price-index-by-category-line-chart.htm)
- [Unemployment Rate (BLS)](https://www.bls.gov/charts/employment-situation/civilian-unemployment-rate.htm)

## 🚀 How to Run

Make sure you have Python 3 installed along with the following packages:

```bash
pip install numpy scipy matplotlib
```
Then run the scripts individually depending on which part you'd like to explore. I ran the files in pyCharm as that's my editor of choice when editing/running python files.
