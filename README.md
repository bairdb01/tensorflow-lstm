# tensorflow-lstm
An LSTM created in tensorflow to predict the temperature at various arenas. In this situation it is predicting the next temperature at the
Capital Center based on the data available.

## To run
py lstm.py [arenas.csv]

An optional first argument is used to specify the CSV file to load the data from. Defaults to arenas.csv.

## Example Output: 
“The temperature at the Capital Center will be: 44.57337951660156”

## Data Formatting
- Normalized temperatures to be between 0 and 1 by dividing by 300 to represent possible values of extremes of temperatures (-150F to 150F)
- Resampled data from a sparse time series with no inherent frequency to have daily frequency
- Interpolated the temperatures using a method which places data points linearly into the missing dates
