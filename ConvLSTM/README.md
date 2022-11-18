# Using ConvLSTM for correcting satellite precipitation

In this folder you can find Python script "ConvLSTM.py," prepared for the correction of IMERG data for Taiwan.

## Sample input files in the "input" folder

test_in_1 is the sample data (IMERG) to be corrected. Each column contains the uncorrected precipitation time series for a grid cell (323 cells in total).
test_in_2 is the reference ground truth (TCCIP) for training.  Each column contains the TCCIP time series for a grid cell (323 cells in total).

## Sample output file in the "output" folder

Each column contains the corrected precipitation time series for a grid cell in 2019 (testing period).

## Reporting bugs

Please make sure to revise the I/O paths in the script before running it.
We will update the script to make them more versatile in the future; meanwhile if you find any bugs, please raise an issue on GitHub and/or send me an email to [cjchen@nchu.edu.tw](mailto:cjchen@nchu.edu.tw).