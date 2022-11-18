# Using LSTM for correcting satellite precipitation

In this folder you can find Python scripts "LSTM_loop(1-1).py~LSTM_loop(9-1).py," corresponding to the 1-to-1 to maximally 9-to-1 input-output cell correspondence (i.e., spatial coherence scheme described in the paper).

IMERG is the data to be corrected.
TCCIP is the reference ground truth for training.

## Sample input file format in the "input" folder

LSTM_loop(1-1):<test_input_1-1.csv>
year-month-day-IMERG-TCCIP
LSTM_loop(2-1):<test_input_2-1.csv>
year-month-day-IMERG-IMERG (surrounding grid 1)-TCCIP
LSTM_loop(3-1):<test_input_3-1.csv>
year-month-day-IMERG-IMERG(surrounding grid 1)-IMERG(surrounding grid 2)-TCCIP
...
(Data from surrounding grids are not in a specific order)

## Sample output file in the "output" folder

Correlations between IMERG vs. TCCIP for all the number of grid-cell combinations (e.g., 256 for 9-to-1) in 2019 (testing period)

## Reporting bugs

Please make sure to revise the I/O paths in the scripts before running them.
We will update the scripts to make them more versatile in the future; meanwhile if you find any bugs, please raise an issue on GitHub and/or send me an email to [cjchen@nchu.edu.tw](mailto:cjchen@nchu.edu.tw).