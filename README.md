# MSWEP_TW Python scripts

Welcome to the repository for the development of **M**ulti-**S**ource **W**eighted-**E**nsemble **P**recipitation for **T**ai**W**an.

In this repository, you will find:

- Application of ConvLSTM to the correction of the satellite precipitation products (i.e., the Early Run and Final Run of Integrated Multi-Satellite Retrievals for Global Precipitation Measurement (IMERG)), in the "ConvLSTM" folder;
- Application of LSTM to the correction of IMERG, in the "LSTM" folder; and
- Merging gauge-, satellite-, and model-based precipitation for Taiwan using the MSWEP technique, in the "MSWEP" folder.

## Documentation

A peer-reviewed study is currently under review in Journal of Hydrology (see _Using and citing this work_ below).

In each folder, we have included a README file that further describes the code, and sample input and output files. 

## Using and citing this work

Please note that this work is under a GPL-3.0 License.

If you're using this code or any parts of it, please cite the following study:

  Kao, Y.C & Chen, C.J. (2023).
  _Development of Multi-Source Weighted-Ensemble Precipitation: Influence of Bias Correction based on Recurrent Convolutional Neural Networks_ Journal of Hydrology, in review.

## Reporting bugs

Please make sure to revise the I/O paths in the scripts before running them.
We will update the scripts to make them more versatile in the future; meanwhile if you find any bugs, please raise an issue on GitHub and/or send me an email to [cjchen@nchu.edu.tw](mailto:cjchen@nchu.edu.tw).