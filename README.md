# Online Non-linear Prediction of Financial Time Series Patterns

This repository contains the code libraries and document for my dissertation, "Online Non-linear Prediction of Financial Time Series Patterns"

## Abstract

We consider a mechanistic non-linear machine learning approach to learning signals in financial time series data. A modularised and decoupled algorithm framework
is established and is proven on daily sampled closing time-series data for JSE equity
markets. The input patterns are based on input data vectors of data windows preprocessed into a sequence of daily, weekly and monthly or quarterly sampled feature
measurement changes (log feature fluctuations). The data processing is split into a
batch processed step where features are learnt using a Stacked AutoEncoder (SAE) via
unsupervised learning, and then both batch and online supervised learning are carried
out on Feedforward Neural Networks (FNN) using these features. The FNN output is
a point prediction of measured time-series feature fluctuations (log differenced data)
in the future (ex-post). Weight initializations for these networks are implemented
with restricted Boltzmann machine pre-training, and variance based initializations.
The validity of the FNN backtest results are shown under a rigorous assessment of
backtest overfitting using both Combinatorially Symmetrical Cross Validation and
Probabilistic and Deflated Sharpe Ratios. Results are further used to develop a view
on the phenomenology of financial markets and the value of complex historical data
under unstable dynamics.

Main document: https://github.com/joel11/Masters/blob/master/Thesis/JoeldaCosta_Thesis.pdf

## Data

The dataset used can be found at: https://zivahub.uct.ac.za/account/articles/11897628

DOI: 10.25375/uct.11897628

## Code Libraries Tutorial 

All libraries can be found in the following directory: https://github.com/joel11/Masters/tree/master/Code%20Libraries

The Main Tutorial module is made available to provide a step by step walkthrough
for using the provided libraries. Configurations and methods are detailed
for some naive training on an AGL dataset. It should
be noted that these are not necessarily the optimal parameter choices possible
for this exercise.

### Create the Database
The first step is to create the database which will be used with the CreateDatabase
function. Once created, the DatabaseOps module should be updated so
that the db variable references the same database by name.

### Data Specifications
The JSE dataset can be read into memory using ReadJSETop40Data, and particular
assets can then be filtered on (as is done for AGL). The data specifications
such as horizons, prediction points and segregation points are also set in
this step.

### Train SAE Networks
Once the database has been created, the SAE networks can be trained. This
step uses the RunSAEExperiment method for training SAE networks, passing
through the configurations chosen and noted in Main Tutorial. The
configurations specify the parameters for the SAE networks and their SGD
training, as detailed in Software Section 6 of the accompanying thesis document.

### Select Best SAE Networks
The selection of SAE networks is done semi-manually, and is dependent on the
task at hand. In this case, the provided GetBestSAE function filters according
to data horizons specified. This may differ according to what is being investigated.
As the case may be, the selection of all `best' SAE networks should be
specified by configuration id in the sae choices vector.

### Train FFN Networks
Much like the SAE networks, the FFN networks can be trained in a combinatorial
manner using the noted configuration choices and the RunFFNExperiment
function. These are available as specified in Section 6 of the accompanying thesis document.

### Run Batch Process Diagnostics
The FFN training will include the recording of IS and OOS predictions for
the specified data. This allows the running of the RunBatchTradeProcess and
RunBatchAnalyticsProcess functions from the DatabaseBatchProcesses module. These processes record the per trade returns, as well as the aggregated
performance metrics such as Sharpe ratios or strategy P&L.

### Diagnostic Visualizations
Once the aggregated performance metrics have been calculated and recorded,
they can be used for diagnostic visualizations. A set of example visualizations
have been included in the Main Tutorial module, though any from the
full set can be used, which are specified in Appendix 11.6.

### CSCV & PBO
Subsequently, the CSCV analysis can be run using ExperimentCSCVProcess. The set of configuration ids for the FFN networks are specified, with
the split values to be run. The method will save the logit distribution graph
in the GraphOutputDir directory and return the PBO score.

### DSR
The DSR implementation is split across the Julia and Python code base. In
Julia, the WriteCSVFileForDSR method will write out the required return data
for the specified FFN configurations. Using the Python module DSR proc, the
same CSV file can be specified and used to run the DSR analysis.
