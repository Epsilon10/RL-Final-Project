The `mbrl` directory contains all the code needed to run the model based RL experiments.
The `mfrl` directory contains all the code needed to run the model free RL experiments.

The `mfrl` directory contains one file per model-free baseline containing its implementation.
The files are named by the algorithm they implement. There is an executable in runner.py that
runs all of the baseline experiments and logs the results. One can deploy multiple experiments
at the same time by changing the `--parallel-procs` parameter.
