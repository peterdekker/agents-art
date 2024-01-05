# ART model
## Installation
Install the required Python packages using
```pip3 install -r requirements.txt```

## Run
Run ```python3 pipeline.py``` in the command line, with one of the following options:
``--single_run_plotdata``: Perform single run of ART model, with fixed vigilance and batch size. Plots data with gold inflection classes and model clustering
``--eval_batches``: Evaluate ART model performance for different batch sizes
``--eval_vigilances``: Evaluate ART model performance for different vigilances
``--baseline``: Run two baselines: majority and random baseline