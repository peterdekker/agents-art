# Code for _Learning Inflection Classes using Adaptive Resonance Theory_
This repository contains the code for the thesis chapter _Learning Inflection Classes using Adaptive Resonance Theory_ from the PhD thesis of Peter Dekker. The code has been developed by Peter Dekker and Heikki Rasilo and conceptual contributions have been made by Bart de Boer.
## Installation
Install the required Python packages using
```pip3 install -r requirements.txt```

## Run
Run ```python3 pipeline.py``` on the command line, with one of the following modes:
 - ``--single_run``: Perform run of ART model using specific vigilance value, which is set using the `--vigilance_single_run` parameter. For example: 
``python3 pipeline.py --single_run --vigilance_single_run 0.2 --language portuguese``
 - ``--eval_vigilances``: Evaluate ART model performance for range of vigilances.
Next to these two main modes, two additional modes can be given in combination with one of the two main modes:
 - ``--baseline``: Run two baselines: majority and random baseline. This baseline mode can be given at the same time with the other modes, to evaluate both the ART model and the baselines.
 - ``--train_test``: Run the given mode (single run or evaluation of vigilances) in train-test mode, where the model is trained on one part of the data and evaluated on another. In this mode, best to set `--n_runs` to a lower value (e.g. 1), because this mode already creates 10 model runs per setting using 10-fold cross-validation. For example: `python3 pipeline.py --eval_vigilances --n_runs 1 --train_test`

Additionally the options `--language` can be used to set the language (`latin` (default), `portuguese` or `estonian`) and `--n_runs` is used to set the number of model runs averaged over (default: 10). More advanced options, set via the code, can be found in the source file `conf.py`. 

