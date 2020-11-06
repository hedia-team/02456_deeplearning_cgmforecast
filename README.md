# 02456_deeplearning_cgm

# Installation
Using the repo requires installation of `poetry`. Guide to installation can be found [here](https://python-poetry.org/docs/#installation)
This tool manages all the required python packages in a virtual environement. After installation run
`poetry install`from the main directory of the repo.

# Scripts
The repo consist of three main scripts.
* `example_script.py`shows how a model can be defined and trained, data can be extracted and how the model evaluation work. Yeah, basically, it just shows the different key functions work.
* `optmizeHypers.py` searches for the best hyperparameters given a model and a searching area. You can play around with the searching area searching technqiues to find even better model configurations. 
* `evaluateAll.py` find the best hyperparameters and evalautes the results on a user defined set of data.

The repo also consist of a bunch of helper functions found in `/src/` that load data, evaluate models and so on. 

Finally, one important function is `train_cgm` that id defined in `./src/tools.py`. This function carries out the training of a given model. You are free to change how the training is carried out if you have some good idea. (Which is of course also the case the for any part of the repo).

## IMPORTANT
Remember to load the correct model in the scripts. That is, if you create a new model architecture and save it in `./src/models/myNet.py`, you should change which model is imported in all scripts as well as in `train_cgm`.


# Execution
If you have installed poetry properly, you should be able to run the scripts from the terminal using `poetry run python3 example_script.py`

