# Test for checking python env setup

# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import datasets, plotting
from nilearn.connectome import ConnectivityMeasure
from sklearn import (
    datasets,
    linear_model,
    metrics,
    model_selection,
    preprocessing,
)
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier

print("")
print("**********************************************************")
print("You have successfully installed all the required packages!")
print("**********************************************************")
