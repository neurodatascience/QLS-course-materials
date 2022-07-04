# Test for checking python env setup

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets, linear_model, model_selection, metrics
from sklearn.dummy import DummyRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting

print('')
print('**********************************************************')
print('You have successfully installed all the required packages!')
print('**********************************************************')
