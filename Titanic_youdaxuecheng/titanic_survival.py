#! /usr/bin/env python2
# -*- conding:utf-8 -*-

from sys import version_info
from titanic_visualizations import survival_stats
from IPython.display import display
import numpy as np
import pandas as pd

file = 'titanic_data.csv'
data = pd.read_csv(file)
display(data.head())

