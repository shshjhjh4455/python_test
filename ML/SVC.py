import pandas as pd 
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


x, lables = read_data("data/points_class_1.txt")