import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import os

owd = os.getcwd()

csv_files = []
os.chdir('cuisine_recipe_ingredient_CSV')
for name in glob.glob('*.csv'):
    csv_files.append(name)

for csvs in csv_files:
    

