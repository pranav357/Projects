from arviz import plot_posterior
from pymc3 import sample
from pymc3 import Model, Normal, Uniform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wget
# Download data
wget.download(
    "https://raw.githubusercontent.com/fonnesbeck/mcmc_pydata_london_2019/master/data/radon.csv")


radon = pd.read_csv('./radon.csv', index_col=0)
radon.head()

anoka_radon = radon.query('county == "ANOKA"').log_radon
sns.distplot(anoka_radon, bins=16)

plt.axvline(1.1)
plt.show()

# Bayesian Workflow
# Specify a prob model
# Assign the most probable probability function for everything

# Calculate posterior distribution


# Defining a Bayesian Model
# Define Radon's Bayesian model with mu and sigma and these parameters will be modeled with a distribution of choice


# Defining a Bayesian model
with Model() as radon_model:
    mu1 = Normal('mu1', mu=0, sd=10)
    sigma1 = Uniform('sigma1', 0, 10)

# Compile the radon model with the initialised probability distribution
with radon_model:
    dist = Normal('dist', mu=mu1, sd=sigma1, observed=anoka_radon)

# Model fitting with data

with radon_model:
    samples = sample(1000, tune=1000, cores=2, random_seed=12)

# Plot how parameter mu is distributed

plot_posterior(samples, var_names=['mu1'], ref_val=1.1)
