# Try out various ML algos on the data to see what works best
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 60)

# Set default font size
plt.rcParams['font.size'] = 24
import pickle

# Seaborn for visualization
sns.set(font_scale=2)

# Imputing missing values and scaling values
# Imputing - ML models cant handle missing data

# Read in data into dataframes
train_features = pd.read_csv('training_features.csv')
test_features = pd.read_csv('testing_features.csv')
train_labels = pd.read_csv('training_labels.csv')
test_labels = pd.read_csv('testing_labels.csv')

figsize(8, 8)

# Histogram of the Energy Star Score
plt.style.use('fivethirtyeight')
plt.hist(train_labels['score'].dropna(), bins=100)
plt.xlabel('Score')
plt.ylabel('Number of Buildings')
plt.title('ENERGY Star Score Distribution')
plt.show()

# Impute using a simple method called median imputation

# Create an imputer object with median filling strategy
imputer = SimpleImputer(strategy='median')

# Train in train dataset
imputer.fit(train_features)

# Transform or fill using it on both the training and test dataset
X = imputer.transform(train_features)
X_test = imputer.transform(test_features)
# This is done to avoid test data leakage

# Convert y to 1D array
y = np.array(train_labels).reshape((-1,))
y_test = np.array(test_labels).reshape((-1,))

print('Missing values in training features: ', np.sum(np.isnan(X)))
print('Missing values in testing features:  ', np.sum(np.isnan(X_test)))

# Ensure all values are finite
print(np.where(~np.isfinite(X)))
print(np.where(~np.isfinite(X_test)))

# Feature Scaling
# Changing the range of a feature, required as various features are present in various ranges and can impact measurements
# Especially affects SVM and KNN, but do it as good practice for all

# Create scaler object with range 0-1
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on training data
scaler.fit(X)

# Transform or fill using it on both the training and test dataset
X = scaler.transform(X)
X_test = scaler.transform(X_test)
# This is done to avoid test data leakage as before

# Implementing ML models in SK learn
# Function to calculate mean absolute error


def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

# Takes in a model, trains the model, and evaluates the model on the test set


def fit_and_evaluate(model):

    # Train the model
    model.fit(X, y)

    # Make predictions and evaluate
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)

    # Return the performance metric
    return model_mae


# Linear Regression
lr = LinearRegression()
lr_mae = fit_and_evaluate(lr)

print('Linear Regression Performance on the test set: MAE = %0.4f' % lr_mae)

# SVM
svm = SVR(C=1000, gamma=0.1)
svm_mae = fit_and_evaluate(svm)

print('Support Vector Machine Regression Performance on the test set: MAE = %0.4f' % svm_mae)

# Random Forest
random_forest = RandomForestRegressor(random_state=60)
random_forest_mae = fit_and_evaluate(random_forest)

print('Random Forest Regression Performance on the test set: MAE = %0.4f' %
      random_forest_mae)

# Gradient Boosting
gradient_boosted = GradientBoostingRegressor(random_state=60)
gradient_boosted_mae = fit_and_evaluate(gradient_boosted)

print('Gradient Boosted Regression Performance on the test set: MAE = %0.4f' %
      gradient_boosted_mae)

# KNN Regressor
knn = KNeighborsRegressor(n_neighbors=10)
knn_mae = fit_and_evaluate(knn)

print('K-Nearest Neighbors Regression Performance on the test set: MAE = %0.4f' % knn_mae)

# Plot all performances
plt.style.use('fivethirtyeight')
figsize(8, 6)

# Dataframe to hold results
model_comparison = pd.DataFrame({'model': ['Linear Regression', 'Support Vector Machine',
                                           'Random Forest', 'Gradient Boosted',
                                           'K-Nearest Neighbors'],
                                 'mae': [lr_mae, svm_mae, random_forest_mae,
                                         gradient_boosted_mae, knn_mae]})

# Horizontal bar chart of test mae
model_comparison.sort_values('mae', ascending=False).plot(x='model', y='mae', kind='barh',
                                                          color='red', edgecolor='black')

# Plot formatting
plt.ylabel('')
plt.yticks(size=14)
plt.xlabel('Mean Absolute Error')
plt.xticks(size=14)
plt.title('Model Comparison on Test MAE', size=20)
plt.show()

# Work on hyperparamter tuning to improve model performance
# To strike balance between under and overfitting

# Use Random Search and Cross Validation
# Random Search - Define a grid and randomly sample different combinations
# Cross Validation - Technique used to evaluate the selected comnbination of hyperparameters
# We first define a grid then peform an iterative process of: randomly sample a set of hyperparameters from the grid,
# evaluate the hyperparameters using 4-fold cross-validation, and then select the hyperparameters with the best performance.

# Loss function to be minimised
loss = ['ls', 'lad', 'huber']

# Number of trees to be used
n_estimators = [100, 500, 900, 1100, 1500]

# Maximum depth of each tree
max_depth = [2, 3, 5, 10, 15]

# Min no of examples required at each leaf node
min_samples_leaf = [1, 2, 4, 6, 8]

# Minimum number of examples to split a node
min_samples_split = [2, 4, 6, 10]

# Maximum number of features to consider for making splits
max_features = ['auto', 'sqrt', 'log2', None]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'loss': loss,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}

# Create the model to use for hyperparameter tuning
model = GradientBoostingRegressor(random_state=42)

# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=hyperparameter_grid,
                               cv=4, n_iter=25,
                               scoring='neg_mean_absolute_error',
                               n_jobs=-1, verbose=1,
                               return_train_score=True,
                               random_state=42)

# scoring: which metric to use when evaluating candidates
# n_jobs: number of cores to run in parallel (-1 will use all available)
# verbose: how much information to display (1 displays a limited amount)

# Fit on the training data
random_cv.fit(X, y)

# Get all results and save in a dataframe
random_results = pd.DataFrame(random_cv.cv_results_).sort_values(
    'mean_test_score', ascending=False)
random_results.head()

random_cv.best_estimator_
# Gives us idea of best parameters
# Can further fine tune by creating a grid search of nearest best values that worked
# Evaluate based on single one: n_estimators

# Create number of trees to evaluate
trees_grid = {'n_estimators': [
    100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}

model = GradientBoostingRegressor(loss='lad', max_depth=5,
                                  min_samples_leaf=6,
                                  min_samples_split=6,
                                  max_features=None,
                                  random_state=42)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator=model, param_grid=trees_grid, cv=4,
                           scoring='neg_mean_absolute_error', verbose=1,
                           n_jobs=-1, return_train_score=True)

# Fit the grid search
grid_search.fit(X, y)

# Get the results into a dataframe
results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
figsize(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_n_estimators'], -1 *
         results['mean_test_score'], label='Testing Error')
plt.plot(results['param_n_estimators'], -1 *
         results['mean_train_score'], label='Training Error')
plt.xlabel('Number of Trees')
plt.ylabel('Mean Abosolute Error')
plt.legend()
plt.title('Performance vs Number of Trees')
plt.show()
# Does seem like overfitting especially due to the larger difference between train and test errors

# Evaluate Model on test set
# Default model
default_model = GradientBoostingRegressor(random_state=42)

# Select the best model
final_model = grid_search.best_estimator_

final_model

default_model.fit(X, y)

final_model.fit(X, y)

default_pred = default_model.predict(X_test)
final_pred = final_model.predict(X_test)

print('Default model performance on the test set: MAE = %0.4f.' %
      mae(y_test, default_pred))
print('Final model performance on the test set:   MAE = %0.4f.' %
      mae(y_test, final_pred))
# Final Model does outperform baseline model

figsize(8, 8)

# Density plot of the final predictions and the test values
sns.kdeplot(final_pred, label='Predictions')
sns.kdeplot(y_test, label='Values')

# Label the plot
plt.xlabel('Energy Star Score')
plt.ylabel('Density')
plt.title('Test Values and Predictions')
plt.show()
# The distribution looks to be nearly the same although the density of the predicted values is closer to the median of the test values rather than to the actual peak at 100.
# It appears the model might be less accurate at predicting the extreme values and instead predicts values closer to the median.

# HIstogram of residuals - should be normally distributed
figsize = (6, 6)

# Calculate the residuals
residuals = final_pred - y_test

# Plot the residuals in a histogram
plt.hist(residuals, color='red', bins=20,
         edgecolor='black')
plt.xlabel('Error')
plt.ylabel('Count')
plt.title('Distribution of Residuals')
plt.show()

#Save final model
pkl_filename = "model/xgb_final_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(final_model, file)


