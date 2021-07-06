# ML models explanibility
# Gradient boosting falls in the middle, as its made up of 100s of explainable decision trees
# We are going to use three methods to explain the model:
# Feature importances
# Visualizing a single decision tree
# LIME: Local Interpretable Model-Agnostic Explainations

# Pandas and numpy for data manipulation
import lime
import lime.lime_tabular
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.core.algorithms import mode

# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 60)

# Matplotlib for visualization
%matplotlib inline

# Set default font size
plt.rcParams['font.size'] = 24


# Seaborn for visualization

sns.set(font_scale=2)

# Imputing missing values

# Machine Learning Models


# LIME for explaining predictions

# Read in data into dataframes
train_features = pd.read_csv('data/training_features.csv')
test_features = pd.read_csv('data/testing_features.csv')
train_labels = pd.read_csv('data/training_labels.csv')
test_labels = pd.read_csv('data/testing_labels.csv')

# Create an imputer object with a median filling strategy
imputer = SimpleImputer(strategy='median')

# Train on the training features
imputer.fit(train_features)

# Transform both training data and testing data
X = imputer.transform(train_features)
X_test = imputer.transform(test_features)

# Sklearn wants the labels as one-dimensional vectors
y = np.array(train_labels).reshape((-1,))
y_test = np.array(test_labels).reshape((-1,))

# Function to calculate mean absolute error


def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))


# Load Model from file
with open(pkl_filename, 'rb') as file:
    final_model = pickle.load(file)

# Feature Importances

# Extract feature importances into a dataframe
feature_results = pd.DataFrame({'feature': list(train_features.columns),
                                'importance': final_model.feature_importances_})

# Show the top 10 results
feature_results = feature_results.sort_values(
    'importance', ascending=False).reset_index(drop=True)
feature_results.head(10)
# Feature importance drops off rapidly after the first few suggesting all might not be needed for the model

figsize(12, 10)
plt.style.use('fivethirtyeight')

# Plot the 10 most important features in a horizontal bar chart
feature_results.loc[:9, :].plot(x='feature', y='importance',
                                edgecolor='k',
                                kind='barh', color='blue')
plt.xlabel('Relative Importance', size=20)
plt.ylabel('')
plt.title('Feature Importances from Random Forest', size=30)
plt.show()

# Use this for feature selection and try using Linear Regression
# Extract most important features
most_important_features = feature_results['feature'][:10]

# Find the index that corresponds to each feature name
indices = [list(train_features.columns).index(x)
           for x in most_important_features]

# Keep only the most important features
X_reduced = X[:, indices]
X_test_reduced = X_test[:, indices]

print('Most important training features shape: ', X_reduced.shape)
print('Most important testing  features shape: ', X_test_reduced.shape)

# Run Linear Regression Problem
lr = LinearRegression()

# Fit on full set of features
lr.fit(X, y)
lr_full_pred = lr.predict(X_test)

# Fit on reduced set of features
lr.fit(X_reduced, y)
lr_reduced_pred = lr.predict(X_test_reduced)

# Display results
print('Linear Regression Full Results: MAE =    %0.4f.' %
      mae(y_test, lr_full_pred))
print('Linear Regression Reduced Results: MAE = %0.4f.' %
      mae(y_test, lr_reduced_pred))
# Reducing the features did not help

# Try with Gradient Boost
# Create Model with same hyperparameters
model_reduced = GradientBoostingRegressor(loss='lad', max_depth=5, max_features=None,
                                          min_samples_leaf=6, min_samples_split=6,
                                          n_estimators=800, random_state=42)

# Fit and test on the reduced feature set
model_reduced.fit(X_reduced, y)
model_reduced_pred = model_reduced.predict(X_test_reduced)

print('Gradient Boosted Reduced Results: MAE = %0.4f' %
      mae(y_test, model_reduced_pred))
# Performance is slightly reduced after removing the features

# LIME or Locally Interpretable Model-agnostic Explanations
# We will look at using LIME to explain individual predictions made the by the model.
# LIME is a relatively new effort aimed at showing how a machine learning model thinks by approximating the region around a prediction with a linear model.

# Find the residuals
residuals = abs(model_reduced_pred - y_test)

# Exact the worst and best prediction
wrong = X_test_reduced[np.argmax(residuals), :]
right = X_test_reduced[np.argmin(residuals), :]

# Create a lime explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_reduced,
                                                   mode='regression',
                                                   training_labels=y,
                                                   feature_names=list(most_important_features))

# Display the predicted and true value for the wrong instance
print('Prediction: %0.4f' % model_reduced.predict(wrong.reshape(1, -1)))
print('Actual Value: %0.4f' % y_test[np.argmax(residuals)])

# Explanation for wrong prediction
wrong_exp = explainer.explain_instance(data_row=wrong,
                                       predict_fn=model_reduced.predict)

# Plot the prediction explaination
wrong_exp.as_pyplot_figure()
plt.title('Explanation of Prediction', size=28)
plt.xlabel('Effect on Prediction', size=22)
plt.show()

# Display the predicted and true value for the wrong instance
print('Prediction: %0.4f' % model_reduced.predict(right.reshape(1, -1)))
print('Actual Value: %0.4f' % y_test[np.argmin(residuals)])

# Explanation for wrong prediction
right_exp = explainer.explain_instance(
    right, model_reduced.predict, num_features=10)
right_exp.as_pyplot_figure()
plt.title('Explanation of Prediction', size=28)
plt.xlabel('Effect on Prediction', size=22)
plt.show()

# Examining a single decision tree
# Extract a single tree
single_tree = model_reduced.estimators_[105][0]

tree.export_graphviz(single_tree, out_file='images/tree.dot',
                     rounded=True,
                     feature_names=most_important_features,
                     filled=True)

single_tree

tree.export_graphviz(single_tree, out_file='images/tree_small.dot',
                     rounded=True, feature_names=most_important_features,
                     filled=True, max_depth=3)
# Unable to visualize
