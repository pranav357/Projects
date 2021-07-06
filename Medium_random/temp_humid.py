#Apparent temperature - Temperature perceived by people caused by combined effects of air temperature, humidity and wind speed

#Import packages
from operator import ne
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
import seaborn as sns
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

weatherDataframe = pd.read_csv('weatherHistory.csv')
weatherDataframe.head();

#Pre processing
print(weatherDataframe.nunique())
#Can drop Loud Cover

#Analyze unique values row wise
data = weatherDataframe.to_numpy().astype(str)
# Summarize unique values in each column
for i in range(data.shape[1]):
    num = len(np.unique(data[:,i]))
    percentage = float(num)/data.shape[0]*100
    print('{}, {}, {:.1f}'.format(weatherDataframe.columns[i], num, percentage))#Specifying format inside this way

dropColumns=['Formatted Date','Loud Cover','Daily Summary']
weatherDataframe.drop(dropColumns, inplace=True, axis=1)

#Remove duplicate rows
# calculate duplicates
dups = weatherDataframe.duplicated()
# report if there are any duplicates
print(dups.any())
# list all duplicate rows
print(weatherDataframe[dups])
# delete duplicate rows
weatherDataframe.drop_duplicates(inplace=True)

#Reset Indices
weatherDataframe=weatherDataframe.reset_index(drop=True)

#Handling Missing Values and Outliers
weatherDataframe.isnull().values.any()   

#Getting the summary of what are missing value columns
weatherDataframe.isnull().sum()

#Drop rows with null data as they are small in number
# make copy to avoid changing original data
new_weatherDf = weatherDataframe.copy()
# removing missing values
new_weatherDf=new_weatherDf.dropna(axis=0)
# We have to reset indexes because our dataframe still having previous indexes after dropping rows
new_weatherDf=new_weatherDf.reset_index(drop=True)

new_weatherDf.isnull().values.any()

#One another way is to define a custom missing value set 
missing_values = ["n.a.","NA","n/a", "na"]
df = pd.read_csv('weatherHistory.csv', na_values = missing_values)
df.isnull().sum()

#Handling Outliers
fig, ax = plt.subplots(figsize = (25,10))
sns.boxplot(data = new_weatherDf, orient = "h", palette="Set2", ax=ax)
plt.show()
#Pressure column has an outlier

new_weatherDf.boxplot(column=['Temperature (C)']);
plt.show()
#Not outlier as there are large no of data points, skewed not outliers

new_weatherDf.boxplot(column=['Apparent Temperature (C)']);
plt.show()
#Same as previous

new_weatherDf.boxplot(column=['Humidity']);
plt.show()
#Possible outlier

fig, axes = plt.subplots(1,2)
plt.tight_layout(0.2)

print('Before Shape:', new_weatherDf.shape)
#Remove humidity below 0 outlier
new_weatherDf2 = new_weatherDf[new_weatherDf['Humidity']>0.0]
print("After Shape:", new_weatherDf2.shape)

sns.boxplot(new_weatherDf['Humidity'], orient='v', ax=axes[0])
axes[0].title.set_text("Before")
sns.boxplot(new_weatherDf2['Humidity'],orient='v',ax=axes[1])
axes[1].title.set_text("After")
plt.show()

## Replace new dataset with previous and resetting indexes
new_weatherDf=new_weatherDf2;
new_weatherDf=new_weatherDf.reset_index(drop=True)

new_weatherDf.boxplot(column=['Wind Speed (km/h)']);
plt.show()

fig, axes = plt.subplots(1,2)
plt.tight_layout(0.2)

print("Before Shape:",new_weatherDf.shape)
## Removing Windspeed above 60kmph outlier
new_weatherDf2 = new_weatherDf[ (new_weatherDf['Wind Speed (km/h)']<60)]
print("After Shape:",new_weatherDf2.shape)

sns.boxplot(new_weatherDf['Wind Speed (km/h)'],orient='v',ax=axes[0])
axes[0].title.set_text("Before")
sns.boxplot(new_weatherDf2['Wind Speed (km/h)'],orient='v',ax=axes[1])
axes[1].title.set_text("After")
plt.show()

## Replace new dataset with previous and reset indexes
new_weatherDf=new_weatherDf2;
new_weatherDf=new_weatherDf.reset_index(drop=True)

new_weatherDf.boxplot(column=['Wind Bearing (degrees)']);
plt.show()
#No outliers

new_weatherDf.boxplot(column=['Visibility (km)']);
plt.show()
#No outliers

new_weatherDf.boxplot(column=['Pressure (millibars)']);
plt.show()

fig, axes = plt.subplots(1,2)
plt.tight_layout(0.2)

print("Before Shape:",new_weatherDf.shape)
## Removing Pressure bellow 800 outliers
new_weatherDf2 = new_weatherDf[ (new_weatherDf['Pressure (millibars)']>800)]
print("After Shape:",new_weatherDf2.shape)

sns.boxplot(new_weatherDf['Pressure (millibars)'],orient='v',ax=axes[0])
axes[0].title.set_text("Before")
sns.boxplot(new_weatherDf2['Pressure (millibars)'],orient='v',ax=axes[1])
axes[1].title.set_text("After")
plt.show()

# ## Replace new dataset with previous and reset indexes
new_weatherDf=new_weatherDf2;
new_weatherDf=new_weatherDf.reset_index(drop=True)

#Final outcome
fig_dims = (25, 10);
fig, ax = plt.subplots(figsize=fig_dims);
sns.boxplot(data=new_weatherDf, orient="h", palette="Set2",ax=ax );
plt.show()

#Now do all transformations
#Split data into train and test, splitting is done to avoid data leakage problem
#No peaking ahead, otherwise data from test set will leak into training data

features_df= new_weatherDf.drop('Apparent Temperature (C)', 1)
features_df

target = pd.DataFrame(new_weatherDf['Apparent Temperature (C)'], columns=["Apparent Temperature (C)"])
target

X_train, X_test, y_train, y_test = train_test_split(features_df, target, test_size=0.2, random_state = 101)

#Reset all indices
X_train=X_train.reset_index(drop=True)
X_test=X_test.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)

#Produce QQ plots and histograms
# Temperature (C). - Training
stats.probplot(X_train["Temperature (C)"], dist="norm", plot=plt);
plt.show();
X_train["Temperature (C)"].hist();
plt.show()

# Temperature (C). - Testing
stats.probplot(X_test["Temperature (C)"], dist="norm", plot=plt);
plt.show();
X_test["Temperature (C)"].hist();
plt.show()
#Shows normal distribution, all data in red line
    
## Humidity. - X_train
stats.probplot(X_train["Humidity"], dist="norm", plot=plt);
plt.show();
X_train["Humidity"].hist();
plt.show()
#Shows a left skewed distribution

#Apply exponential transformation
# create columns variables to hold the columns that need transformation
columns = ['Humidity']

# create the function transformer object with exponentioal transformation
exp_transformer = FunctionTransformer(lambda x:x**3, validate=True)

#Apply the transformation
data_new = exp_transformer.transform(X_train[columns])
df_new = pd.DataFrame(data_new, columns=columns)

#Replace new values with old
X_train.Humidity=df_new['Humidity']

stats.probplot(X_train["Humidity"], dist="norm", plot=plt);
plt.show();
X_train['Humidity'].hist();
plt.show()

stats.probplot(X_train["Humidity"], dist="norm", plot=plt);
plt.show();
X_train['Humidity'].hist();
plt.show()
#Same problem

# create columns variables to hold the columns that need transformation
columns = ['Humidity']

# create the function transformer object with exponentioal transformation
exp_transformer = FunctionTransformer(lambda x:x**3, validate=True)

# apply the transformation 
data_new = exp_transformer.transform(X_test[columns])
df_new = pd.DataFrame(data_new, columns=columns)

# replace new values with previous data frame
X_test.Humidity=df_new['Humidity']

X_test['Humidity'].hist();
plt.show()

## Wind Speed (km/h). - X_train
stats.probplot(X_train["Wind Speed (km/h)"], dist="norm", plot=plt);
plt.show();
X_train["Wind Speed (km/h)"].hist();
plt.show()
#Skewed

#Apply log transformation
X_train['Wind Speed (km/h)'].min()
#There are zero values, so take log(x+1)

# create columns variables to hold the columns that need transformation
columns = ['Wind Speed (km/h)']

# create the function transformer object with logarithm transformation
logarithm_transformer = FunctionTransformer(np.log1p, validate=True)

# apply the transformation 
data_new = logarithm_transformer.transform(X_train[columns])
df_new = pd.DataFrame(data_new, columns=columns)

# replace new values with previous data frame
X_train['Wind Speed (km/h)']=df_new['Wind Speed (km/h)']

# After transformation
stats.probplot(X_train["Wind Speed (km/h)"], dist="norm", plot=plt);
plt.show();
X_train["Wind Speed (km/h)"].hist();
plt.show()

## Wind Speed (km/h). - X_test
stats.probplot(X_test["Wind Speed (km/h)"], dist="norm", plot=plt);
plt.show();
X_test["Wind Speed (km/h)"].hist();
plt.show()
#Same problem

# create columns variables to hold the columns that need transformation
columns = ['Wind Speed (km/h)']

# create the function transformer object with logarithm transformation
logarithm_transformer = FunctionTransformer(np.log1p, validate=True)

# apply the transformation 
data_new = logarithm_transformer.transform(X_test[columns])
df_new = pd.DataFrame(data_new, columns=columns)

# replace new values with previous data frame
X_test['Wind Speed (km/h)']=df_new['Wind Speed (km/h)']
X_test['Wind Speed (km/h)'].hist();
plt.show()

## Wind Bearing (degrees). - X_train
stats.probplot(X_train["Wind Bearing (degrees)"], dist="norm", plot=plt)#Calculated quantiles for the given data against the specified distribution
plt.show();
X_train["Wind Bearing (degrees)"].hist();
plt.show()
#Pretty normalized

## Visibility (km). - X_train
stats.probplot(X_train["Visibility (km)"], dist="norm", plot=plt);
plt.show();
X_train["Visibility (km)"].hist();
plt.show()
#Left skewed

# create columns variables to hold the columns that need transformation
columns = ['Visibility (km)']

# create the function transformer object with exponentioal transformation
exp_transformer = FunctionTransformer(lambda x:x**3, validate=True)

# apply the transformation 
data_new = exp_transformer.transform(X_train[columns])
df_new = pd.DataFrame(data_new, columns=columns)
df_new['Visibility (km)'].hist()
plt.show()

# replace new values with previous data frame
#X_train["Visibility (km)"]=df_new['Visibility (km)']

stats.probplot(df_new["Visibility (km)"], dist="norm", plot=plt);
plt.show()
#Keep original

## Pressure (millibars). - X_train
stats.probplot(X_train["Pressure (millibars)"], dist="norm", plot=plt)
plt.show()
X_train["Pressure (millibars)"].hist()
plt.show()

## Apparent Temperature (C). -y_train
stats.probplot(y_train["Apparent Temperature (C)"], dist="norm", plot=plt);
plt.show();
y_train["Apparent Temperature (C)"].hist();
plt.show()

## Apparent Temperature (C). -y_test
stats.probplot(y_test["Apparent Temperature (C)"], dist="norm", plot=plt);
plt.show();
y_test["Apparent Temperature (C)"].hist();
plt.show()

#Final histograms
X_train.hist(figsize=(16,10));
plt.show()

#Apply suitable feature encoding techniques

#Perform encoding on Summary column
#Create instance of one-hot encoder
enc = OneHotEncoder(handle_unknown='ignore')

#Fit only training data
enc.fit(X_train[['Summary']])#Data Frame vs Series

colnames = enc.get_feature_names()
enc_df = pd.DataFrame(enc.transform(X_train[['Summary']]).toarray(), columns = colnames)

# transforming testing data
enc_df_test = pd.DataFrame(enc.transform(X_test[['Summary']]).toarray(),columns=colnames)

X_train = X_train.join(enc_df)
X_test = X_test.join(enc_df_test)

#Encoding for Precip types
X_train['Precip Type'] = X_train['Precip Type'].astype('category')
X_train['Precip Type'] = X_train['Precip Type'].cat.codes

X_test['Precip Type'] = X_test['Precip Type'].astype('category')
X_test['Precip Type'] = X_test['Precip Type'].cat.codes

#Remove summary as is encoded and joined
X_train.drop('Summary', inplace=True, axis=1)
X_test.drop('Summary', inplace=True, axis=1)

#Scale and standardize feeatures
X_train.describe().loc[['min','max']]

to_standardize_train = X_train[['Temperature (C)', 'Humidity','Wind Speed (km/h)','Visibility (km)','Pressure (millibars)']].copy()
to_standardize_test = X_test[['Temperature (C)', 'Humidity','Wind Speed (km/h)','Visibility (km)','Pressure (millibars)']].copy()

to_standardize_train.hist(figsize=(18,10));
plt.show()

#Apply standardization
#Create scalar object
scaler = StandardScaler()

#Fit only training data
scaler.fit(to_standardize_train)

train_scaled = scaler.transform(to_standardize_train)
test_scaled = scaler.transform(to_standardize_test)

standardized_df_train = pd.DataFrame(train_scaled, columns = to_standardize_train.columns)
standardized_df_test = pd.DataFrame(test_scaled, columns = to_standardize_test.columns)

standardized_df_train.hist(figsize=(18,10));
plt.show()
standardized_df_test.hist(figsize=(18,10));
plt.show()

#Now apply to standard variable
#Create the scaler object
scaler2 = StandardScaler()

#Fit only training data
scaler2.fit(y_train)

target_train_scaled = scaler2.transform(y_train)
target_test_scaled = scaler2.transform(y_test)

standardized_target_df_train = pd.DataFrame(target_train_scaled, columns = y_train.columns)
standardized_target_df_test = pd.DataFrame(target_test_scaled, columns = y_test.columns)

standardized_target_df_train.hist(figsize=(18,10));
plt.show()
standardized_target_df_train.hist(figsize=(18,10));
plt.show()

#Replace old ones with new
X_train.drop(columns=to_standardize_train.columns,inplace=True, axis=1)
X_test.drop(columns=to_standardize_test.columns,inplace=True, axis=1)

X_train = standardized_df_train.join(X_train)
X_test = standardized_df_test.join(X_test)

y_train = standardized_target_df_train;
y_test = standardized_target_df_test;

#Apply feature discretization
#Transforming continuous variables to discrete ones
train_data_disc = pd.DataFrame(X_train, columns=['Wind Bearing (degrees)'])
test_data_disc = pd.DataFrame(X_test, columns=['Wind Bearing (degrees)'])

#Fit scaler to data
discretizer = KBinsDiscretizer(n_bins = 8, encode = 'ordinal', strategy = 'kmeans')
discretizer.fit(train_data_disc)

train_discretized = discretizer.transform(train_data_disc)
test_discretized = discretizer.transform(test_data_disc)

train_discretized_df = pd.DataFrame(train_discretized,columns=['Wind Bearing (bins)'])
test_discretized_df = pd.DataFrame(test_discretized,columns=['Wind Bearing (bins)'])

train_discretized_df.hist();
plt.show()
test_discretized_df.hist();
plt.show()

# Append these into previous datframes
X_train.drop(columns=['Wind Bearing (degrees)'],inplace=True, axis=1)
X_train = train_discretized_df.join(X_train)

X_test.drop(columns=['Wind Bearing (degrees)'],inplace=True, axis=1)
X_test = test_discretized_df.join(X_test)

#Feature Engineering
#Use PCA or SVD for feature reduction
# Make an instance of the Model
pca = PCA()
pca.fit(X_train) 

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

X_train_pca_df = pd.DataFrame(data = X_train_pca,columns=X_train.columns)
X_test_pca_df = pd.DataFrame(data = X_test_pca,columns=X_test.columns)

# Lambda values (Eigon values)
explained_variance_ratio=pca.explained_variance_ratio_
explained_variance_ratio

plt.plot(range(33), pca.explained_variance_ratio_)
plt.plot(range(33), np.cumsum(pca.explained_variance_ratio_)) # Cummualtive frequency graph
plt.title("Component-wise and Cumulative Explained Variance")
plt.show()

arr=explained_variance_ratio 
sum = 0;  
for i in range(0, 10):    
   sum=sum+arr[i]   
print("Sum :" + str(sum));    
#10 components capture 99% of information

pca = PCA(n_components=10)
pca.fit(X_train) 

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#Correlation Matrix with heat map
correlation_mat = X_train.iloc[:,:33].corr()
plt.figure(figsize=(40,40))
sns.heatmap(correlation_mat, annot = True, cmap="RdYlGn")
plt.title("Figure 1 - Correlation matrix for features")
plt.show()

#High Corelated features:
#Humidity vs Temparature
#temperature and visibility
#humidity and visibility
#foggy and visibility
#precip type and temparature
#Humidity is highly correlated(negatively) with temperature

df_combined = X_train.join(y_train)

correlation_mat = df_combined.iloc[:,:34].corr()
plt.figure(figsize=(40,40))
sns.heatmap(correlation_mat, annot = True, cmap="RdYlGn")
plt.title("Correlation matrix ")
plt.show()

#Modelling
#Use Linear Regression Problem
#Try with and without PCA

#Without PCA and 33 features
lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)

predictions = lm.predict(X_test)
y_hat = pd.DataFrame(predictions, columns=["predicted"])
print(y_hat.head(10)) #print predictions for first ten values

#Calculate MSE
mse = mean_squared_error(y_test, y_hat)
mse

#Root mean squared error
rmsq = sqrt(mean_squared_error(y_test, y_hat))
rmsq

#Percentage of explained variance of the predictions
score = lm.score(X_test,y_test)
score

#W parameters of the model
print(lm.coef_)

abs_coef=abs(lm.coef_)
plt.plot(abs_coef[0])
plt.show()

lm.coef_[0].max()

#Intercept of the model
print(lm.intercept_)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.plot(y_test, label = "Actual")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_hat, label = "Pred")
plt.legend()

# Display a figure.
plt.show()

plt.figure(figsize=(20, 10))

# Limiting the data set to 100 rows for more clearence
plt.plot(y_hat[:100], label = "Pred")
plt.plot(y_test[:100], label = "Actual")

plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('Two or more lines on same plot with suitable legends ')
plt.legend()

plt.show()

#Apply with PCA
lm2 = linear_model.LinearRegression()
model2 = lm2.fit(X_train_pca,y_train)

predictions = lm2.predict(X_test_pca)
y_hat_pca = pd.DataFrame(predictions, columns=["predicted"])
print(y_hat_pca.head(10)) #print predictions for first ten values

mse_for_pca=mean_squared_error(y_test, y_hat_pca)
mse_for_pca

rmsq2 = sqrt(mean_squared_error(y_test, y_hat_pca))
rmsq2

score_pca=lm2.score(X_test_pca,y_test)
score_pca

#W parameters of the model
print(lm2.coef_)

#Intercept of the model
print(lm2.intercept_)

abs_coef=abs(lm2.coef_)
plt.plot(abs_coef[0])
plt.show()

plt.figure(figsize=(20, 10))

# Limiting the data set to 100 rows for more clearence
plt.plot(y_hat_pca[:100], label = "Pred")
plt.plot(y_test[:100], label = "Actual")

plt.title('Comparison of Prediction vs Actual - With PCA')
plt.legend()
plt.show()

#K-fold cross validation
#Without PCA
X_combined = np.r_[X_train, X_test]
y_combined = np.r_[y_train, y_test]
#Does row wise merging

# Perform 6-fold cross validation
scores = cross_val_score(model, X_combined, y_combined, cv=6)
print("Cross-validated scores:", scores)

predictions = cross_val_predict(model, X_combined, y_combined, cv=6)
accuracy = metrics.r2_score(y_test, y_hat)
print("Cross-Predicted Accuracy:", accuracy)

#With PCA
X_combined = np.r_[X_train_pca, X_test_pca]
y_combined = np.r_[y_train, y_test]

scores = cross_val_score(model2, X_combined, y_combined, cv=6)
print("Cross-validated scores:", scores)

predictions = cross_val_predict(model2, X_combined, y_combined, cv=6)
accuracy = metrics.r2_score(y_test, y_hat_pca)
print("Cross-Predicted Accuracy:", accuracy)
