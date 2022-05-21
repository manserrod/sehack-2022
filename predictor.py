# -*- coding: utf-8 -*-
"""
Created on Sat May 21 11:58:38 2022

@author: Manuel Serrano / Andrés Jordán
"""

import pandas
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score

# We import the .csv data to dataframes
df1 = pandas.read_csv('train1.csv')
df2 = pandas.read_csv('train2.csv', delimiter=';')
dft = pandas.read_csv('test_x.csv')

# We import the .json data to dataframes
df3 = pandas.read_json('http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/first')
df4 = pandas.read_json('http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/second')
df5 = pandas.read_json('http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/third')

# These 3 fields are dropped since they don't appear in train1 and train2
df3 = df3.drop(['','EPRTRAnnexIMainActivityCode', 'EPRTRSectorCode'], axis=1)
df4 = df4.drop(['','EPRTRAnnexIMainActivityCode', 'EPRTRSectorCode'], axis=1)
df5 = df5.drop(['','EPRTRAnnexIMainActivityCode', 'EPRTRSectorCode'], axis=1)

# The dataframes columns are sorted alphabetically by the headers
df1 = df1.reindex(sorted(df1.columns), axis=1)
df2 = df2.reindex(sorted(df2.columns), axis=1)
df3 = df3.reindex(sorted(df3.columns), axis=1)
df4 = df4.reindex(sorted(df4.columns), axis=1)
df5 = df5.reindex(sorted(df5.columns), axis=1)
dft = dft.reindex(sorted(dft.columns), axis=1)

# All the training datasets are concatenated
df=pandas.concat([df1, df2, df3, df4, df5])

# We define the training dataset for our estimator
target = 'pollutant'
X = df.drop([target], axis = 1)
y = df[target]


# The transformers are created depending on the type of data in the df
integer_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('scaler', StandardScaler())])

continuous_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('lab_enc', OneHotEncoder(handle_unknown='ignore'))])

# We apply the transformations to the correct columns in the dataframe.
integer_features = list(X.columns[X.dtypes == 'int64'])
continuous_features = list(X.columns[X.dtypes == 'float64'])
categorical_features = list(X.columns[X.dtypes == 'object'])

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('ints', integer_transformer, integer_features),
        ('cont', continuous_transformer, continuous_features),
        ('cat', categorical_transformer, categorical_features)])

# Create a pipeline that combines the preprocessor created above with a classifier.
from sklearn.neighbors import KNeighborsClassifier

base = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', KNeighborsClassifier())])

# We introduce the dataset into our model and make the predictions
model = base.fit(X,y)
preds = model.predict(dft)

# The predictions are exported to .csv and .json with the pollutants codes applied
d={'Nitrogen oxides (NOX)':0, 'Carbon dioxide (CO2)':1, 'Methane (CH4)':2}
output=[d[k] for k in preds]
dfout=pandas.DataFrame(output)

dfout.to_csv('prediction.csv', index_label=['test_index'], header=['pollutant'])

dfout.columns = ['pollutants']
dfout.to_json('prediction.json')
