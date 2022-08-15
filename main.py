import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, QuantileTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error

titanic_dataset = pd.read_csv('/home/okechukwu/Downloads/titanic competition/train.csv')
titanic = titanic_dataset.drop('Cabin', axis=1)
numeric_values = titanic.select_dtypes(include='number').columns.drop('Survived')
object_values = titanic.select_dtypes(include='object').columns.drop(['Ticket', 'Name'])

"""
print(titanic.select_dtypes(include='object').drop(['Ticket', 'Name'], axis=1))
impute = SimpleImputer(strategy='constant')
no_null_cat = impute.fit_transform(titanic.select_dtypes(include='object').drop(['Ticket', 'Name'], axis=1))
# ordinal_encoder = OrdinalEncoder()
# cat_val = ordinal_encoder.fit_transform(no_null_cat)
one_hot = OneHotEncoder()
cat_val = one_hot.fit_transform(no_null_cat)
print(cat_val.toarray())
"""

X = titanic.drop(['Survived', 'Ticket', 'Name'], axis=1)
y = titanic.Survived

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# print(train_X.head())
# print(train_y.head)

cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='constant')),
    ('Encoder', OneHotEncoder())
])

num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('Scaler', MinMaxScaler()),
    # ('Scaler', StandardScaler()),
    # ('quantile', QuantileTransformer(n_quantiles=100, random_state=42))
])

preprocessing = ColumnTransformer([
    ('numerical', num_pipe, numeric_values),
    ('object', cat_pipe, object_values)
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessing),
    ('model',DecisionTreeRegressor(max_leaf_nodes=500))
])

full_pipeline.fit(train_X, train_y)
pred_y = full_pipeline.predict(test_X)
# print(pred_y.shape, test_y.shape)
mae = mean_absolute_error(y_true=test_y, y_pred=pred_y)
print(mae)
# print(test_y[50:51])
# print(full_pipeline.predict(test_X[50:51]))


def random_selection(X, y):
    first_num = np.random.randint(1, 100)
    X = X[first_num:first_num+10]
    y = y[first_num:first_num+10]
    test_val = y
    model_pred_val = full_pipeline.predict(X)
    completion = f'Original_val:\n{test_val}, Predicted_val: {model_pred_val}'
    return completion


print(random_selection(X=test_X, y=test_y))
