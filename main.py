import pandas as pd
from sklearn.model_selection import train_test_split
from Model import Model
from sklearn.tree import DecisionTreeRegressor

titanic_dataset = pd.read_csv('/home/okechukwu/Downloads/titanic competition/train.csv')
titanic = titanic_dataset.drop('Cabin', axis=1)
X = titanic.drop('Survived', axis=1)
y = titanic.Survived
numerics = X.select_dtypes(exclude=['object']).columns.drop(['PassengerId'])
objects = X.select_dtypes(exclude=['number']).columns.drop(['Name', 'Ticket'])
objects_2 = X.select_dtypes(exclude=['number']).columns.drop('Name')
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
# print(random_selection(train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y, model_pipeline=full_pipeline))

decision_tree = Model(DecisionTreeRegressor, numeric_val=numerics, object_val=objects, random_state=True)
history = decision_tree.fitting(trainer_x=train_X, trainer_y=train_y)
print(test_X.ndim)
mae = decision_tree.metrics(train_X, train_y, test_X, test_y)
model = decision_tree.modelling()

# cross_val = decision_tree.cross_validation(model.fit(train_X,train_y), test_X, test_y)
# print(cross_val)
# print(mae)


tester_dataset = pd.read_csv('/home/okechukwu/Downloads/titanic competition/test.csv')
tester = tester_dataset.copy()
tester_labels = pd.read_csv('/home/okechukwu/Downloads/titanic competition/gender_submission.csv', index_col=0)
processed = tester[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']]
mae2 = decision_tree.metrics(train_X, train_y, processed, tester_labels)
print(f'test_set_mae: {mae2},\nvalidation_set_mae: {mae}')
