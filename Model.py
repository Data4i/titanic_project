from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


class Model:
    def __init__(self, model, numeric_val, object_val, random_state=False):
        self.numeric_val = numeric_val
        self.object_val = object_val
        self.model = model
        self.random_state = random_state

        assert type(self.random_state) == bool, 'set random_state to a boolean value'

    def processing_types(self):
        cat_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='S')),
            ('encoder', OneHotEncoder()),
        ])

        num_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ])

        preprocessed = ColumnTransformer([
            ('num_val', num_pipe, self.numeric_val),
            ('obj_val', cat_pipe, self.object_val),
        ], remainder='drop')

        return preprocessed

    def modelling(self):
        preprocessed = self.processing_types()
        full_pipeline = None
        if self.random_state:
            full_pipeline = Pipeline([
                ('preprocessed', preprocessed),
                ('model', self.model())
            ])
        else:
            full_pipeline = Pipeline([
                ('preprocessed', preprocessed),
                ('model', self.model(random_state=42))
            ])
        return full_pipeline

    def fitting(self, trainer_x, trainer_y):
        model = self.modelling()
        return model.fit(trainer_x, trainer_y)

    def metrics(self, trainer_x, trainer_y, val_x, val_y):
        model = self.fitting(trainer_x, trainer_y)
        pred_y = model.predict(val_x)
        mae = mean_absolute_error(y_true=val_y, y_pred=pred_y)
        return mae

    def cross_validation(self, model, features, labels):
        scores = cross_val_score(self, model, features, labels, scoring='neg_mean_absolute_error', cv=10)
        mean = scores.mean()
        mae = mean
        return mae
