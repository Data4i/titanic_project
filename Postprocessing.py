# import numpy as np
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import mean_absolute_error
#
#
# Random_forest_parameters = [
#     {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
#     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
# ]
#
#
# def postprocessing(model, prepared_features, labels_for_prepared, param_grid):
#     grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error')
#     return grid_search.fit(prepared_features, labels_for_prepared,)
#
#
# def random_selection(train_X, train_y, test_X, test_y, model_pipeline):
#     model_pipeline.fit(train_X, train_y)
#     pred_y = model_pipeline.predict(test_X)
#     mae = mean_absolute_error(y_true=test_y, y_pred=pred_y)
#     first_num = np.random.randint(1, 100)
#     X = test_X[first_num:first_num+10]
#     y = test_y[first_num:first_num+10]
#     test_val = y
#     model_pred_val = model_pipeline.predict(X)
#     completion = f'Original_val:\n{test_val}, Predicted_val: {model_pred_val}'
#     return mae, completion
#
