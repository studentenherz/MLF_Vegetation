from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import pandas as pd
import pickle
import bz2
import os

from prep_data import preapare_data

DATA_SUBDIR = '../data'

data = pd.read_pickle(os.path.join(DATA_SUBDIR, 'train.pkl'))

# Separete the target
target = data['dosel_forestal']
prep_data = data.drop(['dosel_forestal'], axis=1).to_numpy()

param_grid =[
	{'n_estimators': [600, 800, 1000], 'learning_rate': [0.3], 'max_depth': [12, 14, 16]}
]

xgb_classifier = XGBClassifier(andom_state=1234, n_jobs=-1, tree_method='gpu_hist')

grid_search = GridSearchCV(xgb_classifier, param_grid, cv=10, n_jobs=8, return_train_score=True, verbose=2.5)
grid_search.fit(prep_data, target - 1)

best_model = grid_search.best_estimator_
with bz2.BZ2File('bz2.XGB.pkl', 'wb') as file:
	pickle.dump(best_model, file)


xgb_prediction = best_model.predict(prep_data)

xgbacc = accuracy_score(target.to_numpy(), xgb_prediction)
xgbprec = precision_score(target.to_numpy(), xgb_prediction, average='macro')
xgbrec = recall_score(target.to_numpy(), xgb_prediction, average='macro')

print(f'Training set\n\nAccuracy:\t{xgbacc:.8f}\nPrecision:\t{xgbprec:.8f}\nRecall:\t\t{xgbrec:.8f}')

test_set = pd.read_pickle(os.path.join(DATA_SUBDIR, 'test.pkl'))
prep_test, test_target = preapare_data(test_set)

xgb_test_prediction = best_model.predict(prep_test) + 1

from sklearn.metrics import accuracy_score, precision_score, recall_score

xgbacc = accuracy_score(test_target.to_numpy(), xgb_test_prediction)
xgbprec = precision_score(test_target.to_numpy(), xgb_test_prediction, average='macro')
xgbrec = recall_score(test_target.to_numpy(), xgb_test_prediction, average='macro')

print(f'Testing\n\nAccuracy:\t{xgbacc:.8f}\nPrecision:\t{xgbprec:.8f}\nRecall:\t\t{xgbrec:.8f}')

print(f"Best Model: {best_model}")