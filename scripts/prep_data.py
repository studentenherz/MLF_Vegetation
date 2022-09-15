from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

# Data location
DATA_NAME = 'train_r1.zip'
DATA_SUBDIR = '../data'
DATA_PATH = os.path.join(DATA_SUBDIR, DATA_NAME)

def preapare_data(data):
	'''
		Prepare data for training

		Parameters
		==========

		data: DataFrame with all the data

		Return
		======

		prep_data: Prepared data after categorical encoding normalization

		target: Target column
	'''

	# Combine shades
	data['sombra'] = data[['sombra_maniana', 'sombra_mediodia', 'sombra_tarde']].mean(axis=1)

	# One-hot encoding
	clase_area_silvestre_one_hot_encoded = pd.get_dummies(data['clase_area_silvestre'], prefix='clase_area_silvestre')
	clase_suelo_one_hot_encoded = pd.get_dummies(data['clase_suelo'], prefix='clase_suelo')

	# I'll encode this entirely so I'll drop the original
	data.drop('clase_area_silvestre', axis=1, inplace=True)

	# Combine everything
	data = pd.concat([data, clase_area_silvestre_one_hot_encoded, clase_suelo_one_hot_encoded[['clase_suelo_10', 'clase_suelo_39', 'clase_suelo_38']]], axis=1)


	# Separete the target
	target = data['dosel_forestal']
	prep_data = data.drop(['dosel_forestal'], axis=1).to_numpy()

	return prep_data, target

if __name__ == '__main__':
	# Read raw data
	data = pd.read_csv(DATA_PATH, compression='zip')

	train_set, test_set = train_test_split(data, train_size=0.8, random_state=1234)

	# Save testing data to a pickle
	test_set.to_pickle(os.path.join(DATA_SUBDIR, 'test.pkl'))