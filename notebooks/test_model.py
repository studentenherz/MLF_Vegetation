from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import argparse
import pickle
import bz2

# Prepare de data
def prepare_data(filename):
	'''
		Read and prepare data

	Parameters
	==========

	filename: str 
		If terminated in .zip will try to read a zipped csv file,
		else will try to read a pandas DataFrame in a pickle

	Returns
	=======

	target: pd.DataFrame
		Target feature

	prep_data: np.ndarray
		Prepared data to feed to the model

	'''
	if filename[:-4] == '.zip':
		data = pd.read_csv(filename, compression='zip')
	else:
		data = pd.read_pickle(filename)

	# Combine shades
	data['sombra'] = data[['sombra_maniana', 'sombra_mediodia', 'sombra_tarde']].mean(axis=1)

	# Encode categoric 
	clase_area_silvestre_one_hot_encoded = pd.get_dummies(data['clase_area_silvestre'], prefix='clase_area_silvestre')
	newdata = pd.concat([data, clase_area_silvestre_one_hot_encoded], axis=1)

	clase_suelo_one_hot_encoded = pd.get_dummies(data['clase_suelo'], prefix='clase_suelo')
	newdata = pd.concat([newdata, clase_suelo_one_hot_encoded], axis=1)

	# I'll encode this entirely so I'll drop the original
	data.drop('clase_area_silvestre', axis=1, inplace=True)

	data = pd.concat([data, clase_area_silvestre_one_hot_encoded, clase_suelo_one_hot_encoded[['clase_suelo_10', 'clase_suelo_39', 'clase_suelo_38']]], axis=1)

	target = data['dosel_forestal']
	prep_data = data.drop(['dosel_forestal'], axis=1).to_numpy()


	return target, prep_data

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Apply XGBoost best model to your data and see the result')

	parser.add_argument('-I', '--input', dest='filename', type=str, help='Input file with the testing data, if .zip the program will assume it is a zipped .csv, else it will assume it is a pickled pandas DataFrame.')

	args = parser.parse_args()

	if not args.filename:
		parser.error('Input file is mandatory')

	target, prep_data = prepare_data(args.filename)

	# Load the model 

	with bz2.BZ2File('bz2.best_model.pkl', 'rb') as mfile:
		model = pickle.load(mfile)


	prediction = model.predict(prep_data)

	tacc = accuracy_score(target.to_numpy(), prediction)
	tprec = precision_score(target.to_numpy(), prediction, average='macro')
	trec = recall_score(target.to_numpy(), prediction, average='macro')

	print(f'Model: {model}\n\nAccuracy:\t{tacc:.8f}\nPrecision:\t{tprec:.8f}\nRecall:\t\t{trec:.8f}')