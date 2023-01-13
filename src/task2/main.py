import sys
sys.path.append('src')
import random

from common import *
import torch
import numpy

logging.info('Loading data.')
with multiprocessing.pool.ThreadPool(len(names)) as pool:
	globals().update(dict(zip(names, pool.map(pandas.read_pickle, paths_for_cleaned))))
del pool

seed = 42
rng = numpy.random.default_rng(seed)
number_of_duplications = 3
assert number_of_duplications > 0
new_idana_start_point = 1000000

# Point 1

positive_patients = anagraficapazientiattivi[anagraficapazientiattivi.y].index.to_frame().reset_index(drop=True)

all_events = pandas.concat([
	diagnosi[['idcentro', 'idana', 'data']],
	esamilaboratorioparametri[['idcentro', 'idana', 'data']],
	esamilaboratorioparametricalcolati[['idcentro', 'idana', 'data']],
	esamistrumentali[['idcentro', 'idana', 'data']],
	prescrizionidiabetefarmaci[['idcentro', 'idana', 'data']],
	prescrizionidiabetenonfarmaci[['idcentro', 'idana', 'data']],
	prescrizioninondiabete[['idcentro', 'idana', 'data']],
])
last_event = all_events.groupby(['idcentro', 'idana'], group_keys=True).data.max()
last_event_positive_patients = positive_patients.join(last_event, ['idcentro', 'idana'])

#given a dataset, for patients with positive labels, delete events happened in the last six months
def clean_last_six_months(df: pandas.DataFrame) -> pandas.DataFrame:
	df_with_last_event_for_positive_patients = df.merge(last_event_positive_patients, 'left', ['idcentro', 'idana'])
	assert (df_with_last_event_for_positive_patients.index == df.index).all()
	# Here NaT do exactly what we need.
	mask = df_with_last_event_for_positive_patients.data_x > df_with_last_event_for_positive_patients.data_y - pandas.DateOffset(months=6)
	res = df.drop(mask[mask].index)
	return res

logging.info('Start of task 2.')

logging.info(f'Before six months cleaning: {len(diagnosi)=}')
diagnosi = clean_last_six_months(diagnosi)
logging.info(f'After  six months cleaning: {len(diagnosi)=}')

logging.info(f'Before six months cleaning: {len(esamilaboratorioparametri)=}')
esamilaboratorioparametri = clean_last_six_months(esamilaboratorioparametri)
logging.info(f'After  six months cleaning: {len(esamilaboratorioparametri)=}')

logging.info(f'Before six months cleaning: {len(esamilaboratorioparametricalcolati)=}')
esamilaboratorioparametricalcolati = clean_last_six_months(esamilaboratorioparametricalcolati)
logging.info(f'After  six months cleaning: {len(esamilaboratorioparametricalcolati)=}')

logging.info(f'Before six months cleaning: {len(esamistrumentali)=}')
esamistrumentali = clean_last_six_months(esamistrumentali)
logging.info(f'After  six months cleaning: {len(esamistrumentali)=}')

logging.info(f'Before six months cleaning: {len(prescrizionidiabetefarmaci)=}')
prescrizionidiabetefarmaci = clean_last_six_months(prescrizionidiabetefarmaci)
logging.info(f'After  six months cleaning: {len(prescrizionidiabetefarmaci)=}')

logging.info(f'Before six months cleaning: {len(prescrizionidiabetenonfarmaci)=}')
prescrizionidiabetenonfarmaci = clean_last_six_months(prescrizionidiabetenonfarmaci)
logging.info(f'After  six months cleaning: {len(prescrizionidiabetenonfarmaci)=}')

logging.info(f'Before six months cleaning: {len(prescrizioninondiabete)=}')
prescrizioninondiabete = clean_last_six_months(prescrizioninondiabete)
logging.info(f'After  six months cleaning: {len(prescrizioninondiabete)=}')

del all_events, last_event

logging.info('Starting to balance the dataset.')

assert (positive_patients == positive_patients.drop_duplicates()).all().all(), \
	'there are duplicates in the positive patients'

duplicated_positive_patients = []
for i in range(number_of_duplications-1):
	copy = positive_patients.copy()
	copy['iddup'] = i
	duplicated_positive_patients.append(copy)
del copy, i
duplicated_positive_patients = pandas.concat(duplicated_positive_patients, ignore_index=True)

# This is a biijection to make sure that the new synthetic patients have unique
# ids for each duplication. All indices for this new patients are negative to
# easly distinguish them from the original ones. We are going to use 'index' to
# set a new value for 'idana' (changing 'idcentro' would have been the same).
bijection = duplicated_positive_patients.reset_index()
bijection['index'] = -bijection['index'] - 1

def naive_balancing(df: pandas.DataFrame) -> pandas.DataFrame:
	"""This function does 5 things:
	  1. duplicates the events for the positive patients in this dataframe
	  2. removes some of this events
	  3. perturbates the date of thi events
	  6. cleans the last six months
	  4. update the idana
	"""
	removed_frac = 0.01

	copied_positive_patients_df = df.merge(duplicated_positive_patients, 'inner', ['idcentro', 'idana'])
	assert 'iddup' in copied_positive_patients_df.columns
	assert len(copied_positive_patients_df) == (number_of_duplications-1)*len(df.merge(positive_patients, 'inner', ['idcentro', 'idana']))

	copied_positive_patients_df = copied_positive_patients_df.drop(
		copied_positive_patients_df.sample(None, removed_frac, False, random_state=rng).index
	).reset_index(drop=True)

	offsets = rng.normal(0, 3, len(copied_positive_patients_df)).astype('int')
	pert = pandas.to_timedelta(offsets, unit='d')
	copied_positive_patients_df.data = copied_positive_patients_df.data + pert

	copied_positive_patients_df = clean_last_six_months(copied_positive_patients_df).reset_index(drop=True)

	assert copied_positive_patients_df.idana.isin(bijection.idana).all()
	copied_positive_patients_df.idana = copied_positive_patients_df.merge(bijection)['index']
	copied_positive_patients_df = copied_positive_patients_df.drop('iddup', axis=1)

	return copied_positive_patients_df

# TODO: find number_of_duplications such that the number of patients with label
# y==1 is roughly the same as the ones with y==0.
diagnosi = pandas.concat([diagnosi, naive_balancing(diagnosi)], ignore_index=True)
esamilaboratorioparametri = pandas.concat([esamilaboratorioparametri, naive_balancing(esamilaboratorioparametri)], ignore_index=True)
esamilaboratorioparametricalcolati = pandas.concat([esamilaboratorioparametricalcolati, naive_balancing(esamilaboratorioparametricalcolati)], ignore_index=True)
esamistrumentali = pandas.concat([esamistrumentali, naive_balancing(esamistrumentali)], ignore_index=True)
prescrizionidiabetefarmaci = pandas.concat([prescrizionidiabetefarmaci, naive_balancing(prescrizionidiabetefarmaci)], ignore_index=True)
prescrizionidiabetenonfarmaci = pandas.concat([prescrizionidiabetenonfarmaci, naive_balancing(prescrizionidiabetenonfarmaci)], ignore_index=True)
prescrizioninondiabete = pandas.concat([prescrizioninondiabete, naive_balancing(prescrizioninondiabete)], ignore_index=True)

xx = duplicated_positive_patients.join(anagraficapazientiattivi, ['idcentro', 'idana'], 'inner')
assert len(xx) == len(duplicated_positive_patients)
xx.idana = xx.merge(bijection)['index']
xx = xx.drop('iddup', axis=1).set_index(['idcentro', 'idana'])
# TODO: perturbate data in xx
anagraficapazientiattivi = pandas.concat([anagraficapazientiattivi, xx])

del xx, naive_balancing, bijection, duplicated_positive_patients, \
	last_event_positive_patients, clean_last_six_months

# Point 2

# TODO: for **concentration** rebalancig perturbate anagraficapazientiattivi too and the values of the events.

# Deep Learning Stuff ##########################################################

# Data preparation

# Since the only case in which STITCH codes convey useful information is when
# the AMD code is NA we the a simple substitution of the NAs with the STITCH
# codes and remove the column entirely.
assert (esamilaboratorioparametricalcolati[esamilaboratorioparametricalcolati.codiceamd == 'AMD927'].codicestitch == 'STITCH001').all()
assert (esamilaboratorioparametricalcolati[esamilaboratorioparametricalcolati.codiceamd == 'AMD013'].codicestitch == 'STITCH002').all()
assert (esamilaboratorioparametricalcolati[esamilaboratorioparametricalcolati.codiceamd == 'AMD304'].codicestitch == 'STITCH005').all()
assert esamilaboratorioparametricalcolati[esamilaboratorioparametricalcolati.codiceamd.isna()].codicestitch.isin(['STITCH003', 'STITCH004']).all()
esamilaboratorioparametricalcolati.codiceamd.update(
	esamilaboratorioparametricalcolati[esamilaboratorioparametricalcolati.codiceamd.isna()].codicestitch
)
assert not esamilaboratorioparametricalcolati.codiceamd.isna().any()
esamilaboratorioparametricalcolati = esamilaboratorioparametricalcolati.drop('codicestitch', axis=1)

# For LSTMs the only reasonable way to use all the tables is to drop the
# 'valore' from all of them, since for each codiceamd there is a different
# domain, and keep only the events in the hope that enough information is left
# to make the classification. The idea is the following the value of an exam is
# less inportant than the sequence: if you did a blood pressure exam the result
# probabbli doesn't matter if the next "exam" is chemio therapy.
X = pandas.concat([
	diagnosi[['idcentro', 'idana', 'data', 'codiceamd']],
	esamilaboratorioparametri[['idcentro', 'idana', 'data', 'codiceamd']],
	esamilaboratorioparametricalcolati[['idcentro', 'idana', 'data', 'codiceamd']],
	esamistrumentali[['idcentro', 'idana', 'data', 'codiceamd']],
	prescrizionidiabetefarmaci[['idcentro', 'idana', 'data', 'codiceatc']].rename({'codiceatc': 'codiceamd'}, axis=1),
	prescrizionidiabetenonfarmaci[['idcentro', 'idana', 'data', 'codiceamd']],
	prescrizioninondiabete[['idcentro', 'idana', 'data', 'codiceamd']],
]).rename({'codiceamd': 'codice'}, axis=1)

# Ordinal encoding of codice
codes = pandas.Series(numpy.sort(X.codice.unique())).rename('codice').reset_index()
X = X.merge(codes).drop('codice', axis=1).rename({'index': 'codice'}, axis=1)

X = X.sort_values(['idana', 'data'])

# The histogram clearly shows that the majority of patients are old.
# ages = (sampling_date - anagraficapazientiattivi.annonascita).astype('<m8[Y]').rename('eta')

# To give the models an easier time understanding the date in which the event
# happened we scale it wrt how old the patient is. Any age above 100 years is
# considered to be the same as 100.
tmp = X.join(anagraficapazientiattivi.annonascita, ['idcentro', 'idana'])
assert not tmp.data.isna().any()
seniority = (tmp.data - tmp.annonascita).astype('<m8[Y]').clip(None, 100.0)/100.0
X['seniority'] = seniority
X.drop('data', axis=1, inplace=True)
del tmp, seniority
# FIXME: there are still NaN in seniority because the synthetic patients are not
# in anagraficapazientiattivi

# Ordering columns just for convenience.
new_columns_order = ['idana', 'idcentro', 'seniority', 'codice']
X = X.reindex(columns=new_columns_order)
del new_columns_order

# NOTE: We could add a feature that represents the table from which the data
# comes from.

class Model(torch.nn.Module):

	def __init__(
			self,
			num_input_size: int,
			num_hidden_size: int,
			txt_input_size: int,
			txt_hidden_size: int,
			# Common arguments to all LSTMs.
			num_layers: int,
			bias: bool,
			batch_first: bool,
			dropout: float,
			bidirectional: bool,
			proj_size: int
		) -> None:
		super().__init__()

		common_args = {
			'num_layers':    num_layers,
			'bias':          bias,
			'batch_first':   batch_first,
			'dropout':       dropout,
			'bidirectional': bidirectional,
			'proj_size':     proj_size,
		}

		self.lstm_num = torch.nn.LSTM(num_input_size, num_hidden_size, **common_args)
		self.lstm_txt = torch.nn.LSTM(txt_input_size, txt_hidden_size, **common_args)

		classifier_input_size = num_hidden_size + txt_hidden_size
		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(classifier_input_size, classifier_input_size//2),
			torch.nn.ReLU(),
			torch.nn.Linear(classifier_input_size//2, 1)
		)

	def forward(
			self,
			X_num: torch.Tensor,
			X_txt: torch.Tensor,
		) -> torch.Tensor:

		# NOTE: This step can be parallelized.
		o_num, (h_num, c_num) = self.lstm_num(X_num)
		o_txt, (h_txt, c_txt) = self.lstm_txt(X_txt)

		H = torch.cat([h_num, h_txt], 2) # Concat along the features.
		res = self.classifier(H) # logits

		return res

net = Model(
	5, 5, # num
	5, 5, # txt
	num_layers = 2,
	bias = True,
	batch_first = True,
	dropout = 0.1, # Probability of removing.
	bidirectional = True,
	proj_size = 0 # I don't know what this does.
)
print(net)
_ = net(
	# The dimensions mean
	#   * batch, row,     columns, or
	#   * batch, samples, features
	torch.ones(1, 10, 5), # num
	torch.ones(1, 10, 5)  # txt
)
