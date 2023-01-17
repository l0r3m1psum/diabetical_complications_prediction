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
number_of_duplications = 3 # TODO: right now n-1 duplications are added to the data. This have to be changed.
assert number_of_duplications > 0

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

	assert pandas.MultiIndex.from_frame(copied_positive_patients_df[['idcentro', 'idana']]) \
		.isin(pandas.MultiIndex.from_frame(bijection[['idcentro', 'idana']])).all()
	copied_positive_patients_df = copied_positive_patients_df.merge(bijection) \
		.drop(['iddup', 'idana'], axis=1).rename({'index': 'idana'}, axis=1)

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

xx = duplicated_positive_patients.join(anagraficapazientiattivi, ['idcentro', 'idana'], 'inner').reset_index(drop=True)
assert len(xx) == len(duplicated_positive_patients)
assert xx.y.all()
xx = xx.merge(bijection).drop(['iddup', 'idana'], axis=1) \
	.rename({'index': 'idana'}, axis=1).set_index(['idcentro', 'idana'])
# TODO: perturbate data in xx
anagraficapazientiattivi = pandas.concat([anagraficapazientiattivi, xx])

assert len(diagnosi)                           == len(diagnosi                          .join(anagraficapazientiattivi, ['idcentro', 'idana'], 'inner'))
assert len(esamilaboratorioparametri)          == len(esamilaboratorioparametri         .join(anagraficapazientiattivi, ['idcentro', 'idana'], 'inner'))
assert len(esamilaboratorioparametricalcolati) == len(esamilaboratorioparametricalcolati.join(anagraficapazientiattivi, ['idcentro', 'idana'], 'inner'))
assert len(esamistrumentali)                   == len(esamistrumentali                  .join(anagraficapazientiattivi, ['idcentro', 'idana'], 'inner'))
assert len(prescrizionidiabetefarmaci)         == len(prescrizionidiabetefarmaci        .join(anagraficapazientiattivi, ['idcentro', 'idana'], 'inner'))
assert len(prescrizionidiabetenonfarmaci)      == len(prescrizionidiabetenonfarmaci     .join(anagraficapazientiattivi, ['idcentro', 'idana'], 'inner'))
assert len(prescrizioninondiabete)             == len(prescrizioninondiabete            .join(anagraficapazientiattivi, ['idcentro', 'idana'], 'inner'))

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

# There are probbaly less wasteful ways to do this but this is the easiest one
# to keep labels in sync with the data.
assert len(X) == len(X.join(anagraficapazientiattivi.y, ['idcentro', 'idana']))
X = X.join(anagraficapazientiattivi.y, ['idcentro', 'idana'])

# Ordinal encoding of codice
codes = pandas.Series(numpy.concatenate([['PADDING'], numpy.sort(X.codice.unique())])).rename('codice').reset_index()
X = X.merge(codes).drop('codice', axis=1).rename({'index': 'codice'}, axis=1)

# NOTE: this should be useless since we order again below.
X = X.sort_values(['idcentro', 'idana', 'data']).reset_index(drop=True)

# The histogram clearly shows that the majority of patients are old.
# ages = (sampling_date - anagraficapazientiattivi.annonascita).astype('<m8[Y]').rename('eta')

# To give the models an easier time understanding the date in which the event
# happened we scale it wrt how old the patient is. Any age above 100 years is
# considered to be the same as 100.
tmp = X.join(anagraficapazientiattivi.annonascita, ['idcentro', 'idana'])
assert not tmp.data.isna().any()
assert not tmp.annonascita.isna().any()
seniority = (tmp.data - tmp.annonascita).astype('<m8[Y]').clip(None, 100.0)/100.0
X['seniority'] = seniority
X.drop('data', axis=1, inplace=True)
del tmp, seniority

# Ordering columns just for convenience.
new_columns_order = ['idana', 'idcentro', 'seniority', 'codice', 'y']
X = X.reindex(columns=new_columns_order)
del new_columns_order

# NOTE: We could add a feature that represents the table from which the data
# comes from.

# Here we create a tensor for the history of each patients.
# TODO: put some assertions to verify that this split is correct.
s = X.sort_values(['idcentro', 'idana', 'seniority']).reset_index(drop=True)
indexes = numpy.nonzero(numpy.diff(s.idana.values))[0]+1
tot = len(X)
splits = numpy.concatenate([indexes[:1], numpy.diff(indexes)])
splits = numpy.concatenate([splits, numpy.array([tot - splits.sum()])])
assert splits.sum() == tot
codes = torch.split(torch.tensor(X.codice.values), list(splits))
seniorities = torch.split(torch.tensor(X.seniority.values, dtype=torch.float32), list(splits))
labels = torch.split(torch.tensor(X.y.values), list(splits))
# Since they are all equal.
labels = [t[0] for t in labels]
del s, indexes, tot, splits

class TensorListDataset(torch.utils.data.Dataset):

	def __init__(self, *tensors_lists) -> None:
		lengths = [len(tensors_list) for tensors_list in tensors_lists]
		if not all(lengths[0] == length for length in lengths):
			raise ValueError("All lists of tensors must have the same length.")
		# NOTE: all list should contain only tensors.
		self.tensors_lists = tensors_lists

	def __len__(self) -> int:
		return len(self.tensors_lists[0])

	def __getitem__(self, index):
		return tuple(tensors_list[index] for tensors_list in self.tensors_lists)

dataset = TensorListDataset(seniorities, codes, labels)

def collate_fn(batch):
	seniorities = torch.nn.utils.rnn.pad_sequence(
		[seniority for seniority, _, _ in batch],
		batch_first=True,
		padding_value=0.0
	)
	codes = torch.nn.utils.rnn.pad_sequence([code for _, code, _ in batch], True, 0)
	labels = torch.tensor([label for _, _, label in batch])
	return seniorities, codes, labels

batch_size = 128
# TODO: find a way to shuffle them consistently.
dataloader = torch.utils.data.DataLoader(
	dataset=dataset,
	batch_size=batch_size,
	shuffle=True,
	collate_fn=collate_fn
)

class Model(torch.nn.Module):

	def __init__(
			self,
			num_embeddings: int,
			embedding_dim: int,
			lstm_hidden_size: int,
			lstm_num_layers: int,
			lstm_bias: bool,
			lstm_batch_first: bool,
			lstm_dropout: float,
			lstm_bidirectional: bool,
			lstm_proj_size: int
		) -> None:
		super().__init__()

		self.codes_embeddings = torch.nn.Embedding(
			num_embeddings, embedding_dim, padding_idx=0
		)

		self.lstm = torch.nn.LSTM(
			embedding_dim + 1, # The additional dimension is for seniority.,
			lstm_hidden_size,
			lstm_num_layers,
			lstm_bias,
			lstm_batch_first,
			lstm_dropout,
			lstm_bidirectional,
			lstm_proj_size
		)

		# if lstm_bidirectional the input size doubles
		classifier_input_size = 2*lstm_hidden_size if lstm_bidirectional \
			else lstm_hidden_size
		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(classifier_input_size, classifier_input_size//2),
			torch.nn.ReLU(),
			torch.nn.Linear(classifier_input_size//2, 1)
		)

	def forward(
			self,
			X_seniority: torch.Tensor,
			X_codes: torch.Tensor,
		) -> torch.Tensor:
		assert X_seniority.shape[:2] == X_codes.shape[:2]

		X_codes_embeddings = self.codes_embeddings(X_codes)
		assert X_codes_embeddings.dtype == X_seniority.dtype
		# batch, len, dim
		X = torch.cat([X_seniority.unsqueeze(2), X_codes_embeddings], 2)

		o, (h, c) = self.lstm(X)

		res = self.classifier(o) # logits

		return res

net = Model(
	num_embeddings=len(codes),
	embedding_dim=100,
	lstm_hidden_size=50,
	lstm_num_layers=3,
	lstm_bias=True,
	lstm_batch_first=True,
	lstm_dropout=0.1, # Probability of removing.
	lstm_bidirectional=True,
	lstm_proj_size=0 # I don't know what this does.
)
print(net)
sen, cod, target = next(iter(dataloader))
logits = net(sen, cod)

