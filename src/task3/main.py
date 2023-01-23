import sys
sys.path.append('src')
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
batch_size = 128
torch.manual_seed(seed)

#####this will be sostituted by a load#######
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

#####this will be sostituted by a load#######


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

X_macro = X[X['codice'].isin(macro_vascular_diseases)]
X_micro = X[~X['codice'].isin(macro_vascular_diseases)]
assert len(X_micro) + len(X_macro) == len(X)

# Ordinal encoding of codice
codes = pandas.Series(numpy.concatenate([['PADDING'], numpy.sort(X.codice.unique())])).rename('codice').reset_index()

X_macro = X_macro.merge(codes).drop('codice', axis=1).rename({'index': 'codice'}, axis=1)
X_micro = X_micro.merge(codes).drop('codice', axis=1).rename({'index': 'codice'}, axis=1)
assert len(X_micro) + len(X_macro) == len(X)

unique_micro = X_micro[["idana","idcentro"]].drop_duplicates().reset_index(drop=True)
X_macro = X_macro.merge(unique_micro, 'inner', ['idcentro', 'idana'])
unique_macro = X_macro[["idana","idcentro"]].drop_duplicates().reset_index(drop=True)
X_micro = X_micro.merge(unique_macro, 'inner', ['idcentro', 'idana'])

# NOTE: this should be useless since we order again below.
#X = X.sort_values(['idcentro', 'idana', 'data']).reset_index(drop=True)
X_macro = X_macro.sort_values(['idcentro', 'idana', 'data']).reset_index(drop=True)
X_micro = X_micro.sort_values(['idcentro', 'idana', 'data']).reset_index(drop=True)

# To give the models an easier time understanding the date in which the event
# happened we scale it wrt how old the patient is. Any age above 100 years is
# considered to be the same as 100.
tmp = X_macro.join(anagraficapazientiattivi.annonascita, ['idcentro', 'idana'])
assert not tmp.data.isna().any()
assert not tmp.annonascita.isna().any()
seniority = (tmp.data - tmp.annonascita).astype('<m8[Y]').clip(None, 100.0)/100.0
X_macro['seniority'] = seniority
X_macro.drop('data', axis=1, inplace=True)
del tmp, seniority, X

X_micro.drop('data', axis=1, inplace=True) #we don't care about the order of the micro events
# Ordering columns just for convenience.
macro_columns_order = ['idana', 'idcentro', 'seniority', 'codice', 'y']
micro_columns_order = ['idana', 'idcentro', 'codice', 'y']
X_macro = X_macro.reindex(columns=macro_columns_order)
X_micro = X_micro.reindex(columns=micro_columns_order)

del macro_columns_order, micro_columns_order

# NOTE: We could add a feature that represents the table from which the data
# comes from.

# Here we create a tensor for the history of each patients.
# macro events with date
s = X_macro.sort_values(['idcentro', 'idana', 'seniority']).reset_index(drop=True)
indexes = numpy.nonzero(numpy.diff(s.idana.values))[0]+1
tot = len(X_macro)
splits = numpy.concatenate([indexes[:1], numpy.diff(indexes)])
splits = numpy.concatenate([splits, numpy.array([tot - splits.sum()])])
assert splits.sum() == tot
codes = torch.split(torch.tensor(X_macro.codice.values), list(splits))
seniorities = torch.split(torch.tensor(X_macro.seniority.values, dtype=torch.float32), list(splits))
labels = torch.split(torch.tensor(X_macro.y.values), list(splits))
# Since they are all equal.
labels = [t[0] for t in labels]
del s, indexes, tot, splits


#micro events without date
s = X_micro.sort_values(['idcentro', 'idana']).reset_index(drop=True)
indexes = numpy.nonzero(numpy.diff(s.idana.values))[0]+1
tot = len(X_micro)
splits = numpy.concatenate([indexes[:1], numpy.diff(indexes)])
splits = numpy.concatenate([splits, numpy.array([tot - splits.sum()])])
assert splits.sum() == tot
micro_codes = torch.split(torch.tensor(X_micro.codice.values), list(splits))
# Since they are all equal.
del s, indexes, tot, splits


split = int(len(labels)*.8) # 80% of the data
train_codes = codes[:split]
train_seniorities = seniorities[:split]
train_micro_codes = micro_codes[:split]
train_labels = labels[:split]

test_codes = codes[split:]
test_seniorities = seniorities[split:]
test_micro_codes = micro_codes[split:]
test_labels = labels[split:]
assert len(train_codes) + len(test_codes) == len(codes)
assert len(train_seniorities) + len(test_seniorities) == len(seniorities)
assert len(train_labels) + len(test_labels) == len(labels)
assert len(train_micro_codes) + len(test_micro_codes) == len(micro_codes)
del split

print(len(micro_codes), len(codes))

class TensorListDataset(torch.utils.data.Dataset):

		def __init__(self, *tensors_lists) -> None:
			lengths = [len(tensors_list) for tensors_list in tensors_lists]
			for tensors_list in tensors_lists:
				print(len(tensors_list))
			if not all(lengths[0] == length for length in lengths):
				raise ValueError("All lists of tensors must have the same length.")
			# NOTE: all list should contain only tensors.
			self.tensors_lists = tensors_lists

		def __len__(self) -> int:
			return len(self.tensors_lists[0])

		def __getitem__(self, index):
			return tuple(tensors_list[index] for tensors_list in self.tensors_lists)

train_dataset = TensorListDataset(train_seniorities, train_codes, train_micro_codes, train_labels)
test_dataset = TensorListDataset(test_seniorities, test_codes, test_micro_codes, test_labels)

def collate_fn(batch):
	seniorities = torch.nn.utils.rnn.pad_sequence(
			[seniority for seniority, _, _, _ in batch],
			batch_first=True,
			padding_value=0.0
		)
	codes = torch.nn.utils.rnn.pad_sequence([code for _, code, _, _ in batch], True, 0)
	micro_codes = torch.nn.utils.rnn.pad_sequence([micro_code for _, _, micro_code, _ in batch], True, 0)
	labels = torch.tensor([label for _, _, _, label in batch])
	return seniorities, codes, microcodes, labels

train_dataloader = torch.utils.data.DataLoader(
		dataset=train_dataset,
		batch_size=batch_size,
		shuffle=True,
		collate_fn=collate_fn
)
test_dataloader = torch.utils.data.DataLoader(
		dataset=test_dataset,
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

		res = self.classifier(h) # logits

		return res


"""import optuna
from optuna.visualization import plot_contour, plot_edf, plot_optimization_history,\
  plot_parallel_coordinate, plot_param_importances, plot_slice  

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#each patient is represented by two tensors
#TENSOR 1: contains only the macroevents of the patient, *with* timestamps 
#TENSOR 2: contains only the microevents, without timesamps, processed by an invariant LSTM
#they are later combined into a fully connected layer
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.set2set = nn.Set2Set(input_size, processing_steps=2)
        self.fc = nn.Linear(hidden_size + input_size, output_size)

    def forward(self, x1, x2):
        x1, _ = self.lstm(x1)
        x1 = x1[-1]
        x2 = self.set2set(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x


# Define the objective function for Optuna
def objective(trial):
    # Sample the hyperparameters
    input_size = trial.suggest_int("input_size", 50, 300)
    hidden_size = trial.suggest_int("hidden_size", 50, 300)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    num_classes = 2
    batch_size = trial.suggest_int("batch_size", 8, 64)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-10, 1e-3)
    num_epochs = trial.suggest_int("num_epochs", 10, 100)
    
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model and move it to the device
    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Define a DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # Training loop
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

			trial.report(loss, i+1)

    		if trial.should_prune():
      			raise optuna.TrialPruned()

  	return loss
           
study = optuna.create_study(study_name="RidgeRegression")
study.optimize(objective, n_trials=15)"""