import sys
sys.path.append('src')
import random

from common import *
import torch

logging.info('Loading data.')
with multiprocessing.pool.ThreadPool(len(names)) as pool:
	globals().update(dict(zip(names, pool.map(pandas.read_pickle, paths_for_cleaned))))
del pool

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

logging.info('Starting cleaning last six months of history')

diagnosi = clean_last_six_months(diagnosi)
logging.info(f'{len(esamilaboratorioparametri)=}')
esamilaboratorioparametri = clean_last_six_months(esamilaboratorioparametri)
logging.info(f'{len(esamilaboratorioparametri)=}')
esamilaboratorioparametricalcolati = clean_last_six_months(esamilaboratorioparametricalcolati)
esamistrumentali = clean_last_six_months(esamistrumentali)
prescrizionidiabetefarmaci = clean_last_six_months(prescrizionidiabetefarmaci)
prescrizionidiabetenonfarmaci = clean_last_six_months(prescrizionidiabetenonfarmaci)
prescrizioninondiabete = clean_last_six_months(prescrizioninondiabete)

del all_events, last_event

m = 3
assert m > 0
new_idana_start_point = 1000000
seed = 42

# NOTE: saddly just a bijection between new and old idana is not enough because
# each new group of duplicated patiens needs a bijection between new and old
# idana. Because otherwise if we duplicate the data n times there are going to
# be n copies of the patients with the new idana. A possible solution could be
# to add a new row containing a unique number for each duplicated block and
# create a bijection based on idana and this row.
copied_positive_patients = pandas.concat([positive_patients]*(m-1), ignore_index=True)
idana = positive_patients.idana.unique()
idana_conv = pandas.DataFrame({ # The bijection.
	'idana': idana,
	'new': pandas.Series([new_idana_start_point + i for i in range(len(idana))])
})
del idana

import numpy

rng = numpy.random.default_rng(seed)

def naive_balancing(df: pandas.DataFrame) -> pandas.DataFrame:
	"""This function does 5 things:
	  1. duplicates the events for the positive patients in this dataframe
	  2. removes some of this events
	  3. perturbates the date of thi events
	  6. cleans the last six months
	  4. update the idana
	"""
	removed_frac = 0.01
	copied_positive_patients_df = df.merge(copied_positive_patients, 'inner', ['idcentro', 'idana'])
	assert len(copied_positive_patients_df) == (m-1)*len(df.merge(positive_patients, 'inner', ['idcentro', 'idana']))
	copied_positive_patients_df = copied_positive_patients_df.drop(
		copied_positive_patients_df.sample(None, removed_frac, False).index
	).reset_index(drop=True)
	offsets = rng.normal(0, 3, len(copied_positive_patients_df)).astype('int')
	pert = pandas.to_timedelta(offsets, unit='d')
	copied_positive_patients_df.data = copied_positive_patients_df.data + pert
	copied_positive_patients_df = clean_last_six_months(copied_positive_patients_df).reset_index(drop=True)
	assert copied_positive_patients_df.idana.isin(idana_conv.idana).all()
	copied_positive_patients_df.idana = copied_positive_patients_df.merge(idana_conv).new
	return copied_positive_patients_df

# TODO: add new patients to anagraficapazientiattivi
diagnosi = pandas.concat([diagnosi, naive_balancing(diagnosi)], ignore_index=True)
esamilaboratorioparametri = pandas.concat([esamilaboratorioparametri, naive_balancing(esamilaboratorioparametri)], ignore_index=True)
esamilaboratorioparametricalcolati = pandas.concat([esamilaboratorioparametricalcolati, naive_balancing(esamilaboratorioparametricalcolati)], ignore_index=True)
esamistrumentali = pandas.concat([esamistrumentali, naive_balancing(esamistrumentali)], ignore_index=True)
prescrizionidiabetefarmaci = pandas.concat([prescrizionidiabetefarmaci, naive_balancing(prescrizionidiabetefarmaci)], ignore_index=True)
prescrizionidiabetenonfarmaci = pandas.concat([prescrizionidiabetenonfarmaci, naive_balancing(prescrizionidiabetenonfarmaci)], ignore_index=True)
prescrizioninondiabete = pandas.concat([prescrizioninondiabete, naive_balancing(prescrizioninondiabete)], ignore_index=True)

# Point 2

# TODO: for **concentration** rebalancig perturbate anagraficapazientiattivi too and the values of the events.

# Deep Learning Stuff ##########################################################

# NOTE: one hot encoding AMD codes could be very large, to improve performance
# we could do a random projection to a smaller input vector.
# NOTE: We could add a feature that represents the table from which the data
# comes from.

# Since the only case in which STITCH codes convey useful information is when
# the AMD code is NA we the a simple substitution of the NAs with the STITCH
# codes and remove the column entirely.
assert (esamilaboratorioparametricalcolati[esamilaboratorioparametricalcolati.codiceamd == 'AMD927'].codicestitch == 'STITCH001').all()
assert (esamilaboratorioparametricalcolati[esamilaboratorioparametricalcolati.codiceamd == 'AMD013'].codicestitch == 'STITCH002').all()
assert (esamilaboratorioparametricalcolati[esamilaboratorioparametricalcolati.codiceamd == 'AMD304'].codicestitch == 'STITCH005').all()
assert esamilaboratorioparametricalcolati[esamilaboratorioparametricalcolati.codiceamd.isnull()].codicestitch.isin(['STITCH003', 'STITCH004']).all()
esamilaboratorioparametricalcolati.codiceamd.update(
	esamilaboratorioparametricalcolati[esamilaboratorioparametricalcolati.codiceamd.isnull()].codicestitch
)
assert not esamilaboratorioparametricalcolati.codiceamd.isna().any()
esamilaboratorioparametricalcolati = esamilaboratorioparametricalcolati.drop('codicestitch', axis=1)

# Now we try to group all the tables in 2 categories the ones with numeric value
# and the ones with string values.
num_table = pandas.concat([esamilaboratorioparametri, esamilaboratorioparametricalcolati], ignore_index=True)
txt_table = pandas.concat([diagnosi, esamistrumentali, prescrizionidiabetenonfarmaci, prescrizioninondiabete], ignore_index=True)
# TODO: load cleaned AMD data.
#assert (txt_table.join(amd, 'codiceamd').tipo == 'Testo').all()
# Woohoo no more work needed!

# The histogram clearly shows that the majority of patients are old.
# ages = (sampling_date - anagraficapazientiattivi.annonascita).astype('<m8[Y]').rename('eta')

# To give the models an easier time understanding the date in which the event
# happened we scale it wrt how old the patient is. Any age above 100 years is
# considered to be the same as 100.
def add_seniority_level(df: pandas.DataFrame) -> None:
	tmp = diagnosi.join(anagraficapazientiattivi.annonascita, ['idcentro', 'idana'])
	seniority = (tmp.data - tmp.annonascita).astype('<m8[Y]').clip(None, 100.0)/100.0
	df['seniority'] = seniority

add_seniority_level(num_table)
add_seniority_level(txt_table)

del add_seniority_level



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
