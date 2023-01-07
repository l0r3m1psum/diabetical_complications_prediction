import sys
sys.path.append('src')

from common import *
import torch

logging.info('Loading data.')
with multiprocessing.pool.ThreadPool(len(names)) as pool:
	globals().update(dict(zip(names, pool.map(pandas.read_pickle, paths_for_cleaned))))
del pool

#Point 1

#given a dataset, for patients with positive labels, delete events happened in the last six months
def clean_last_six_months(df):
	df = df.merge(positive_patients)
	df= df.set_index(['idcentro', 'idana'])
	grouped_df = df.groupby(['idcentro','idana'])
	filtered_data = []
	# Iterate through each patient's group
	for name, group in grouped_df:
	# Get the most recent event for this patient
		max_date = group['data'].max()
	# Filter the events to only keep those that are more than 6 months from the most recent event
		filtered_group = group[group['data'] < max_date - pandas.DateOffset(months=6)]
	# Append the filtered events to the list
		filtered_data.append(filtered_group)
	# Concatenate all of the filtered data into a single DataFrame
	filtered_df = pandas.concat(filtered_data)
	return filtered_df

#select patients with cardiovascular problems (i.e. y=1)
positive_patients = anagraficapazientiattivi[anagraficapazientiattivi.y].index.to_frame().reset_index(drop=True)

clean_last_six_months(diagnosi)
clean_last_six_months(esamilaboratorioparametri)
clean_last_six_months(esamilaboratorioparametricalcolati)
clean_last_six_months(esamistrumentali)
#are prescrizioni considered events too?
clean_last_six_months(prescrizionidiabetefarmaci)
clean_last_six_months(prescrizionidiabetenonfarmaci)
clean_last_six_months(prescrizioninondiabete)

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
num_table = pandas.concat([esamilaboratorioparametri, esamilaboratorioparametricalcolati])
obj_table = pandas.concat([diagnosi, esamistrumentali, prescrizionidiabetenonfarmaci, prescrizioninondiabete])
# TODO: load cleaned AMD data.
#assert (obj_table.join(amd, 'codiceamd').tipo == 'Testo').all()
# Woohoo no more work needed!

(sampling_date - anagraficapazientiattivi.annonascita).astype('<m8[Y]')

# TODO: encode dates for LSTM between 0 and 1
# 1 trovare il paziente più anziano usando sampling_date.
# 2 stabilire un valore di età oltre il cui la data dell'esame sarà 1.0 (e.g. tutti gli esami fatti dopo 90 anni)
# 3 a questo punto si possono trasformare tutte le date (per ogni paziente) in un valore tra 0 e 1.

# LSTM1 \
# LSTM2 -> Fully connected -> classificazione.
# LSMT3 /

# diagnosi                           = (data, codiceamd, valore)
# esamilaboratorioparametri          = (data, codiceamd, valore) *
# esamilaboratorioparametricalcolati = (data, codiceamd, valore, codicestitch) *
# esamistrumentali                   = (data, codiceamd, valore)
# prescrizionidiabetefarmaci         = (data, codiceatc, quantita, idpasto, descrizionefarmaco)
# prescrizionidiabetenonfarmaci      = (data, codiceamd, valore)
# prescrizioninondiabete             = (data, codiceamd, valore)

class Model(torch.nn.Module):
	"""In this class the names are shortened as such for readability:
	  * e = esami
	  * p = prescrizioni
	  * labparam = laboratorioparametri
	"""
	def __init__(
			self,
			diagnosi_input_size: int,
			diagnosi_hidden_size: int,
			elabparam_input_size: int,
			elabparam_hidden_size: int,
			elabparamcalcolati_input_size: int,
			elabparamcalcolati_hidden_size: int,
			estrumentali_input_size: int,
			estrumentali_hidden_size: int,
			pdiabetefarmaci_input_size: int,
			pdiabetefarmaci_hidden_size: int,
			pdiabetenonfarmaci_input_size: int,
			pdiabetenonfarmaci_hidden_size: int,
			pnondiabete_input_size: int,
			pnondiabete_hidden_size: int,
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

		self.lstm_diagnosi           = torch.nn.LSTM(diagnosi_input_size, diagnosi_hidden_size, **common_args)
		self.lstm_elabparam          = torch.nn.LSTM(elabparam_input_size, elabparam_hidden_size, **common_args)
		self.lstm_elabparamcalcolati = torch.nn.LSTM(elabparamcalcolati_input_size, elabparamcalcolati_hidden_size, **common_args)
		self.lstm_estrumentali       = torch.nn.LSTM(estrumentali_input_size, estrumentali_hidden_size, **common_args)
		self.lstm_pdiabetefarmaci    = torch.nn.LSTM(pdiabetefarmaci_input_size, pdiabetefarmaci_hidden_size, **common_args)
		self.lstm_pdiabetenonfarmaci = torch.nn.LSTM(pdiabetenonfarmaci_input_size, pdiabetenonfarmaci_hidden_size, **common_args)
		self.lstm_pnondiabete        = torch.nn.LSTM(pnondiabete_input_size, pnondiabete_hidden_size, **common_args)

		classifier_input_size = (diagnosi_hidden_size + elabparam_hidden_size
			+ elabparamcalcolati_hidden_size + estrumentali_hidden_size
			+ pdiabetefarmaci_hidden_size + pdiabetenonfarmaci_hidden_size
			+ pnondiabete_hidden_size)
		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(classifier_input_size, classifier_input_size//2),
			torch.nn.ReLU(),
			torch.nn.Linear(classifier_input_size//2, 1)
		)

	def forward(self,
			X_diagnosi:           torch.Tensor,
			X_elabparam:          torch.Tensor,
			X_elabparamcalcolati: torch.Tensor,
			X_estrumentali:       torch.Tensor,
			X_pdiabetefarmaci:    torch.Tensor,
			X_pdiabetenonfarmaci: torch.Tensor,
			X_pnondiabete:        torch.Tensor
		) -> torch.Tensor:

		# NOTE: This step can be parallelized.
		o_diagnosi,           (h_diagnosi,           c_diagnosi)           = self.lstm_diagnosi(X_diagnosi)
		o_elabparam,          (h_elabparam,          c_elabparam)          = self.lstm_elabparam(X_elabparam)
		o_elabparamcalcolati, (h_elabparamcalcolati, c_elabparamcalcolati) = self.lstm_elabparamcalcolati(X_elabparamcalcolati)
		o_estrumentali,       (h_estrumentali,       c_estrumentali)       = self.lstm_estrumentali(X_estrumentali)
		o_pdiabetefarmaci,    (h_pdiabetefarmaci,    c_pdiabetefarmaci)    = self.lstm_pdiabetefarmaci(X_pdiabetefarmaci)
		o_pdiabetenonfarmaci, (h_pdiabetenonfarmaci, c_pdiabetenonfarmaci) = self.lstm_pdiabetenonfarmaci(X_pdiabetenonfarmaci)
		o_pnondiabete,        (h_pnondiabete,        c_pnondiabete)        = self.lstm_pnondiabete(X_pnondiabete)

		H = torch.cat(
			[h_diagnosi,
			h_elabparam,
			h_elabparamcalcolati,
			h_estrumentali,
			h_pdiabetefarmaci,
			h_pdiabetenonfarmaci,
			h_pnondiabete],
			2 # Concat along the features.
		)

		res = self.classifier(H) # logits

		return res

net = Model(
	5, 5, # diagnosi
	5, 5, # esamilaboratorioparametri
	5, 5, # esamilaboratorioparametricalcolati
	5, 5, # esamistrumentali
	5, 5, # prescrizionidiabetefarmaci
	5, 5, # prescrizionidiabetenonfarmaci
	5, 5, # prescrizioninondiabete
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
	torch.ones(1, 10, 5), # diagnosi
	torch.ones(1, 10, 5), # esamilaboratorioparametri
	torch.ones(1, 10, 5), # esamilaboratorioparametricalcolati
	torch.ones(1, 10, 5), # esamistrumentali
	torch.ones(1, 10, 5), # prescrizionidiabetefarmaci
	torch.ones(1, 10, 5), # prescrizionidiabetenonfarmaci
	torch.ones(1, 10, 5)  # prescrizioninondiabete
)
