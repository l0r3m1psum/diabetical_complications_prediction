import torch
import pandas
import matplotlib.pyplot
import multiprocessing.pool
import logging

names = [
	'anagraficapazientiattivi',
	'diagnosi',
	'esamilaboratorioparametri',
	'esamilaboratorioparametricalcolati',
	'esamistrumentali',
	'prescrizionidiabetefarmaci',
	'prescrizionidiabetenonfarmaci',
	'prescrizioninondiabete',
]
paths_for_cleaned = [f'data/{name}_clean.csv' for name in names]

logging.info('Loading data.')
with multiprocessing.pool.ThreadPool(len(names)) as pool:
	globals().update(dict(zip(names, pool.map(pandas.read_csv, paths_for_cleaned))))
del pool

anagraficapazientiattivi = anagraficapazientiattivi.set_index(['idcentro', 'idana'])
anagraficapazientiattivi[~anagraficapazientiattivi.y]

# Most tables have no intersections among their codiceamd. Except for *. This
# can help find optimal encodings for the AMD codes to give to the LSTMs, since
# we can get away with not using a single encoding for all AMD codes.
sets = [
	set(diagnosi.codiceamd),
	set(esamilaboratorioparametri.codiceamd), # *
	set(esamilaboratorioparametricalcolati.codiceamd), # *
	set(esamistrumentali.codiceamd),
	set(prescrizionidiabetenonfarmaci.codiceamd),
	set(prescrizioninondiabete.codiceamd),
]

for j in range(len(sets)):
	for i in range(len(sets)):
		print(int(bool(sets[i] & sets[j])), '', end='')
	print()

# https://github.com/duskybomb/tlstm/blob/master/tlstm.py

n_features = 1
lstm = torch.nn.LSTM(
	input_size=n_features+1, # The +1 is the time for the T-LSTM
	hidden_size=n_features, # TODO: optimize this.
	num_layers=2, # TODO: optimize this.
	bias=True,
	batch_first=True,
	dropout=0.1,
	bidirectional=False, # NOTE: Can this be interpreted as looking in the future?
	proj_size=0 # NOTE: I don't know what this does.
)

X = torch.Tensor([[0.2, 0.6, 0.7], [1, 2, 3]]).T
output, (h_n, c_n) = lstm(X.unsqueeze(0))

# LSTM1 \
# LSTM2 -> Fully connected -> classificazione.
# LSMT3 /

# diagnosi                           = (data, codiceamd, valore)
# esamilaboratorioparametri          = (data, codiceamd, valore)
# esamilaboratorioparametricalcolati = (data, codiceamd, valore, codicestitch)
# esamistrumentali                   = (data, codiceamd, valore)
# prescrizionidiabetefarmaci         = (data, codiceatc, quantita, idpasto, descrizionefarmaco)
# prescrizionidiabetenonfarmaci      = (data, codiceamd, valore)
# prescrizioninondiabete             = (data, codiceamd, valore)

class Model(torch.nn.Module):
	def __init__(self) -> None:
		self.lstm_diagnosi           = torch.nn.LSTM(1, 1)
		self.lstm_elabparam          = torch.nn.LSTM(1, 1)
		self.lstm_elabparamcalcolati = torch.nn.LSTM(1, 1)
		self.lstm_estrumentali       = torch.nn.LSTM(1, 1)
		self.lstm_pdiabetefarmaci    = torch.nn.LSTM(1, 1)
		self.lstm_pdiabetenonfarmaci = torch.nn.LSTM(1, 1)
		self.lstm_pnondiabete        = torch.nn.LSTM(1, 1)
		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(7, 4),
			torch.nn.ReLU(),
			torch.nn.Linear(4, 1),
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

		# e = esami
		# p = prescrizioni
		# labparam = laboratorioparametri

		o_diagnosi           (h_diagnosi,           c_diagnosi)           = self.lstm_diagnosi(X_diagnosi)
		o_elabparam          (h_elabparam,          c_elabparam)          = self.lstm_elabparam(X_elabparam)
		o_elabparamcalcolati (h_elabparamcalcolati, c_elabparamcalcolati) = self.lstm_elabparamcalcolati(X_elabparamcalcolati)
		o_estrumentali       (h_estrumentali,       c_estrumentali)       = self.lstm_estrumentali(X_estrumentali)
		o_pdiabetefarmaci    (h_pdiabetefarmaci,    c_pdiabetefarmaci)    = self.lstm_pdiabetefarmaci(X_pdiabetefarmaci)
		o_pdiabetenonfarmaci (h_pdiabetenonfarmaci, c_pdiabetenonfarmaci) = self.lstm_pdiabetenonfarmaci(X_pdiabetenonfarmaci)
		o_pnondiabete        (h_pnondiabete,        c_pnondiabete)        = self.lstm_pnondiabete(X_pnondiabete)

		H = torch.cat(
			[h_diagnosi,
			h_elabparam,
			h_elabparamcalcolati,
			h_estrumentali,
			h_pdiabetefarmaci,
			h_pdiabetenonfarmaci,
			h_pnondiabete],
			0
		)

		res = self.classifier(H)

		return res
