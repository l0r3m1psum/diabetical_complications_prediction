import sys
sys.path.append('src')
import random
import optuna
from common import *
import torch
import numpy

logging.info('Loading data.')
with multiprocessing.pool.ThreadPool(len(names)) as pool:
	globals().update(dict(zip(names, pool.map(pandas.read_pickle, paths_for_cleaned))))
del pool

seed = 42
rng = numpy.random.default_rng(seed)
batch_size = 32
torch.manual_seed(seed)

#loading the database
X = pandas.read_pickle("data/X_clean.pickle.zip")

#cleaning the data for the model#
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

def collate_fn(batch):
	seniorities = torch.nn.utils.rnn.pad_sequence(
			[seniority for seniority, _, _, _ in batch],
			batch_first=True,
			padding_value=0.0
		)
	codes = torch.nn.utils.rnn.pad_sequence([code for _, code, _, _ in batch], True, 0)
	micro_codes = torch.nn.utils.rnn.pad_sequence([micro_code for _, _, micro_code, _ in batch], True, 0)
	labels = torch.tensor([label for _, _, _, label in batch])

	return seniorities, codes, micro_codes, labels

train_dataset = TensorListDataset(train_seniorities, train_codes, train_micro_codes, train_labels)
test_dataset = TensorListDataset(test_seniorities, test_codes, test_micro_codes, test_labels)

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
			lstm_proj_size: int,
			mlp_output_size: int
		) -> None:
		super().__init__()

		self.codes_embeddings = torch.nn.Embedding(
			num_embeddings, embedding_dim, padding_idx=0
		)

		self.micro_codes_embeddings = torch.nn.Embedding(
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

		self.mlp = torch.nn.Sequential(
    # Input layer
    torch.nn.Linear(embedding_dim, embedding_dim*4),
    torch.nn.ReLU(),
    # Hidden layers
    torch.nn.Linear(embedding_dim*4, embedding_dim*2),
    torch.nn.ReLU(),
    torch.nn.Linear(embedding_dim*2, embedding_dim),
    torch.nn.ReLU(),
    # Output layer
    torch.nn.Linear(embedding_dim, mlp_output_size),
	)

		# if lstm_bidirectional the input size doubles
		classifier_input_size = 2*lstm_hidden_size + mlp_output_size if lstm_bidirectional \
			else lstm_hidden_size + mlp_output_size

		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(classifier_input_size, classifier_input_size//2),
			torch.nn.ReLU(),
			torch.nn.Linear(classifier_input_size//2, 1)
		)



	def forward(
			self,
			X_seniority: torch.Tensor,
			X_codes: torch.Tensor,
			X_micro_codes: torch.Tensor
		) -> torch.Tensor:
		assert X_seniority.shape[:2] == X_codes.shape[:2]

		X_codes_embeddings = self.codes_embeddings(X_codes)
		assert X_codes_embeddings.dtype == X_seniority.dtype
		# batch, len, dim
		X = torch.cat([X_seniority.unsqueeze(2), X_codes_embeddings], 2)

		o, (h, c) = self.lstm(X)

		X_micro_codes_embeddings = self.codes_embeddings(X_micro_codes)

		X_micro = self.mlp(X_micro_codes_embeddings)
		micro_state = torch.mean(X_micro, dim=1)

		concat = torch.cat((h[-1], micro_state), dim=1)

		res = self.classifier(concat) # logits

		return res


def train(
		net: torch.nn.Module,
		epochs: int,
		patience: int,
		train_dataloader: torch.utils.data.DataLoader,
		logit_normalizer: torch.nn.Module,
		label_postproc: torch.nn.Module,
		criterion: torch.nn.Module,
		optimizer, # no type here :( torch.optimizer.Optimizer
		test_dataloader: torch.utils.data.DataLoader
	) -> float:
	patience_kept = 0
	best_epoch = 0
	best_accuracy = float('-inf')

	for epoch in range(epochs):
		if patience_kept >= patience: break

		net.train()
		losses: list[float] = []
		for i, (seniorities, codes, micro_codes, labels) in enumerate(train_dataloader):
			# NOTE: should tensors be moved here to device instead of a priori?
			seniorities: torch.Tensor
			codes: torch.Tensor
			micro_codes: torch.Tensor
			labels: torch.Tensor

			logits = net(seniorities, codes, micro_codes)
			predictions = logit_normalizer(logits)

			loss = criterion(predictions.squeeze(), label_postproc(labels).to(torch.float32).squeeze())
			losses.append(loss.item())

			loss.backward()

			optimizer.step()
			optimizer.zero_grad()
		avg_loss = torch.mean(torch.tensor(losses))

		net.eval()
		correct = 0
		with torch.no_grad():
			for seniorities, codes, micro_codes, labels in test_dataloader:
				logits = net(seniorities, codes, micro_codes)
				predictions = (logit_normalizer(logits) < 0.5).squeeze() #it should be 0.5

				# assert (predictions < N_CLASSES).all()
				# assert (labels < N_CLASSES).all()
				# assert predictions.shape == labels.shape, f"{predictions.shape} {labels.shape}"

				correct += (predictions == labels).sum().item()
		accuracy = correct/len(test_dataloader.dataset)
		assert accuracy <= 1.0

		if accuracy > best_accuracy:
			patience_kept = 0
			# best_params = net.params()
			best_epoch = epoch
			best_accuracy = accuracy
			marker = ' *'
		else:
			patience_kept += 1
			marker = ''

		print(f'{epoch=:02} {accuracy=:.3f} {avg_loss=:.3f}{marker}')

	print(f'{best_epoch=} {best_accuracy=:.3f}')
	return best_accuracy


def objective(trial):

	embedding_dim = trial.suggest_int("embedding_dim", 50, 100, step=10)
	lstm_hidden_size = trial.suggest_int("lstm_hidden_size", 8, 32, log=True)
	lstm_num_layers = trial.suggest_int("lstm_num_layers", 2, 3)
	lstm_dropout = trial.suggest_float("lstm_dropout", 0.1, 0.4, step=0.1)
	mlp_output_size = trial.suggest_int("mlp_output_size", 10, 100, step=0.1)

	net = Model(
			num_embeddings=len(codes), 
			embedding_dim=embedding_dim,
			lstm_hidden_size=lstm_hidden_size,
			lstm_num_layers=lstm_num_layers,
			lstm_bias=True,
			lstm_batch_first=True,
			lstm_dropout=lstm_dropout,
			lstm_bidirectional=False,
			lstm_proj_size=0,
			mlp_output_size=mlp_output_size
		)

	n_epochs = trial.suggest_int("n_epochs", 5, 25)
	learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

	accuracy = train(
		net,
		n_epochs,
		4,
		train_dataloader,
		torch.nn.Softmax(dim=1),
		torch.nn.Identity(),
		torch.nn.BCELoss(),
		torch.optim.SGD(net.parameters(), lr=learning_rate),
		test_dataloader
	)

	return -accuracy

study = optuna.create_study(study_name="Task3")
study.optimize(objective, n_trials=10)
best_params = study.best_params
print("Best accuracy: ", -study.best_value)
print("Best hyperparameters", best_params)