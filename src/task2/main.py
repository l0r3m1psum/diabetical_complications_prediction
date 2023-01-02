import torch

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
