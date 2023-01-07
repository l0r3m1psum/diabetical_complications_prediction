import logging
import multiprocessing.pool

import pandas

logging.basicConfig(format='%(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

def is_float(x: any) -> bool:
	try:
		_ = float(x)
	except (ValueError, TypeError):
		return False
	return True

def source(fname: str) -> None:
	with open(fname) as f:
		exec(f.read())

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
paths = [f'data/{name}.csv' for name in names]
# Where the cleaned data will be saved.
paths_for_cleaned = [f'data/{name}_clean.pickle.zip' for name in names]

# The date in which the dataset was sampled.
# TODO: find the real one.
sampling_date = pandas.Timestamp(year=2022, month=1, day=1)

# This are also the cardiovascular events.
macro_vascular_diseases = ['AMD047', 'AMD048', 'AMD049', 'AMD071', 'AMD081', 'AMD082', 'AMD208', 'AMD303']

# TODO: this should take an optional argument (e.g. a lambda) to select what to
# print from all the data frames.
def print_all() -> None:
	# Evil hack since globals() is bound to the globals in this module and not
	# the calling one.
	import inspect
	called_from = inspect.stack()[1]
	for name in names: print(name, called_from.frame.f_globals[name], sep='\n')
