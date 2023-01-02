import pandas
import matplotlib.pyplot
import multiprocessing.pool
import logging

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

# Data loading and definition ##################################################

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

logging.info('Loading data.')
with multiprocessing.pool.ThreadPool(len(names)) as pool:
	globals().update(dict(zip(names, pool.map(pandas.read_csv, paths))))
del pool

codice_amd = pandas.Categorical(f'AMD{i:03}' for i in range(1, 1000))
codice_stitch = pandas.Categorical(f'STITCH{i:03}' for i in range(1, 6))
sex = pandas.Categorical(['M', 'F'])
meal_id = pandas.Categorical(i for i in range(1, 7))
# NOTE: should I create a sepatate Categorical for codiceatc too?

# The date in which the dataset was sampled.
# TODO: find the real one.
sampling_date = pandas.Timestamp(year=2022, month=1, day=1)

def print_all() -> None:
	for name in names: print(name, globals()[name], sep='\n')

macro_vascular_diseases = codice_amd.take(pandas.Series([47, 48, 49, 71, 81, 82, 208, 303])-1)

# Initial data cleaning ########################################################

# TODO: idcentro can probably be and int16 and idana can probabbly be an int32
# to use less memory.

# The first colum of each table contains the row number which is useless for us.
def remove_first_column(df: pandas.DataFrame) -> pandas.DataFrame:
	res = df.drop(df.columns[0], axis=1)
	return res

logging.info(f'Before initial cleaning: {len(anagraficapazientiattivi)=}')
anagraficapazientiattivi = remove_first_column(anagraficapazientiattivi)
anagraficapazientiattivi = anagraficapazientiattivi.set_index(['idcentro', 'idana'], verify_integrity=True)
anagraficapazientiattivi.annodiagnosidiabete = pandas.to_datetime(anagraficapazientiattivi.annodiagnosidiabete.astype('Int16'), format='%Y')
anagraficapazientiattivi.annonascita = pandas.to_datetime(anagraficapazientiattivi.annonascita, format='%Y')
anagraficapazientiattivi.annoprimoaccesso = pandas.to_datetime(anagraficapazientiattivi.annoprimoaccesso.astype('Int16'), format='%Y')
anagraficapazientiattivi.annodecesso = pandas.to_datetime(anagraficapazientiattivi.annodecesso.astype('Int16'), format='%Y')
anagraficapazientiattivi.sesso = anagraficapazientiattivi.sesso.astype(sex.dtype)
# NOTE: this can probabbly be dropped since they are all equal to 5
anagraficapazientiattivi.scolarita = anagraficapazientiattivi.scolarita.astype('category')
anagraficapazientiattivi.statocivile = anagraficapazientiattivi.statocivile.astype('category')
anagraficapazientiattivi.professione = anagraficapazientiattivi.professione.astype('category')
anagraficapazientiattivi.origine = anagraficapazientiattivi.origine.astype('category')
anagraficapazientiattivi.tipodiabete = anagraficapazientiattivi.tipodiabete.astype('category')
# TODO: understand scolarita ,statocivile, professione, origine
assert not anagraficapazientiattivi.annonascita.isnull().any()
# Dropping inconsistent birth and deaths (keep in mind that comparisons on NaTs always return False).
anagraficapazientiattivi = anagraficapazientiattivi.drop(anagraficapazientiattivi[anagraficapazientiattivi.annonascita >= anagraficapazientiattivi.annodecesso].index)
# Dropping future deaths.
anagraficapazientiattivi = anagraficapazientiattivi.drop((anagraficapazientiattivi.annodecesso[~anagraficapazientiattivi.annodecesso.isnull()] < sampling_date).index)
def is_not_between(s: pandas.Series) -> pandas.MultiIndex:
	res = (anagraficapazientiattivi.annonascita < s) & (s < anagraficapazientiattivi.annodecesso.fillna(sampling_date))
	res = (anagraficapazientiattivi[~res]).index
	return res
# TODO: what should I do with NaTs in annodiagnosidiabete and annoprimoaccesso.
anagraficapazientiattivi = anagraficapazientiattivi.drop(is_not_between(anagraficapazientiattivi.annodiagnosidiabete))
anagraficapazientiattivi = anagraficapazientiattivi.drop(is_not_between(anagraficapazientiattivi.annoprimoaccesso))
logging.info(f'After  initial cleaning: {len(anagraficapazientiattivi)=}')
del is_not_between

#patient remaining TODO class distribution (i.e. with or without cardiovascular events)

diagnosi = remove_first_column(diagnosi)
diagnosi.data = pandas.to_datetime(diagnosi.data)
diagnosi.codiceamd = diagnosi.codiceamd.astype(codice_amd.dtype)
# TODO: understand valore
# wtf = diagnosi.valore[~diagnosi.valore.apply(is_float)].value_counts()

esamilaboratorioparametri = remove_first_column(esamilaboratorioparametri)
esamilaboratorioparametri.data = pandas.to_datetime(esamilaboratorioparametri.data)
esamilaboratorioparametri.codiceamd = esamilaboratorioparametri.codiceamd.astype(codice_amd.dtype)

esamilaboratorioparametricalcolati = remove_first_column(esamilaboratorioparametricalcolati)
esamilaboratorioparametricalcolati.data = pandas.to_datetime(esamilaboratorioparametricalcolati.data)
esamilaboratorioparametricalcolati.codiceamd = esamilaboratorioparametricalcolati.codiceamd.astype(codice_amd.dtype)
esamilaboratorioparametricalcolati.codicestitch = esamilaboratorioparametricalcolati.codicestitch.astype(codice_stitch.dtype)

esamistrumentali = remove_first_column(esamistrumentali)
esamistrumentali.data = pandas.to_datetime(esamistrumentali.data)
esamistrumentali.codiceamd = esamistrumentali.codiceamd.astype(codice_amd.dtype)
esamistrumentali.valore = esamistrumentali.valore.astype('category')

prescrizionidiabetefarmaci = remove_first_column(prescrizionidiabetefarmaci)
prescrizionidiabetefarmaci.data = pandas.to_datetime(prescrizionidiabetefarmaci.data)
# NOTE: A10BD is a probably malformed.
prescrizionidiabetefarmaci.codiceatc = prescrizionidiabetefarmaci.codiceatc.astype('category')
prescrizionidiabetefarmaci.idpasto = prescrizionidiabetefarmaci.idpasto.astype(meal_id.dtype)

prescrizionidiabetenonfarmaci = remove_first_column(prescrizionidiabetenonfarmaci)
prescrizionidiabetenonfarmaci.data = pandas.to_datetime(prescrizionidiabetenonfarmaci.data)
prescrizionidiabetenonfarmaci.codiceamd = prescrizionidiabetenonfarmaci.codiceamd.astype(codice_amd.dtype)
# TODO: understand valore

prescrizioninondiabete = remove_first_column(prescrizioninondiabete)
prescrizioninondiabete.data = pandas.to_datetime(prescrizioninondiabete.data)
prescrizioninondiabete.codiceamd = prescrizioninondiabete.codiceamd.astype(codice_amd.dtype)
# TODO: understand valore, is it categorical?

del remove_first_column

# Tasks 1 ######################################################################

logging.info('Start of task 1.')

# Point 1

# The diagnosi table is the only one containing data of patients with macro
# vascular diseases.
assert diagnosi.codiceamd.isin(macro_vascular_diseases).any()
assert not esamilaboratorioparametri.codiceamd.isin(macro_vascular_diseases).any()
assert not esamilaboratorioparametricalcolati.codiceamd.isin(macro_vascular_diseases).any()
assert not esamistrumentali.codiceamd.isin(macro_vascular_diseases).any()
assert not prescrizionidiabetenonfarmaci.codiceamd.isin(macro_vascular_diseases).any()
assert not prescrizioninondiabete.codiceamd.isin(macro_vascular_diseases).any()

logging.info(f'Before Point 1: {len(diagnosi)=}')
diagnosi = diagnosi[diagnosi.codiceamd.isin(macro_vascular_diseases)]

logging.info(f'Before Point 1: {len(anagraficapazientiattivi)=}')
anagraficapazientiattivi = diagnosi[['idcentro','idana']] \
	.join(anagraficapazientiattivi, ['idcentro','idana'], 'inner') \
	.drop_duplicates(['idcentro','idana']) \
	.set_index(['idcentro','idana'])
logging.info(f'After  Point 1: {len(anagraficapazientiattivi)=}')

# Used to make sure that there are no patients outside of this group in the other tables.
patients = anagraficapazientiattivi.index.to_frame().reset_index(drop=True)

diagnosi = diagnosi.merge(patients)
logging.info(f'After  Point 1: {len(diagnosi)=}')

logging.info(f'Before Point 1: {len(esamilaboratorioparametri)=}')
esamilaboratorioparametri = esamilaboratorioparametri.merge(patients)
logging.info(f'After  Point 1: {len(esamilaboratorioparametri)=}')

logging.info(f'Before Point 1: {len(esamilaboratorioparametricalcolati)=}')
esamilaboratorioparametricalcolati = esamilaboratorioparametricalcolati.merge(patients)
logging.info(f'After  Point 1: {len(esamilaboratorioparametricalcolati)=}')

logging.info(f'Before Point 1: {len(esamistrumentali)=}')
esamistrumentali = esamistrumentali.merge(patients)
logging.info(f'After  Point 1: {len(esamistrumentali)=}')

logging.info(f'Before Point 1: {len(prescrizionidiabetefarmaci)=}')
prescrizionidiabetefarmaci = prescrizionidiabetefarmaci.merge(patients)
logging.info(f'After  Point 1: {len(prescrizionidiabetefarmaci)=}')

logging.info(f'Before Point 1: {len(prescrizionidiabetenonfarmaci)=}')
prescrizionidiabetenonfarmaci = prescrizionidiabetenonfarmaci.merge(patients)
logging.info(f'After  Point 1: {len(prescrizionidiabetenonfarmaci)=}')

logging.info(f'Before Point 1: {len(prescrizioninondiabete)=}')
prescrizioninondiabete = prescrizioninondiabete.merge(patients)
logging.info(f'After  Point 1: {len(prescrizioninondiabete)=}')

del patients

# Point 2

birth_death = anagraficapazientiattivi[['annonascita', 'annodecesso']]

def clean_is_between(df: pandas.DataFrame) -> pandas.DataFrame:
	# We add birth and death information of each patient to the table.

	mdf = df.join(birth_death, ['idcentro','idana'], 'inner')
	assert len(mdf) == len(df)
	res = df[(mdf.annonascita <= mdf.data) & (mdf.data <= mdf.annodecesso.fillna(sampling_date))]
	return res

logging.info(f'Before point 2: {len(diagnosi)=}')
diagnosi = clean_is_between(diagnosi)
logging.info(f'After  point 2: {len(diagnosi)=}')

logging.info(f'Before point 2: {len(esamilaboratorioparametri)=}')
esamilaboratorioparametri = clean_is_between(esamilaboratorioparametri)
logging.info(f'After  point 2: {len(esamilaboratorioparametri)=}')

logging.info(f'Before point 2: {len(esamilaboratorioparametricalcolati)=}')
esamilaboratorioparametricalcolati = clean_is_between(esamilaboratorioparametricalcolati)
logging.info(f'After  point 2: {len(esamilaboratorioparametricalcolati)=}')

logging.info(f'Before point 2: {len(esamistrumentali)=}')
esamistrumentali = clean_is_between(esamistrumentali)
logging.info(f'After  point 2: {len(esamistrumentali)=}')

logging.info(f'Before point 2: {len(prescrizionidiabetefarmaci)=}')
prescrizionidiabetefarmaci = clean_is_between(prescrizionidiabetefarmaci)
logging.info(f'After  point 2: {len(prescrizionidiabetefarmaci)=}')

logging.info(f'Before point 2: {len(prescrizionidiabetenonfarmaci)=}')
prescrizionidiabetenonfarmaci = clean_is_between(prescrizionidiabetenonfarmaci)
logging.info(f'After  point 2: {len(prescrizionidiabetenonfarmaci)=}')

logging.info(f'Before point 2: {len(prescrizioninondiabete)=}')
prescrizioninondiabete = clean_is_between(prescrizioninondiabete)
logging.info(f'After  point 2: {len(prescrizioninondiabete)=}')

del birth_death, clean_is_between

# Point 3

def clean_same_month(df: pandas.DataFrame) -> pandas.DataFrame:
	group = df.groupby(['idcentro', 'idana'], group_keys=True).data
	# The visits are in different months iff the min month in the group for a
	# patient is different from the max of that group.
	serie = (group.min().dt.to_period('M') != group.max().dt.to_period('M'))
	# We create a dataframe with only the idcentro and idana of valid patients.
	frame = serie[serie].index.to_frame().reset_index(drop=True)
	res = df.merge(frame)
	return res

logging.info(f'Before point 3: {len(diagnosi)=}')
diagnosi = clean_same_month(diagnosi)
logging.info(f'After  point 3: {len(diagnosi)=}')

logging.info(f'Before point 3: {len(esamilaboratorioparametri)=}')
esamilaboratorioparametri = clean_same_month(esamilaboratorioparametri)
logging.info(f'After  point 3: {len(esamilaboratorioparametri)=}')

logging.info(f'Before point 3: {len(esamilaboratorioparametricalcolati)=}')
esamilaboratorioparametricalcolati = clean_same_month(esamilaboratorioparametricalcolati)
logging.info(f'After  point 3: {len(esamilaboratorioparametricalcolati)=}')

logging.info(f'Before point 3: {len(esamistrumentali)=}')
esamistrumentali = clean_same_month(esamistrumentali)
logging.info(f'After  point 3: {len(esamistrumentali)=}')

logging.info(f'Before point 3: {len(prescrizionidiabetefarmaci)=}')
prescrizionidiabetefarmaci = clean_same_month(prescrizionidiabetefarmaci)
logging.info(f'After  point 3: {len(prescrizionidiabetefarmaci)=}')

logging.info(f'Before point 3: {len(prescrizionidiabetenonfarmaci)=}')
prescrizionidiabetenonfarmaci = clean_same_month(prescrizionidiabetenonfarmaci)
logging.info(f'After  point 3: {len(prescrizionidiabetenonfarmaci)=}')

logging.info(f'Before point 3: {len(prescrizioninondiabete)=}')
prescrizioninondiabete = clean_same_month(prescrizioninondiabete)
logging.info(f'After  point 3: {len(prescrizioninondiabete)=}')

del clean_same_month

# Point 4

# Point 6
# TODO: remove NA, NaN and NaT from the data and plotting.

#sex is always present
assert anagraficapazientiattivi['sesso'].isna().sum() == 0

#tipodiabete is always 5, we can remove it
if (anagraficapazientiattivi.tipodiabete == 5).all():
	anagraficapazientiattivi = anagraficapazientiattivi.drop(columns = ['tipodiabete'])

dataset_len = len(anagraficapazientiattivi)
max_null_percentage = 40

null_scolarita = anagraficapazientiattivi.scolarita.isnull().sum()
if null_scolarita*100/dataset_len > max_null_percentage:
	anagraficapazientiattivi = anagraficapazientiattivi.drop(columns = ['scolarita'])
del null_scolarita

null_statocivile = anagraficapazientiattivi.statocivile.isnull().sum()
if null_statocivile*100/dataset_len > max_null_percentage:
	anagraficapazientiattivi = anagraficapazientiattivi.drop(columns = ['statocivile'])
del null_statocivile

null_professione = anagraficapazientiattivi.professione.isnull().sum()
if null_professione*100/dataset_len > max_null_percentage:
	anagraficapazientiattivi = anagraficapazientiattivi.drop(columns = ['professione'])
del null_professione

null_origine = anagraficapazientiattivi.origine.isnull().sum()
if null_origine*100/dataset_len > max_null_percentage:
	anagraficapazientiattivi = anagraficapazientiattivi.drop(columns = ['origine'])
del null_origine

del dataset_len, max_null_percentage

#matplotlib.pyplot.hist(anagraficapazientiattivi['scolarita'])
#matplotlib.pyplot.show()
