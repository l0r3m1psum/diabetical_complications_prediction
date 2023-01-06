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

# This are also the cardiovascular events.
macro_vascular_diseases = codice_amd.take(pandas.Series([47, 48, 49, 71, 81, 82, 208, 303])-1)

# Where the cleaned data will be saved.
paths_for_cleaned = [f'data/{name}_clean.csv' for name in names]

# Initial data cleaning ########################################################

logging.info('Initial cleaning of AMD data.')
amd = pandas.read_csv('data/AMD.csv')
amd = amd.drop_duplicates()
assert diagnosi.codiceamd.isin(amd.codice).all()
assert esamilaboratorioparametri.codiceamd.isin(amd.codice).all()
assert esamilaboratorioparametricalcolati.codiceamd.isin(amd.codice).all()
assert esamistrumentali.codiceamd.isin(amd.codice).all()
assert prescrizionidiabetenonfarmaci.codiceamd.isin(amd.codice).all()
assert prescrizioninondiabete.codiceamd.isin(amd.codice).all()
# NA can be safelly dropped since there is all codes needed are already there.
amd = amd.dropna(axis=0, subset='codice')
amd = amd.set_index('codice', verify_integrity=True)
amd = amd.rename_axis('codiceamd')
assert not (diagnosi.codiceamd == 'AMD243').any()
assert not (esamilaboratorioparametri.codiceamd == 'AMD243').any()
assert not (esamilaboratorioparametricalcolati.codiceamd == 'AMD243').any()
assert not (esamistrumentali.codiceamd == 'AMD243').any()
assert not (prescrizionidiabetenonfarmaci.codiceamd == 'AMD243').any()
assert not (prescrizioninondiabete.codiceamd == 'AMD243').any()
# If you look at this two examples below it is clear that they are both Testo.
# diagnosi[diagnosi.codiceamd == 'AMD049']
# prescrizioninondiabete[prescrizioninondiabete.codiceamd == 'AMD121']
amd.loc['AMD049'] = 'Testo'
amd.loc['AMD121'] = 'Testo'
amd = amd.dropna()

# TODO: What do we do about this?
# There are ATC codes not in the main table (of ATC codes).
atc = pandas.read_csv('data/ATC.csv')
tmp = prescrizionidiabetefarmaci[~prescrizionidiabetefarmaci.codiceatc.isna()]
tmp[~tmp.codiceatc.isin(atc.atc_code)].codiceatc.unique()
del tmp

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
prescrizionidiabetefarmaci.codiceatc = prescrizionidiabetefarmaci.codiceatc.astype('category')
prescrizionidiabetefarmaci.idpasto = prescrizionidiabetefarmaci.idpasto.astype(meal_id.dtype)

prescrizionidiabetenonfarmaci = remove_first_column(prescrizionidiabetenonfarmaci)
prescrizionidiabetenonfarmaci.data = pandas.to_datetime(prescrizionidiabetenonfarmaci.data)
prescrizionidiabetenonfarmaci.codiceamd = prescrizionidiabetenonfarmaci.codiceamd.astype(codice_amd.dtype)

prescrizioninondiabete = remove_first_column(prescrizioninondiabete)
prescrizioninondiabete.data = pandas.to_datetime(prescrizioninondiabete.data)
prescrizioninondiabete.codiceamd = prescrizioninondiabete.codiceamd.astype(codice_amd.dtype)

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

logging.info('Updating esamilaboratorioparametri for point 4.')

esamilaboratorioparametri.valore.update(
	esamilaboratorioparametri[esamilaboratorioparametri.codiceamd == 'AMD004'].valore.clip(40, 200)
)
esamilaboratorioparametri.valore.update(
	esamilaboratorioparametri[esamilaboratorioparametri.codiceamd == 'AMD005'].valore.clip(40, 130)
)
esamilaboratorioparametri.valore.update(
	esamilaboratorioparametri[esamilaboratorioparametri.codiceamd == 'AMD007'].valore.clip(50, 500)
)
esamilaboratorioparametri.valore.update(
	esamilaboratorioparametri[esamilaboratorioparametri.codiceamd == 'AMD008'].valore.clip(5, 15)
)

# STITCH002 .clip(30, 300)
# STITCH003 .clip(60, 330)

# Point 5
# TODO: log this step.

# Saddly we have to do this step again to make sure that all the patients that
# were removed from diagnosi in the previous steps are removed also from the
# other tables.
anagraficapazientiattivi = diagnosi[['idcentro','idana']] \
	.join(anagraficapazientiattivi, ['idcentro','idana'], 'inner') \
	.drop_duplicates(['idcentro','idana']) \
	.set_index(['idcentro','idana'])
patients = anagraficapazientiattivi.index.to_frame().reset_index(drop=True)
diagnosi = diagnosi.merge(patients)
esamilaboratorioparametri = esamilaboratorioparametri.merge(patients)
esamilaboratorioparametricalcolati = esamilaboratorioparametricalcolati.merge(patients)
esamistrumentali = esamistrumentali.merge(patients)
prescrizionidiabetefarmaci = prescrizionidiabetefarmaci.merge(patients)
prescrizionidiabetenonfarmaci = prescrizionidiabetenonfarmaci.merge(patients)
prescrizioninondiabete = prescrizioninondiabete.merge(patients)

all_events = pandas.concat([
	diagnosi[['idcentro', 'idana', 'data']],
	esamilaboratorioparametri[['idcentro', 'idana', 'data']],
	esamilaboratorioparametricalcolati[['idcentro', 'idana', 'data']],
	esamistrumentali[['idcentro', 'idana', 'data']],
	prescrizionidiabetefarmaci[['idcentro', 'idana', 'data']],
	prescrizionidiabetenonfarmaci[['idcentro', 'idana', 'data']],
	prescrizioninondiabete[['idcentro', 'idana', 'data']],
])

# TODO: delete patients with less than two events.
(all_events.groupby(['idcentro', 'idana'], group_keys=True).count() < 2)

# NOTE: probabably the join here is useless.
last_event = all_events.join(anagraficapazientiattivi, ['idcentro', 'idana']) \
	.groupby(['idcentro', 'idana'], group_keys=True).data.max().dt.date

assert diagnosi.codiceamd.isin(macro_vascular_diseases).all(), \
	'diagnosi does not contain only macro vascular diseases'

last_cardiovascular_event = diagnosi.groupby(['idcentro', 'idana'], group_keys=True).data.max()

anagraficapazientiattivi = anagraficapazientiattivi.join(
	(last_cardiovascular_event >= last_event - pandas.tseries.offsets.DateOffset(month=6)).rename('y')
)

del patients, all_events, last_event, last_cardiovascular_event

# Point 6
# TODO: remove NA, NaN and NaT from the data and plotting.

assert anagraficapazientiattivi['sesso'].isna().sum() == 0

assert (anagraficapazientiattivi.tipodiabete == 5).all()
anagraficapazientiattivi = anagraficapazientiattivi.drop('tipodiabete', axis=1)
assert anagraficapazientiattivi.annodecesso.isna().all()
anagraficapazientiattivi = anagraficapazientiattivi.drop('annodecesso', axis=1)

percentages = anagraficapazientiattivi.isna().sum()/len(anagraficapazientiattivi)
mask = percentages > 0.4
anagraficapazientiattivi = anagraficapazientiattivi.drop(percentages[mask].index, axis=1)
del percentages, mask

#matplotlib.pyplot.hist(anagraficapazientiattivi['scolarita'])
#matplotlib.pyplot.show()

# TODO: remove remaining NA: for name in names: print(name, globals()[name].isna().any(), sep='\n')

# Data dumping #################################################################

logging.info('Dumping data.')
dataframes = [globals()[name] for name in names]
# TODO: fix warnings.
# NOTE: Maybe we can use piclke to preserve the data types of all dataframes.
with multiprocessing.pool.ThreadPool(len(names)) as pool:
	_ = pool.starmap(
		lambda df, path: df.to_csv(path),
		zip(dataframes, paths_for_cleaned)
	)
del pool
