import sys
sys.path.append('src')

from common import *

# Data loading and definition ##################################################

logging.info('Loading data.')
with multiprocessing.pool.ThreadPool(len(names)) as pool:
	globals().update(dict(zip(names, pool.map(pandas.read_csv, paths))))
del pool

# Initial data cleaning ########################################################

# logging.info('Initial cleaning of AMD data.')
# amd = pandas.read_csv('data/AMD.csv')
# amd = amd.drop_duplicates()
# assert diagnosi.codiceamd.isin(amd.codice).all()
# assert esamilaboratorioparametri.codiceamd.isin(amd.codice).all()
# assert esamilaboratorioparametricalcolati.codiceamd.isin(amd.codice).all()
# assert esamistrumentali.codiceamd.isin(amd.codice).all()
# assert prescrizionidiabetenonfarmaci.codiceamd.isin(amd.codice).all()
# assert prescrizioninondiabete.codiceamd.isin(amd.codice).all()
# # NA can be safelly dropped since there is all codes needed are already there.
# amd = amd.dropna(axis=0, subset='codice')
# amd = amd.set_index('codice', verify_integrity=True)
# amd = amd.rename_axis('codiceamd')
# assert not (diagnosi.codiceamd == 'AMD243').any()
# assert not (esamilaboratorioparametri.codiceamd == 'AMD243').any()
# assert not (esamilaboratorioparametricalcolati.codiceamd == 'AMD243').any()
# assert not (esamistrumentali.codiceamd == 'AMD243').any()
# assert not (prescrizionidiabetenonfarmaci.codiceamd == 'AMD243').any()
# assert not (prescrizioninondiabete.codiceamd == 'AMD243').any()
# # If you look at this two examples below it is clear that they are both Testo.
# # diagnosi[diagnosi.codiceamd == 'AMD049']
# # prescrizioninondiabete[prescrizioninondiabete.codiceamd == 'AMD121']
# amd.loc['AMD049'] = 'Testo'
# amd.loc['AMD121'] = 'Testo'
# amd = amd.dropna()
# amd.to_pickle('data/AMD_clean.pickle.zip', 'infer', -1)

amd = pandas.read_csv('data/amd_codes_for_bert.csv')
if (not diagnosi.codiceamd.isin(amd.codice).all()
	or not esamilaboratorioparametri.codiceamd.isin(amd.codice).all()
	or not esamilaboratorioparametricalcolati.codiceamd.isin(amd.codice).all()
	or not esamistrumentali.codiceamd.isin(amd.codice).all()
	or not prescrizionidiabetenonfarmaci.codiceamd.isin(amd.codice).all()
	or not prescrizioninondiabete.codiceamd.isin(amd.codice).all()):
	logging.warning('Not all AMD codes are described.')
del amd

atc = pandas.read_csv('data/atc_info_nodup.csv')
if not prescrizionidiabetefarmaci.codiceatc.isin(atc.codiceatc).all():
	logging.warning('Not all ATC codes are described.')
del atc

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
# Dropping inconsistent birth and deaths (keep in mind that comparisons on NaTs always return False).
anagraficapazientiattivi = anagraficapazientiattivi.drop(
	anagraficapazientiattivi[anagraficapazientiattivi.annonascita >= anagraficapazientiattivi.annodecesso].index
)
# Dropping future deaths.
anagraficapazientiattivi = anagraficapazientiattivi.drop(
	(anagraficapazientiattivi.annodecesso[~anagraficapazientiattivi.annodecesso.isnull()] < sampling_date).index
)
def is_not_between(s: pandas.Series) -> pandas.MultiIndex:
	res = (anagraficapazientiattivi.annonascita < s) \
		& (s < anagraficapazientiattivi.annodecesso.fillna(sampling_date))
	res = (anagraficapazientiattivi[~res]).index
	return res
# NOTE: since we don't know the true meaning of the data it is hard to decide
# what should we do with NaTs in annodiagnosidiabete and annoprimoaccesso. Right
# now they are all removed, it would be nice to find a way to keep some of them.
anagraficapazientiattivi = anagraficapazientiattivi.drop(is_not_between(anagraficapazientiattivi.annodiagnosidiabete))
anagraficapazientiattivi = anagraficapazientiattivi.drop(is_not_between(anagraficapazientiattivi.annoprimoaccesso))
logging.info(f'After  initial cleaning: {len(anagraficapazientiattivi)=}')
del is_not_between

diagnosi = remove_first_column(diagnosi)
diagnosi.data = pandas.to_datetime(diagnosi.data)

esamilaboratorioparametri = remove_first_column(esamilaboratorioparametri)
esamilaboratorioparametri.data = pandas.to_datetime(esamilaboratorioparametri.data)

esamilaboratorioparametricalcolati = remove_first_column(esamilaboratorioparametricalcolati)
esamilaboratorioparametricalcolati.data = pandas.to_datetime(esamilaboratorioparametricalcolati.data)

esamistrumentali = remove_first_column(esamistrumentali)
esamistrumentali.data = pandas.to_datetime(esamistrumentali.data)
esamistrumentali.valore = esamistrumentali.valore.astype('category')

prescrizionidiabetefarmaci = remove_first_column(prescrizionidiabetefarmaci)
prescrizionidiabetefarmaci.data = pandas.to_datetime(prescrizionidiabetefarmaci.data)

prescrizionidiabetenonfarmaci = remove_first_column(prescrizionidiabetenonfarmaci)
prescrizionidiabetenonfarmaci.data = pandas.to_datetime(prescrizionidiabetenonfarmaci.data)

prescrizioninondiabete = remove_first_column(prescrizioninondiabete)
prescrizioninondiabete.data = pandas.to_datetime(prescrizioninondiabete.data)

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

# In the original table there are also the STITCH code below but they are not in
# esamilaboratorioparametri.
# STITCH002 .clip(30, 300)
# STITCH003 .clip(60, 330)

# Point 5

# Saddly we have to do this step again to make sure that all the patients that
# were removed from diagnosi in the previous steps are removed also from the
# other tables.
logging.info(f'Before point 5: {len(anagraficapazientiattivi)=}')
anagraficapazientiattivi = diagnosi[['idcentro','idana']] \
	.join(anagraficapazientiattivi, ['idcentro','idana'], 'inner') \
	.drop_duplicates(['idcentro','idana']) \
	.set_index(['idcentro','idana'])

patients = anagraficapazientiattivi.index.to_frame().reset_index(drop=True)

logging.info(f'Before point 5: {len(diagnosi)=}')
diagnosi = diagnosi.merge(patients)
logging.info(f'After  point 5: {len(diagnosi)=}')

logging.info(f'Before point 5: {len(esamilaboratorioparametri)=}')
esamilaboratorioparametri = esamilaboratorioparametri.merge(patients)
logging.info(f'After  point 5: {len(esamilaboratorioparametri)=}')

logging.info(f'Before point 5: {len(esamilaboratorioparametricalcolati)=}')
esamilaboratorioparametricalcolati = esamilaboratorioparametricalcolati.merge(patients)
logging.info(f'After  point 5: {len(esamilaboratorioparametricalcolati)=}')

logging.info(f'Before point 5: {len(esamistrumentali)=}')
esamistrumentali = esamistrumentali.merge(patients)
logging.info(f'After  point 5: {len(esamistrumentali)=}')

logging.info(f'Before point 5: {len(prescrizionidiabetefarmaci)=}')
prescrizionidiabetefarmaci = prescrizionidiabetefarmaci.merge(patients)
logging.info(f'After  point 5: {len(prescrizionidiabetefarmaci)=}')

logging.info(f'Before point 5: {len(prescrizionidiabetenonfarmaci)=}')
prescrizionidiabetenonfarmaci = prescrizionidiabetenonfarmaci.merge(patients)
logging.info(f'After  point 5: {len(prescrizionidiabetenonfarmaci)=}')

logging.info(f'Before point 5: {len(prescrizioninondiabete)=}')
prescrizioninondiabete = prescrizioninondiabete.merge(patients)
logging.info(f'After  point 5: {len(prescrizioninondiabete)=}')

all_events = pandas.concat([
	diagnosi[['idcentro', 'idana', 'data']],
	esamilaboratorioparametri[['idcentro', 'idana', 'data']],
	esamilaboratorioparametricalcolati[['idcentro', 'idana', 'data']],
	esamistrumentali[['idcentro', 'idana', 'data']],
	prescrizionidiabetefarmaci[['idcentro', 'idana', 'data']],
	prescrizionidiabetenonfarmaci[['idcentro', 'idana', 'data']],
	prescrizioninondiabete[['idcentro', 'idana', 'data']],
])

assert not (all_events.groupby(['idcentro', 'idana'], group_keys=True).count() < 2).data.any()
logging.info('There are no patients with less than 2 events to remove.')

assert diagnosi.codiceamd.isin(macro_vascular_diseases).all(), \
	'diagnosi does not contain only macro vascular diseases'

last_event = all_events.groupby(['idcentro', 'idana'], group_keys=True).data.max().dt.date
last_cardiovascular_event = diagnosi.groupby(['idcentro', 'idana'], group_keys=True).data.max()
anagraficapazientiattivi = anagraficapazientiattivi.join(
	(last_cardiovascular_event >= last_event - pandas.DateOffset(month=6)).rename('y')
)
logging.info(f'After  point 5: {len(anagraficapazientiattivi)=}')

def compute_class_distribution():
	pazienti = globals()['anagraficapazientiattivi'][:]
	try:
		pazienti = pazienti.drop(columns='y')
	except Exception as e:
		pass
	diagnosi_all = globals()['diagnosi'][:]

	last_event = globals()['all_events'].join(pazienti, ['idcentro', 'idana']) \
	.groupby(['idcentro', 'idana'], group_keys=True).data.max().dt.date

	last_cardiovascular_event = diagnosi_all.groupby(['idcentro', 'idana'], group_keys=True).data.max()
	positives = pazienti.join(
		(last_cardiovascular_event >= last_event - pandas.DateOffset(month=6)).rename('y')
	)
	positive_labels = positives['y'].sum() 
	total_labels = len(pazienti)

	return positive_labels, total_labels

del patients, all_events, last_event, last_cardiovascular_event

# Point 6

assert not anagraficapazientiattivi.sesso.isna().any()

assert (anagraficapazientiattivi.tipodiabete == 5).all()
anagraficapazientiattivi = anagraficapazientiattivi.drop('tipodiabete', axis=1)
assert anagraficapazientiattivi.annodecesso.isna().all()
anagraficapazientiattivi = anagraficapazientiattivi.drop('annodecesso', axis=1)

percentages = anagraficapazientiattivi.isna().sum()/len(anagraficapazientiattivi)
mask = percentages > 0.4
anagraficapazientiattivi = anagraficapazientiattivi.drop(percentages[mask].index, axis=1)
del percentages, mask

# We can't do numerical imputation with codes, but we can't drop data either
# since they're too many therefore we will fill the empty rows with the most
# frequent value.
# Here we can drop the NA since they are just 27
prescrizionidiabetefarmaci = prescrizionidiabetefarmaci.dropna()
# NAs in shall not be esamilaboratorioparametricalcolati since they have meaning.
assert (esamilaboratorioparametricalcolati[esamilaboratorioparametricalcolati.codiceamd == 'AMD927'].codicestitch == 'STITCH001').all()
assert (esamilaboratorioparametricalcolati[esamilaboratorioparametricalcolati.codiceamd == 'AMD013'].codicestitch == 'STITCH002').all()
assert (esamilaboratorioparametricalcolati[esamilaboratorioparametricalcolati.codiceamd == 'AMD304'].codicestitch == 'STITCH005').all()
assert esamilaboratorioparametricalcolati[esamilaboratorioparametricalcolati.codiceamd.isna()].codicestitch.isin(['STITCH003', 'STITCH004']).all()
# Here we drop the NA where codiceamd == 'AMD096' because they are just 11. The
# remaining NA alle have codiceamd == 'AMD152'.
mask = prescrizionidiabetenonfarmaci.valore.isna() \
	& (prescrizionidiabetenonfarmaci.codiceamd == 'AMD096')
prescrizionidiabetenonfarmaci = prescrizionidiabetenonfarmaci.drop(mask[mask].index)
mode = prescrizionidiabetenonfarmaci[prescrizionidiabetenonfarmaci.codiceamd == 'AMD152'].valore.mode().item()
assert (prescrizionidiabetenonfarmaci[prescrizionidiabetenonfarmaci.valore.isna()].codiceamd == 'AMD152').all()
prescrizionidiabetenonfarmaci.valore.fillna(mode, inplace=True)
# Here we drop NA where codiceamd == 'AMD126' since they are just 7 and fill NA
# where codiceamd == 'AMD125' with the proper mode.
mask = esamistrumentali.valore.isna() \
	& (esamistrumentali.codiceamd == 'AMD126')
esamistrumentali = esamistrumentali.drop(mask[mask].index)
mode = esamistrumentali[esamistrumentali.codiceamd == 'AMD125'].valore.mode().item()
assert (esamistrumentali[esamistrumentali.valore.isna()].codiceamd == 'AMD125').all()
esamistrumentali.valore.fillna(mode, inplace=True)
# FIXME: interpolation depends on order and most importantly on the codiceamd.
# Look at esamilaboratorioparametri[esamilaboratorioparametri.valore.isna()].codiceamd.value_counts()
# and at esamilaboratorioparametri.codiceamd.value_counts() to make a better
# decision about how to fill this data.
esamilaboratorioparametri['valore'].interpolate(method='linear', inplace=True) # NOTE: linear interpolation, assumes values equally spaced...
del mask, mode

# This is important for task 2 but in general, since we drop rows from random
# places in the dataframe, we have to keep the index sequential.
diagnosi = diagnosi.reset_index(drop=True)
esamilaboratorioparametri = esamilaboratorioparametri.reset_index(drop=True)
esamilaboratorioparametricalcolati = esamilaboratorioparametricalcolati.reset_index(drop=True)
esamistrumentali = esamistrumentali.reset_index(drop=True)
prescrizionidiabetefarmaci = prescrizionidiabetefarmaci.reset_index(drop=True)
prescrizionidiabetenonfarmaci = prescrizionidiabetenonfarmaci.reset_index(drop=True)
prescrizioninondiabete = prescrizioninondiabete.reset_index(drop=True)

# All dataframes don't have any NA left (except for esamilaboratorioparametricalcolati).
assert not anagraficapazientiattivi.isna().any().any()
assert not diagnosi.isna().any().any()
assert not esamilaboratorioparametri.isna().any().any()
# assert not esamilaboratorioparametricalcolati.isna().any().any()
assert not esamistrumentali.isna().any().any()
assert not prescrizionidiabetefarmaci.isna().any().any()
assert not prescrizionidiabetenonfarmaci.isna().any().any()
assert not prescrizioninondiabete.isna().any().any()

# Data dumping #################################################################

logging.info('Dumping data.')
dataframes = [globals()[name] for name in names]
with multiprocessing.pool.ThreadPool(len(names)) as pool:
	_ = pool.starmap(
		# -1 means use the latest protocol for serialization.
		lambda df, path: df.to_pickle(path, 'infer', -1),
		zip(dataframes, paths_for_cleaned)
	)
del dataframes, pool
