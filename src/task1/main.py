import pandas
import matplotlib.pyplot
import multiprocessing.pool

def is_float(x: any) -> bool:
	try:
		_ = float(x)
	except (ValueError, TypeError):
		return False
	return True

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

with multiprocessing.pool.ThreadPool(len(names)) as pool:
	globals().update(dict(zip(names, pool.map(pandas.read_csv, paths))))
del pool

codice_amd = pandas.Categorical(f'AMD{i:03}' for i in range(1, 1000))
codice_stitch = pandas.Categorical(f'STITCH{i:03}' for i in range(1, 6))
# NOTE: should I create a sepatate Categorical for codiceatc too?

# The date in which the dataset was sampled.
# TODO: find the real one.
sampling_date = pandas.Timestamp(year=2022, month=1, day=1)

# Initial data cleaning ########################################################

# TODO: idcentro can probably be and int16 and idana can probabbly be an int32
# to use less memory.

# The first colum of each table contains the row number which is useless for us.
def remove_first_column(df: pandas.DataFrame) -> pandas.DataFrame:
	res = df.drop(df.columns[0], axis=1)
	return res

anagraficapazientiattivi = remove_first_column(anagraficapazientiattivi)
anagraficapazientiattivi = anagraficapazientiattivi.set_index(['idcentro', 'idana'], verify_integrity=True)
anagraficapazientiattivi.annodiagnosidiabete = pandas.to_datetime(anagraficapazientiattivi.annodiagnosidiabete.astype('Int16'), format='%Y')
anagraficapazientiattivi.annonascita = pandas.to_datetime(anagraficapazientiattivi.annonascita, format='%Y')
anagraficapazientiattivi.annoprimoaccesso = pandas.to_datetime(anagraficapazientiattivi.annoprimoaccesso.astype('Int16'), format='%Y')
anagraficapazientiattivi.annodecesso = pandas.to_datetime(anagraficapazientiattivi.annodecesso.astype('Int16'), format='%Y')
anagraficapazientiattivi.sesso = anagraficapazientiattivi.sesso.astype('category')
anagraficapazientiattivi.tipodiabete = anagraficapazientiattivi.tipodiabete.astype('category')
# TODO: understand scolarita ,statocivile, professione, origine

#we can remove the feature "tipodiabete" since every instance of the dataset has the same value (5)
print("Does tipodiabete contain always the same value? ", (anagraficapazientiattivi.tipodiabete == 5).all())

# Invalid feature cleaning
assert not anagraficapazientiattivi.annonascita.isnull().any()
print(f'Before initial cleaning: {len(anagraficapazientiattivi)=}')
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
print(f'After initial cleaning: {len(anagraficapazientiattivi)=}')
del is_not_between

#patient remaining TODO class distribution (i.e. with or without cardiovascular events)

# Used to make sure that there are no patience outside of this group in the other tables.
patients = anagraficapazientiattivi.index.to_frame().reset_index(drop=True)

print(f'Before initial cleaning: {len(diagnosi)=}')
diagnosi = remove_first_column(diagnosi)
diagnosi = diagnosi.merge(patients)
diagnosi.data = pandas.to_datetime(diagnosi.data)
diagnosi.codiceamd = diagnosi.codiceamd.astype(codice_amd.dtype)
# TODO: understand valore
# wtf = diagnosi.valore[~diagnosi.valore.apply(is_float)].value_counts()
print(f'After initial cleaning: {len(diagnosi)=}')

print(f'Before initial cleaning: {len(esamilaboratorioparametri)=}')
esamilaboratorioparametri = remove_first_column(esamilaboratorioparametri)
esamilaboratorioparametri = esamilaboratorioparametri.merge(patients)
esamilaboratorioparametri.data = pandas.to_datetime(esamilaboratorioparametri.data)
esamilaboratorioparametri.codiceamd = esamilaboratorioparametri.codiceamd.astype(codice_amd.dtype)
print(f'After initial cleaning: {len(esamilaboratorioparametri)=}')

print(f'Before initial cleaning: {len(esamilaboratorioparametricalcolati)=}')
esamilaboratorioparametricalcolati = remove_first_column(esamilaboratorioparametricalcolati)
esamilaboratorioparametricalcolati = esamilaboratorioparametricalcolati.merge(patients)
esamilaboratorioparametricalcolati.data = pandas.to_datetime(esamilaboratorioparametricalcolati.data)
esamilaboratorioparametricalcolati.codiceamd = esamilaboratorioparametricalcolati.codiceamd.astype(codice_amd.dtype)
esamilaboratorioparametricalcolati.codicestitch = esamilaboratorioparametricalcolati.codicestitch.astype(codice_stitch.dtype)
print(f'After initial cleaning: {len(esamilaboratorioparametricalcolati)=}')

print(f'Before initial cleaning: {len(esamistrumentali)=}')
esamistrumentali = remove_first_column(esamistrumentali)
esamistrumentali = esamistrumentali.merge(patients)
esamistrumentali.data = pandas.to_datetime(esamistrumentali.data)
esamistrumentali.codiceamd = esamistrumentali.codiceamd.astype(codice_amd.dtype)
esamistrumentali.valore = esamistrumentali.valore.astype('category')
print(f'After initial cleaning: {len(esamistrumentali)=}')

print(f'Before initial cleaning: {len(prescrizionidiabetefarmaci)=}')
prescrizionidiabetefarmaci = remove_first_column(prescrizionidiabetefarmaci)
prescrizionidiabetefarmaci = prescrizionidiabetefarmaci.merge(patients)
prescrizionidiabetefarmaci.data = pandas.to_datetime(prescrizionidiabetefarmaci.data)
# NOTE: A10BD is a probably malformed.
prescrizionidiabetefarmaci.codiceatc = prescrizionidiabetefarmaci.codiceatc.astype('category')
print(f'After initial cleaning: {len(prescrizionidiabetefarmaci)=}')

print(f'Before initial cleaning: {len(prescrizionidiabetenonfarmaci)=}')
prescrizionidiabetenonfarmaci = remove_first_column(prescrizionidiabetenonfarmaci)
prescrizionidiabetenonfarmaci = prescrizionidiabetenonfarmaci.merge(patients)
prescrizionidiabetenonfarmaci.data = pandas.to_datetime(prescrizionidiabetenonfarmaci.data)
prescrizionidiabetenonfarmaci.codiceamd = prescrizionidiabetenonfarmaci.codiceamd.astype(codice_amd.dtype)
# TODO: understand valore
print(f'After initial cleaning: {len(prescrizionidiabetenonfarmaci)=}')

print(f'Before initial cleaning: {len(prescrizioninondiabete)=}')
prescrizioninondiabete = remove_first_column(prescrizioninondiabete)
prescrizioninondiabete = prescrizioninondiabete.merge(patients)
prescrizioninondiabete.data = pandas.to_datetime(prescrizioninondiabete.data)
prescrizioninondiabete.codiceamd = prescrizioninondiabete.codiceamd.astype(codice_amd.dtype)
# TODO: understand valore, is it categorical?
print(f'After initial cleaning: {len(prescrizioninondiabete)=}')

del remove_first_column, patients

# Tasks 1 ######################################################################

# Point 2

birth_death = anagraficapazientiattivi[['annonascita', 'annodecesso']]

def clean_is_between(df: pandas.DataFrame) -> pandas.DataFrame:
	# We add birth and death information of each patient to the table.
	mdf = df.join(birth_death, ['idcentro','idana'], 'inner')
	assert len(mdf) == len(df)
	res = df[(mdf.annonascita <= mdf.data) & (mdf.data <= mdf.annodecesso)]
	return res

diagnosi = clean_is_between(diagnosi)
esamilaboratorioparametri = clean_is_between(esamilaboratorioparametri)
esamilaboratorioparametricalcolati = clean_is_between(esamilaboratorioparametricalcolati)
esamistrumentali = clean_is_between(esamistrumentali)
prescrizionidiabetefarmaci = clean_is_between(prescrizionidiabetefarmaci)
prescrizionidiabetenonfarmaci = clean_is_between(prescrizionidiabetenonfarmaci)
prescrizioninondiabete = clean_is_between(prescrizioninondiabete)

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

diagnosi = clean_same_month(diagnosi)
esamilaboratorioparametri = clean_same_month(esamilaboratorioparametri)
esamilaboratorioparametricalcolati = clean_same_month(esamilaboratorioparametricalcolati)
esamistrumentali = clean_same_month(esamistrumentali)
prescrizionidiabetefarmaci = clean_same_month(prescrizionidiabetefarmaci)
prescrizionidiabetenonfarmaci = clean_same_month(prescrizionidiabetenonfarmaci)
prescrizioninondiabete = clean_same_month(prescrizioninondiabete)

del clean_same_month
