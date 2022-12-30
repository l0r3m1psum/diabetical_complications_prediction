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

# Initial data cleaning ########################################################

# TODO: idcentro can probably be and int16 and idana can probabbly be an int32
# to use less memory.

# The first colum of each table contains the row number which is useless for us.
def remove_first_column(df: pandas.DataFrame) -> pandas.DataFrame:
	res = df.drop(df.columns[0], axis=1)
	return res

anagraficapazientiattivi = remove_first_column(anagraficapazientiattivi)
anagraficapazientiattivi.annodiagnosidiabete = pandas.to_datetime(anagraficapazientiattivi.annodiagnosidiabete.astype('Int16'), format='%Y')
anagraficapazientiattivi.annonascita = pandas.to_datetime(anagraficapazientiattivi.annonascita, format='%Y')
anagraficapazientiattivi.annoprimoaccesso = pandas.to_datetime(anagraficapazientiattivi.annoprimoaccesso.astype('Int16'), format='%Y')
anagraficapazientiattivi.annodecesso = pandas.to_datetime(anagraficapazientiattivi.annodecesso.astype('Int16'), format='%Y')
anagraficapazientiattivi.sesso = anagraficapazientiattivi.sesso.astype('category')
anagraficapazientiattivi.tipodiabete = anagraficapazientiattivi.tipodiabete.astype('category')
# TODO: understand scolarita ,statocivile, professione, origine

#we can remove the feature "tipodiabete" since every instance of the dataset has the same value (5)
print("Does tipodiabete contain always the same value? ", (anagraficapazientiattivi.tipodiabete == 5).all())

# Checking that (idcentro, idana) is a primary key.
if (anagraficapazientiattivi.groupby(['idcentro', 'idana']).size() != 1).any():
	raise Exception("(idcentro, idana) are not the primary key for anagraficapazientiattivi")
anagraficapazientiattivi = anagraficapazientiattivi.set_index(['idcentro', 'idana'])

# Invalid feature cleaning
mask = anagraficapazientiattivi.annonascita >= anagraficapazientiattivi.annodecesso
n_invalid_patients = (mask).sum()
anagraficapazientiattivi = anagraficapazientiattivi.drop(
	anagraficapazientiattivi[mask].index
)
assert not (anagraficapazientiattivi.annonascita >= anagraficapazientiattivi.annodecesso).any()
# TODO: check that annodiagnosidiabete and annoprimoaccesso are between birth and death.
print("Remaining patients after eliminating inconsistent birth and deaths:", len(anagraficapazientiattivi), "patients eliminated:", n_invalid_patients)
#patient remaining TODO class distribution (i.e. with or without cardiovascular events)
del mask, n_invalid_patients

# Used to make sure that there are no patience outside of this group in the other tables.
patients = anagraficapazientiattivi.index.to_frame().reset_index(drop=True)

diagnosi = remove_first_column(diagnosi)
diagnosi = diagnosi.merge(patients)
diagnosi.data = pandas.to_datetime(diagnosi.data)
diagnosi.codiceamd = diagnosi.codiceamd.astype(codice_amd.dtype)
# TODO: understand valore
# wtf = diagnosi.valore[~diagnosi.valore.apply(is_float)].value_counts()

esamilaboratorioparametri = remove_first_column(esamilaboratorioparametri)
esamilaboratorioparametri = esamilaboratorioparametri.merge(patients)
esamilaboratorioparametri.data = pandas.to_datetime(esamilaboratorioparametri.data)
esamilaboratorioparametri.codiceamd = esamilaboratorioparametri.codiceamd.astype(codice_amd.dtype)

esamilaboratorioparametricalcolati = remove_first_column(esamilaboratorioparametricalcolati)
esamilaboratorioparametricalcolati = esamilaboratorioparametricalcolati.merge(patients)
esamilaboratorioparametricalcolati.data = pandas.to_datetime(esamilaboratorioparametricalcolati.data)
esamilaboratorioparametricalcolati.codiceamd = esamilaboratorioparametricalcolati.codiceamd.astype(codice_amd.dtype)
esamilaboratorioparametricalcolati.codicestitch = esamilaboratorioparametricalcolati.codicestitch.astype(codice_stitch.dtype)

esamistrumentali = remove_first_column(esamistrumentali)
esamistrumentali = esamistrumentali.merge(patients)
esamistrumentali.data = pandas.to_datetime(esamistrumentali.data)
esamistrumentali.codiceamd = esamistrumentali.codiceamd.astype(codice_amd.dtype)
esamistrumentali.valore = esamistrumentali.valore.astype('category')

prescrizionidiabetefarmaci = remove_first_column(prescrizionidiabetefarmaci)
prescrizionidiabetefarmaci = prescrizionidiabetefarmaci.merge(patients)
prescrizionidiabetefarmaci.data = pandas.to_datetime(prescrizionidiabetefarmaci.data)
# NOTE: A10BD is a probably malformed.
prescrizionidiabetefarmaci.codiceatc = prescrizionidiabetefarmaci.codiceatc.astype('category')

prescrizionidiabetenonfarmaci = remove_first_column(prescrizionidiabetenonfarmaci)
prescrizionidiabetenonfarmaci = prescrizionidiabetenonfarmaci.merge(patients)
prescrizionidiabetenonfarmaci.data = pandas.to_datetime(prescrizionidiabetenonfarmaci.data)
prescrizionidiabetenonfarmaci.codiceamd = prescrizionidiabetenonfarmaci.codiceamd.astype(codice_amd.dtype)
# TODO: understand valore

prescrizioninondiabete = remove_first_column(prescrizioninondiabete)
prescrizioninondiabete = prescrizioninondiabete.merge(patients)
prescrizioninondiabete.data = pandas.to_datetime(prescrizioninondiabete.data)
prescrizioninondiabete.codiceamd = prescrizioninondiabete.codiceamd.astype(codice_amd.dtype)
# TODO: understand valore, is it categorical?

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
