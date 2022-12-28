import pandas
import matplotlib.pyplot
import multiprocessing.pool

def is_float(x: any) -> bool:
	try:
		_ = float(x)
	except (ValueError, TypeError):
		return False
	return True

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
paths = [f'sample/{name}.csv' for name in names]

with multiprocessing.pool.ThreadPool(len(names)) as pool:
	globals().update(dict(zip(names, pool.map(pandas.read_csv, paths))))
del pool

codice_amd = pandas.Categorical(f'AMD{i:03}' for i in range(1, 1000))
codice_stitch = pandas.Categorical(f'STITCH{i:03}' for i in range(1, 6))
# NOTE: should I create a sepatate Categorical for codiceatc too?

# TODO: idcentro can probably be and int16 and idana can probabbly be an int32
# to use less memory.

anagraficapazientiattivi = anagraficapazientiattivi.drop(anagraficapazientiattivi.columns[0], axis=1)
anagraficapazientiattivi.sesso = anagraficapazientiattivi.sesso.astype('category')
anagraficapazientiattivi.annodiagnosidiabete = pandas.to_datetime(anagraficapazientiattivi.annodiagnosidiabete.astype('Int16'), format='%Y')
# NOTE: tipodiabete is always 5.
anagraficapazientiattivi.tipodiabete = anagraficapazientiattivi.tipodiabete.astype('category')
anagraficapazientiattivi.annonascita = pandas.to_datetime(anagraficapazientiattivi.annonascita, format='%Y')
anagraficapazientiattivi.annoprimoaccesso = pandas.to_datetime(anagraficapazientiattivi.annoprimoaccesso.astype('Int16'), format='%Y')
anagraficapazientiattivi.annodecesso = pandas.to_datetime(anagraficapazientiattivi.annodecesso.astype('Int16'), format='%Y')
# TODO: understand scolarita ,statocivile, professione, origine
if (anagraficapazientiattivi.groupby(['idcentro', 'idana']).size() != 1).any():
	raise Exception("(idcentro, idana) are not the primary key for anagraficapazientiattivi")

diagnosi = diagnosi.drop(diagnosi.columns[0], axis=1)
diagnosi.data = pandas.to_datetime(diagnosi.data)
diagnosi.codiceamd = diagnosi.codiceamd.astype(codice_amd.dtype)
# TODO: understand valore
# wtf = diagnosi.valore[~diagnosi.valore.apply(is_float)].value_counts()

esamilaboratorioparametri = esamilaboratorioparametri.drop(esamilaboratorioparametri.columns[0], axis=1)
esamilaboratorioparametri.data = pandas.to_datetime(esamilaboratorioparametri.data)
esamilaboratorioparametri.codiceamd = esamilaboratorioparametri.codiceamd.astype(codice_amd.dtype)

esamilaboratorioparametricalcolati = esamilaboratorioparametricalcolati.drop(esamilaboratorioparametricalcolati.columns[0], axis=1)
esamilaboratorioparametricalcolati.data = pandas.to_datetime(esamilaboratorioparametricalcolati.data)
esamilaboratorioparametricalcolati.codiceamd = esamilaboratorioparametricalcolati.codiceamd.astype(codice_amd.dtype)
esamilaboratorioparametricalcolati.codicestitch = esamilaboratorioparametricalcolati.codicestitch.astype(codice_stitch.dtype)

esamistrumentali = esamistrumentali.drop(esamistrumentali.columns[0], axis=1)
esamistrumentali.data = pandas.to_datetime(esamistrumentali.data)
esamistrumentali.codiceamd = esamistrumentali.codiceamd.astype(codice_amd.dtype)
esamistrumentali.valore = esamistrumentali.valore.astype('category')

prescrizionidiabetefarmaci = prescrizionidiabetefarmaci.drop(prescrizionidiabetefarmaci.columns[0], axis=1)
prescrizionidiabetefarmaci.data = pandas.to_datetime(prescrizionidiabetefarmaci.data)
# NOTE: A10BD is a probably malformed.
prescrizionidiabetefarmaci.codiceatc = prescrizionidiabetefarmaci.codiceatc.astype('category')

prescrizionidiabetenonfarmaci = prescrizionidiabetenonfarmaci.drop(prescrizionidiabetenonfarmaci.columns[0], axis=1)
prescrizionidiabetenonfarmaci.data = pandas.to_datetime(prescrizionidiabetenonfarmaci.data)
prescrizionidiabetenonfarmaci.codiceamd = prescrizionidiabetenonfarmaci.codiceamd.astype(codice_amd.dtype)
# TODO: understand valore

prescrizioninondiabete = prescrizioninondiabete.drop(prescrizioninondiabete.columns[0], axis=1)
prescrizioninondiabete.data = pandas.to_datetime(prescrizioninondiabete.data)
prescrizioninondiabete.codiceamd = prescrizioninondiabete.codiceamd.astype(codice_amd.dtype)
# TODO: understand valore, is it categorical?

# Task 1 point 2
anagraficapazientiattivi = anagraficapazientiattivi.drop(
	anagraficapazientiattivi[anagraficapazientiattivi.annonascita >= anagraficapazientiattivi.annodecesso].index
)
assert not (anagraficapazientiattivi.annonascita >= anagraficapazientiattivi.annodecesso).any()

birth_death = anagraficapazientiattivi[['idcentro', 'idana', 'annonascita', 'annodecesso']]

def is_between(df: pandas.DataFrame):
	return (df.annonascita <= df.data) & (df.data <= df.annodecesso)

# diagnosi.merge(birth_death, on=['idcentro', 'idana'])
# esamilaboratorioparametri.data
# esamilaboratorioparametricalcolati.data
# esamistrumentali.data
# prescrizionidiabetefarmaci.data
# prescrizionidiabetenonfarmaci.data
# prescrizioninondiabete.data
