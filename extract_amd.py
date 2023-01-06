import pdfreader
from pprint import pprint

fd = open('AMD.pdf', "rb")
viewer = pdfreader.SimplePDFViewer(fd)

# https://www.syncfusion.com/succinctly-free-ebooks/pdf/text-operators

text_objects = []
while True:
	try:
		# We skip the first page on puropose.
		viewer.next()
	except pdfreader.PageDoesNotExist:
		break
	viewer.render()
	markdown = viewer.canvas.text_content
	if 'Tj' in markdown:
		raise Exception('there is a Tj')

	page_text_objects = []
	end = -2
	while True:
		try:
			# The space is an hack to avoid writing a real parser, and avoid a false positive.
			start = markdown.index(' BT', end+2)
		except ValueError:
			break
		try:
			end = markdown.index(' ET', end+2)
		except ValueError:
			raise
		if end < start:
			breakpoint()
			raise Exception('bad text_content')

		text = markdown[start+2:end]
		page_text_objects.append(text)
	del start, end, text

	assert markdown.count(' BT') == markdown.count(' ET')
	assert len(page_text_objects) == markdown.count(' ET')
	text_objects.extend(page_text_objects)
	del markdown, page_text_objects

# Let's assume that  all operators are on their own line.
# We are intrested in Tm and TJ.
# `<a> <b> <c> <d> <e> <f> Tm` e and f are the position of the text.
# `[str, num, str num, ...] TJ` we discard all the numbers and join the strings
# which are between parenthesis.
# We are going to asume that every text object (BT ET) contains only one Tm and
# only one TJ.
# We are going to assume that the operator is always in the last two characters.
# Parenthesis in strings are excaped with a backslash.

positions = []
strings = []
for text_object in text_objects:
	position = None
	string = ''
	for line in text_object.split('\n'):
		if len(line) < 2: continue
		if line[-2:] == 'Tm':
			a, b, c, d, e, f, Tm = line.split(' ')
			position = float(e), float(f)
		if line[-2:] == 'TJ':
			inside_string = False
			escaped = False
			for char in line:
				if char == '(' and not escaped:
					inside_string = True
					continue
				if char == ')' and not escaped:
					inside_string = False
					continue
				if char == '\\':
					escaped == True
					continue
				if inside_string:
					string += char
					if escaped: escaped = False
					continue

	if not string:
		breakpoint()
		raise Exception(f'no string in text object {repr(text_object)}')
	if not position:
		breakpoint()
		raise Exception(f'no position in text object {repr(text_object)}')
	if string == ' ': continue
	strings.append(string)
	positions.append(position)

import pandas

df = pandas.DataFrame(positions, columns=['x', 'y'])
df['s'] = strings

col1 = df[(40  < df.x) & (df.x < 50)]
col2 = df[(55  < df.x) & (df.x < 100)]
col3 = df[(110 < df.x) & (df.x < 200)]
col4 = df[(210 < df.x) & (df.x < 300)]
col5 = df[(320 < df.x) & (df.x < 370)]
col6 = df[(380 < df.x) & (df.x < 400)]
col7 = df[(410 < df.x) & (df.x < 500)]
col8 = df[(510 < df.x) & (df.x < 720)]
col9 = df[df.x > 720]

# After this step there is still some cleaning to do.
# col2.s.to_clipboard()
# col6.s.to_clipboard()
