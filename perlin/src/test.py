"""Writes a 256x256 grayscale simplex noise texture file in pgm format
(see http://netpbm.sourceforge.net/doc/pgm.html)
"""
# $Id: 2dtexture.py 21 2008-05-21 07:52:29Z casey.duncan $

import sys
from noise import pnoise2, snoise2

width = 2400
height = 2400

if len(sys.argv) not in (2, 3) or '--help' in sys.argv or '-h' in sys.argv:
	print('2dtexture.py FILE [OCTAVES]')
	print()
	print(__doc__)
	raise SystemExit

f = open(sys.argv[1], 'wt')
if len(sys.argv) > 2:
	octaves = int(sys.argv[2])
else:
	octaves = 1
freq = width / 16.0 * octaves
f.write('P2\n')
f.write('2400 2400\n')
f.write('2399\n')
for y in range(height):
	for x in range(width):
		f.write("%s\n" % int(snoise2(x / freq, y / freq, octaves) * (width / 2.0 - 1 )+ height / 2.0))
f.close()