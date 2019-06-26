import os
import sys

if len(sys.argv) < 4:
    print 'Usage: %s <multiplier> <gamma> <image...>' % (sys.argv[0])
    exit()

multiplier = float( sys.argv[1] )
gamma = float( sys.argv[2] )

for infile in sys.argv[3:]:

    words = infile.split( '/' )
    outfile = words[-1]
    
    words = outfile.split('.')
    words[-2] = words[-2] + '-mg'

    outfile = ''
    for s in words[:-1]:
        outfile += s + '.'
    outfile += words[-1]

    cmd = 'convert ' + infile + ' -evaluate Multiply %.3f -gamma %.3f ' % (multiplier, gamma) + outfile
    print cmd
    os.system( cmd )
