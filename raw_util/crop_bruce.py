import os
import sys

if len(sys.argv) < 5:
    print 'Usage: %s <dx> <dy> <offx> <offy> <images...>' % sys.argv[0]
    exit()

cropx = int( sys.argv[1] )
cropy = int( sys.argv[2] )
offx = int( sys.argv[3] )
offy = int( sys.argv[4] )

for infile in sys.argv[5:]:

    words = infile.split( '/' )
    outfile = words[-1]
    
    words = outfile.split('.')
    words[-2] = words[-2] + '-crop'

    outfile = ''
    for s in words[:-1]:
        outfile += s + '.'
    outfile += words[-1]

    cmd = 'convert ' + infile + ' -crop %dx%d+%d+%d ' % (cropx, cropy, offx, offy) + outfile
    print cmd
    os.system( cmd )
