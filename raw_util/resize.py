import os
import sys

if len(sys.argv) < 3:
    print 'Usage: %s <width> <image...>' % (sys.argv[0])
    exit()

width = int( sys.argv[1] )

for infile in sys.argv[2:]:

    words = infile.split( '/' )
    outfile = words[-1]
    
    words = outfile.split('.')
    words[-2] = words[-2] + '-thumb'

    outfile = ''
    for s in words[:-1]:
        outfile += s + '.'
    outfile += 'png'

    cmd = 'convert ' + infile + ' -gamma 1.7  -resize %d ' % (width) + outfile
    print cmd
    os.system( cmd )
