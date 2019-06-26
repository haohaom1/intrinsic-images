import os
import sys

# scale the images by 50% and 25%
if len(sys.argv) < 3:
    print 'Usage: %s <image 1...>' % (sys.argv[0])

for infile in sys.argv[1:]:

    for scale in [50, 25]:
        words = infile.split('.')
        words[-2] += '-%d' % (scale)
        outfile = words[0]
        for word in words[1:]:
            outfile += '.' + word

        cmd = 'convert ' + infile + ' -scale %d%% ' % (scale) + outfile

        print cmd
        os.system( cmd )

    words = infile.split('.')
    words[-2] += '-thumb'
    words[-1] = 'png'
    outfile = words[0]
    for word in words[1:]:
        outfile += '.' + word

    cmd = 'convert ' + infile + ' -gamma 1.7  -resize %d ' % (150) + outfile

#    cmd = 'convert ' + infile + ' -scale 15%% ' + outfile

    print cmd
    os.system( cmd )

