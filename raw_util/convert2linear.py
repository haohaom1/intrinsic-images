import os
import sys

# parametersx
texp = 0.0  # set higher if image is underexposed, lower if over-exposed
try:
    texp = float( sys.argv[1] )
    startindex = 2
except:
    startindex = 1

exposure = texp

files = ''
for infile in sys.argv[startindex:]:

    files += ' ' + infile

# Casey's command
#ufraw-batch --wb=camera --base-curve=linear --restore=lch --clip=digital --linearity=1.0 --saturation=1.0 --exposure=0.0 --wavelet-denoising-threshold=0.0 --hotpixel-sensitivity=0.0 --black-point=0 --interpolation=ahd --shrink=1 --out-type=tiff --out-depth=16 --create-id=no --noexif --nozip IMG_5323.CR2 


cmd = 'ufraw-batch --wb=camera --base-curve=linear --restore=lch --clip=digital --linearity=1.0 --saturation=1.0 --exposure=%.1f --wavelet-denoising-threshold=0.0 --hotpixel-sensitivity=0.0 --black-point=0 --interpolation=ahd --shrink=1 --out-type=tiff --out-depth=16  --create-id=no --noexif --nozip' % (exposure) + files

print(cmd)
os.system( cmd )

for infile in sys.argv[startindex:]:

    words = infile.split('.')
    words[-1] = 'tiff'

    newfile = words[0]
    for word in words[1:]:
        newfile += '.' + word

    words[-2] += '-acac'
    outfile = words[0]
    for word in words[1:]:
        outfile += '.' + word

    cmd = 'acac ' + newfile + ' ' + outfile
    print(cmd)
    os.system( cmd )

