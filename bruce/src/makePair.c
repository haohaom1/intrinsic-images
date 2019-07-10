/*
  Bruce A. Maxwell
  Fall 2018

  Generates a pair of images

  The images have the same material map
  The images have different illumination maps

  Process
  - generate a material map using one of the following options

    All materials have a minimum intensity of 4% and a maximum intensity of 100%

    1. solid color material

    2. two materials with an arbitrary line, but the center of the
    line in the image is a normal distribution centered on the image
    center

    3. two materials with a randomized selection by pixel

    4. two materials with randomized seeds, growing outwards
    
    5. anywhere from 2-25 materials with randomized seeds, growing outwards

    6. 2-25 materials with randomized selection by pixel

    7. 2-25 materials as a Mondrian with rectangles of randomized size

  - generate two illumination maps

    Direct illumination is a normal distribution around neutral in saturation, random hue

    Ambient illumination is a normal distribution around neutral in saturation, random hue

    Ambient illumination is constant over the image, normal distribution 

    Direct illumination varies over the image by adjusting gamma

    1. planar shading: pick a randomized plane by choosing the four corner values

    2. pick an arbitrary line, blur the boundary by a randomized amount

    3. randomized gamma by pixel

    4. fractal noise / Perlin noise

    5. blocks

    6. polynomial boundaries?  Bezier curve boundaries with randomized control points?

  - generate two images, same material map, different illumination maps

  - write out the two images as ppms?

 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ppmIO.h"
#include "makePair.h"

int main(int argc, char *argv[]) {
  FPixel *src;
  Pixel *ppm;
  int rows, cols, N;
  char prefix[32];
  char buffer[64];
  char type[16];
  int i, j;

  if( argc < 4 ) {
    printf("Usage; %s <N> <rows> <cols> <prefix = frame>\n  Generates N images with size rows x cols with the given\n  optional prefix (default frame)\n", argv[0]);
    exit(-1);
  }

  N = atoi(argv[1]);
  rows = atoi(argv[2]);
  cols = atoi(argv[3]);

  if( rows < 1 || rows > 10000 || cols < 1 || cols > 10000 || N < 1) {
    printf("Invalid arguments rows %d [1, 10000], cols %d [1, 10000], N %d [> 1]\n", rows, cols, N);
    exit(-2);
  }

  srand48(time(NULL));

  if( argc >= 5 )
    strcpy(prefix, argv[4]);
  else
    strcpy(prefix, "frame_");

  ppm = malloc(sizeof(FPixel)*rows*cols);

  // modified by Allen to generate all the kinds of images
  for (int t = 0; t < 5; t++) {
    for(i=0;i<N;i++) {
      int which = t; // rand() % 5;
      switch(which) {
      case 0:
        src = makeMaterialMap_solid(rows, cols);
        strcpy(type, "solid");
        break;
      case 1:
        src = makeMaterialMap_randomPixel( rows, cols );
        strcpy(type, "random");
        break;
      case 2:
        src = makeMaterialMap_twoColorLinearSplit( rows, cols );
        strcpy(type, "twoColor");
        break;
      case 3:
        src = makeMaterialMap_manySeed( rows, cols );
        strcpy(type, "manySeed");
        break;
      case 4:
        src = makeMaterialMap_squares( rows, cols );
        strcpy(type, "squares");
        break;
      }

      // convert to a ppm
      for(j=0;j<rows*cols;j++) {
        ppm[j].r = (int)(src[j].r*255);
        ppm[j].g = (int)(src[j].g*255);
        ppm[j].b = (int)(src[j].b*255);
      }

      sprintf(buffer, "%s%s%04d.ppm", prefix, type, i);
      writePPM( ppm, rows, cols, 255, buffer );
    }
  }

  return(0);
}
