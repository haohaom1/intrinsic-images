/*
  Bruce A. Maxwell

  Functions for making material maps

  Generates the linear version of the material map

  Returns an FPixel array of size rows * cols

*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ppmIO.h"
#include "makePair.h"

/*
  This is a function for picking two illumination colors: ambient and direct

  Illumination should not be highly saturated

  The ambient illumination intensity will be normally distributed
  around 20% of max intensity, with a hard lower bound of 3%.

  The direct illumination intensity will be normally distributed around 80% of max intensity

  Both illumination types will avoid saturation, using a normal
  distribution centered on 0 (reflected to be always positive) with a
  standard deviation of 0.15

  The hue will be uniformly distributed.
 */
void pickIlluminationColor( float rgb_ambient[], float rgb_direct[] ) {
  float hsv[3];
  const float ambient_std_saturation = 0.15;
  const float ambient_std_intensity = 0.10;
  const float ambient_mean_intensity = 0.20;
  const float ambient_lower_bound_intensity = 0.03;

  const float direct_std_saturation = 0.15;
  const float direct_std_intensity = 0.30;
  const float direct_mean_intensity = 0.80;
  const float direct_lower_bound_intensity = 0.25;
  const float min_ambient_direct_ratio = 0.50;

  do {
    // pick the ambient color
  
    // pick a random hue
    // uniform distribution 
    hsv[0] = drand48()*2*M_PI; // value between 0, 2PI

    // pick a random saturation using a normal distribution centered on 0 with std of 0.15
    // Gaussian distribution
    hsv[1] = gaussDist() * ambient_std_saturation;
    hsv[1] = hsv[1] < 0 ? -hsv[1] : hsv[1];
    hsv[1] = hsv[1] > 1.0 ? 1.0 : hsv[1];
  
    // intensity/value pick a random intensity between 4% and 100%
    // uniform distribution
    hsv[2] = gaussDist() * ambient_std_intensity + ambient_mean_intensity;
    hsv[2] = hsv[2] < 0 ? -hsv[2] : hsv[2];
    hsv[2] = hsv[2] < ambient_lower_bound_intensity ? ambient_lower_bound_intensity : hsv[2]; // hard lower bound
  
    // convert to RGB
    hsv2rgb( hsv, rgb_ambient );


    // pick the direct color
  
    // pick a random hue
    // uniform distribution 
    hsv[0] = drand48()*2*M_PI; // value between 0, 2PI

    // pick a random saturation using a normal distribution centered on 0 with std of 0.15
    // Gaussian distribution
    hsv[1] = gaussDist() * direct_std_saturation;
    hsv[1] = hsv[1] < 0 ? -hsv[1] : hsv[1];
    hsv[1] = hsv[1] > 1.0 ? 1.0 : hsv[1];
  
    // intensity/value pick a random intensity between 4% and 100%
    // uniform distribution
    hsv[2] = gaussDist() * direct_std_intensity + direct_mean_intensity;
    hsv[2] = hsv[2] < 0 ? -hsv[2] : hsv[2];
    hsv[2] = hsv[2] > 1.0 ? 1.0 : hsv[2];
    hsv[2] = hsv[2] < direct_lower_bound_intensity ? direct_lower_bound_intensity : hsv[2]; // hard lower bound
    
  
    // convert to RGB
    hsv2rgb( hsv, rgb_direct );
  } while( (rgb_direct[0] + rgb_direct[1] + rgb_direct[2])*min_ambient_direct_ratio < (rgb_ambient[0] + rgb_ambient[1] + rgb_ambient[2]) );

  return;
}

/*
  this is a function for picking material colors
  the input array rgb receives the material color
*/
void pickMaterialColor(float rgb[]) {
  float hsv[3];

  // pick a random hue
  // uniform distribution 
  hsv[0] = drand48()*2*M_PI; // value between 0, 2PI

  // pick a random saturation using a normal distribution centered on 0 with std of .3
  // Gaussian distribution
  hsv[1] = gaussDist() * 0.3;
  hsv[1] = hsv[1] < 0 ? -hsv[1] : hsv[1];
  hsv[1] = hsv[1] > 1.0 ? 1.0 : hsv[1];

  // intensity/value pick a random intensity between 4% and 100%
  // uniform distribution
  hsv[2] = drand48() * 0.95 + 0.05;

  // convert to RGB
  hsv2rgb( hsv, rgb );

  return;
}

/*
  this function executes a 3x3 Gaussian filter approximation
  it's good for making reflectance map boundaries smoother
  it can be run multiple times for additional blurring
*/
void blend(FPixel *src, int rows, int cols) {
  static FPixel *buffer = NULL;
  static int brows=0, bcols=0;
  float r, g, b;
  int i, j, base;

  if( buffer != NULL && (brows != rows || bcols != cols)) {
    free(buffer);
    buffer = NULL;
  }
  
  if( buffer == NULL) {
    buffer = malloc(sizeof(FPixel) * rows * cols );
    brows = rows;
    bcols = cols;
  }

  // upper left corner
  r = (src[0].r*4.0 + src[1].r*2.0 + src[cols].r*2.0 + src[cols+1].r) / 9.0;
  g = (src[0].g*4.0 + src[1].g*2.0 + src[cols].g*2.0 + src[cols+1].g) / 9.0;
  b = (src[0].b*4.0 + src[1].b*2.0 + src[cols].b*2.0 + src[cols+1].b) / 9.0;
  buffer[0].r = r; buffer[0].g = g; buffer[0].b = b;

  // upper right corner
  base = cols-1;
  r = (src[base].r*4.0 + src[base-1].r*2.0 + src[base+cols].r*2.0 + src[base+cols-1].r) / 9.0;
  g = (src[base].g*4.0 + src[base-1].g*2.0 + src[base+cols].g*2.0 + src[base+cols-1].g) / 9.0;
  b = (src[base].b*4.0 + src[base-1].b*2.0 + src[base+cols].b*2.0 + src[base+cols-1].b) / 9.0;
  buffer[base].r = r; buffer[base].g = g; buffer[base].b = b;

  // lower left corner
  base = (rows-1)*cols;
  r = (src[base].r*4.0 + src[base+1].r*2.0 + src[base-cols].r*2.0 + src[base-cols+1].r) / 9.0;
  g = (src[base].g*4.0 + src[base+1].g*2.0 + src[base-cols].g*2.0 + src[base-cols+1].g) / 9.0;
  b = (src[base].b*4.0 + src[base+1].b*2.0 + src[base-cols].b*2.0 + src[base-cols+1].b) / 9.0;
  buffer[base].r = r; buffer[base].g = g; buffer[base].b = b;

  // lower right corner
  base = rows*cols-1;
  r = (src[base].r*4.0 + src[base-1].r*2.0 + src[base-cols].r*2.0 + src[base-cols-1].r) / 9.0;
  g = (src[base].g*4.0 + src[base-1].g*2.0 + src[base-cols].g*2.0 + src[base-cols-1].g) / 9.0;
  b = (src[base].b*4.0 + src[base-1].b*2.0 + src[base-cols].b*2.0 + src[base-cols-1].b) / 9.0;
  buffer[base].r = r; buffer[base].g = g; buffer[base].b = b;

  // bondary conditions
  for(i=1;i<rows-1;i++) {
    // first column
    base = i*cols;
    r = (src[base].r*4.0 + src[base-cols].r*2.0 + src[base+cols].r*2.0 + src[base+1].r*2.0 + src[base-cols+1].r + src[base+cols+1].r)/12.0;
    g = (src[base].g*4.0 + src[base-cols].g*2.0 + src[base+cols].g*2.0 + src[base+1].g*2.0 + src[base-cols+1].g + src[base+cols+1].g)/12.0;
    b = (src[base].b*4.0 + src[base-cols].b*2.0 + src[base+cols].b*2.0 + src[base+1].b*2.0 + src[base-cols+1].b + src[base+cols+1].b)/12.0;
    buffer[base].r = r; buffer[base].g = g; buffer[base].b = b;

    // last column
    base = (i+1)*cols-1;
    r = (src[base].r*4.0 + src[base-cols].r*2.0 + src[base+cols].r*2.0 + src[base-1].r*2.0 + src[base-cols-1].r + src[base+cols-1].r)/12.0;
    g = (src[base].g*4.0 + src[base-cols].g*2.0 + src[base+cols].g*2.0 + src[base-1].g*2.0 + src[base-cols-1].g + src[base+cols-1].g)/12.0;
    b = (src[base].b*4.0 + src[base-cols].b*2.0 + src[base+cols].b*2.0 + src[base-1].b*2.0 + src[base-cols-1].b + src[base+cols-1].b)/12.0;
    buffer[base].r = r; buffer[base].g = g; buffer[base].b = b;

  }

  for(i=1;i<cols-1;i++) {

    // first row
    base = i;
    r = (src[base].r*4.0 + src[base-1].r*2.0 + src[base+1].r*2.0 + src[base+cols].r*2.0 + src[base+cols-1].r + src[base+cols+1].r)/12.0;
    g = (src[base].g*4.0 + src[base-1].g*2.0 + src[base+1].g*2.0 + src[base+cols].g*2.0 + src[base+cols-1].g + src[base+cols+1].g)/12.0;
    b = (src[base].b*4.0 + src[base-1].b*2.0 + src[base+1].b*2.0 + src[base+cols].b*2.0 + src[base+cols-1].b + src[base+cols+1].b)/12.0;
    buffer[base].r = r; buffer[base].g = g; buffer[base].b = b;

    // last row
    base = (rows-1)*cols + i;
    r = (src[base].r*4.0 + src[base-1].r*2.0 + src[base+1].r*2.0 + src[base-cols].r*2.0 + src[base-cols-1].r + src[base-cols+1].r)/12.0;
    g = (src[base].g*4.0 + src[base-1].g*2.0 + src[base+1].g*2.0 + src[base-cols].g*2.0 + src[base-cols-1].g + src[base-cols+1].g)/12.0;
    b = (src[base].b*4.0 + src[base-1].b*2.0 + src[base+1].b*2.0 + src[base-cols].b*2.0 + src[base-cols-1].b + src[base-cols+1].b)/12.0;
    buffer[base].r = r; buffer[base].g = g; buffer[base].b = b;
  }

  // center block
  for(i=1;i<rows-1;i++) {
    for(j=1;j<cols-1;j++) {
      base = i*cols + j;
      buffer[base].r = (src[base].r*4.0 + src[base-1].r*2.0 + src[base+1].r*2.0 + src[base-cols].r*2.0 + src[base+cols].r*2.0 + src[base-cols-1].r + src[base-cols+1].r + src[base+cols-1].r + src[base+cols+1].r) / 16.0;
      buffer[base].g = (src[base].g*4.0 + src[base-1].g*2.0 + src[base+1].g*2.0 + src[base-cols].g*2.0 + src[base+cols].g*2.0 + src[base-cols-1].g + src[base-cols+1].g + src[base+cols-1].g + src[base+cols+1].g) / 16.0;
      buffer[base].b = (src[base].b*4.0 + src[base-1].b*2.0 + src[base+1].b*2.0 + src[base-cols].b*2.0 + src[base+cols].b*2.0 + src[base-cols-1].b + src[base-cols+1].b + src[base+cols-1].b + src[base+cols+1].b) / 16.0;
    }
  }

  // copy buffer base to base
  for(i=0;i<rows*cols;i++) {
    src[i] = buffer[i];
  }

  return;
}

/*
  Generates an reflectance map that is a randomly chosen solid color

  Returns an FPixel array of the given number of rows and columns
*/
FPixel *makeMaterialMap_solid(int rows, int cols) {
  FPixel *src;
  float rgb[3];
  int i;

  // get a material color
  pickMaterialColor( rgb );

  // allocate the Pixel array
  src = malloc(sizeof(FPixel) * rows * cols);
  if( !src )
    return(NULL);

  // fill it in
  for(i=0;i<rows*cols;i++) {
    src[i].r = rgb[0];
    src[i].g = rgb[1];
    src[i].b = rgb[2];
  }

  // return it
  return(src);
}

/*
  Generates a reflectance map where every pixels is a random color

  Returns an FPixel array of the given number of rows and columns
*/
FPixel *makeMaterialMap_randomPixel(int rows, int cols) {
  FPixel *src;
  float rgb[3];
  int i;

  src = malloc(sizeof(FPixel) * rows * cols );
  if(!src)
    return(NULL);

  for(i=0;i<rows*cols;i++) {
    pickMaterialColor(rgb);
    src[i].r = rgb[0];
    src[i].g = rgb[1];
    src[i].b = rgb[2];
  }

  blend(src, rows, cols);

  return(src);
}


/*
  returns a two-color image with a straight boundary between the two colors

  The colors are blended along the boundary pixel according to the
  fraction of the pixel covering each color.
 */
FPixel *makeMaterialMap_twoColorLinearSplit(int rows, int cols) {
  FPixel *src;
  float rgb1[3], rgb2[3];
  int i, j, end;
  float cx, cy;
  float vx, vy, d;
  float xIntersect, dxPerScan, alpha;

  /* // uncomment this and comment out the next two to make illumination maps

     pickIlluminationColor( rgb1, rgb2 );
  */

  pickMaterialColor( rgb1 );
  pickMaterialColor( rgb2 );

  // pick an anchor pixel in the middle 50% of the image
  cx = drand48() * cols / 2 + cols/4;
  cy = drand48() * rows / 2 + rows/4;

  // pick a random direction that's not 0, 0
  vx = drand48()*2.0 - 1.0;
  vy = drand48()*2.0 - 1.0;
  while( vy*vy < 0.01 )
    vy = drand48()*2.0 - 1.0;

  // normalize the direction vector
  d = sqrt( vx*vx + vy*vy );
  vx /= d;
  vy /= d;
  dxPerScan = vx / vy;

  src = malloc(sizeof(FPixel) * rows * cols );

  // start rendering
  xIntersect = cx - cy * dxPerScan;

  for(i=0;i<rows;i++) {
    if( xIntersect > 1 ) { // if the row contains the first color
      end = floor(xIntersect);
      end = end < cols ? end : cols;
      alpha = xIntersect - floor(xIntersect); // percent of color 1

      // draw the first color
      for(j=0;j<end;j++) {
	int index = i*cols + j;
	src[index].r = rgb1[0];
	src[index].g = rgb1[1];
	src[index].b = rgb1[2];
      }

      // draw the second half
      if( end < cols ) { // if the row contains the second color
	// draw the blend pixel
	src[i*cols + end].r = alpha * rgb1[0] + (1.0 - alpha) * rgb2[0];
	src[i*cols + end].g = alpha * rgb1[1] + (1.0 - alpha) * rgb2[1];
	src[i*cols + end].b = alpha * rgb1[2] + (1.0 - alpha) * rgb2[2];

	// draw the second color
	for(j=end+1;j<cols;j++) {
	  int index = i*cols + j;
	  src[index].r = rgb2[0];
	  src[index].g = rgb2[1];
	  src[index].b = rgb2[2];
	}
      }
    } // end row contains first color
    else { // row doesn't contain first color
      // draw the second color
      for(j=0;j<cols;j++) {
	int index = i*cols + j;
	src[index].r = rgb2[0];
	src[index].g = rgb2[1];
	src[index].b = rgb2[2];
      }
    }
    xIntersect += dxPerScan;
  }

  /* // put this in to blend the bound more 
  int psize = 1 + rand() % 100;
  for(i=0;i<psize;i++) {
    blend( src, rows, cols );
  }
  */

  return(src);
}

/*
  Generates a reflectance map with many different regions

  The number of regions is between 3 and 21, with the number of
  regions being more likely to be in the middle of that range.

  The image uses blend to make the region boundaries somewhat smooth

  Returns an FPixel array of the given number of rows and columns

 */
FPixel *makeMaterialMap_manySeed(int rows, int cols) {
  FPixel *src;
  FPixel black = {0.0, 0.0, 0.0};
  float rgb[3];
  int N;
  int i, j;
  int flag;
  
  N = 3 + rand() % 10 + rand() % 10;

  src = malloc(sizeof(FPixel) * rows * cols );

  for(i=0;i<rows*cols;i++) {
    src[i] = black;
  }

  // initialize the seeds
  for(i=0;i<N;i++) {
    int rs = 1 + rand() % (rows-2);
    int cs = 1 + rand() % (cols-2);
    pickMaterialColor( rgb );
    src[rs*cols + cs].r = rgb[0];
    src[rs*cols + cs].g = rgb[1];
    src[rs*cols + cs].b = rgb[2];
  }

  flag = 1;
  while(flag) { // iterate until nothing changes
    flag = 0;

    // forward pass
    for(i=0;i<rows-1;i++) {
      for(j=0;j<cols-1;j++) {
	if( src[i*cols + j].r == 0.0 ) {
	  if( src[i*cols + j+1].r > 0.0 ) {
	    src[i*cols + j] = src[i*cols + j+1];
	    flag = 1;
	    continue;
	  }
	  if( src[(i+1)*cols + j].r > 0.0 ) {
	    src[i*cols + j] = src[(i+1)*cols + j];
	    flag = 1;
	    continue;
	  }
	  if( src[(i+1)*cols + j+1].r > 0.0 ) {
	    src[i*cols + j] = src[(i+1)*cols + j+1];
	    flag = 1;
	    continue;
	  }
	  if( j>0 && src[(i+1)*cols + j-1].r > 0.0 ) {
	    src[i*cols + j] = src[(i+1)*cols + j-1];
	    flag = 1;
	    continue;
	  }
	}
      }
    }

    // backward pass
    for(i=rows-1;i>0;i--) {
      for(j=cols-1;j>0;j--) {
	if( src[i*cols + j].r == 0.0 ) {
	  if( src[i*cols + j-1].r > 0.0 ) {
	    src[i*cols + j] = src[i*cols + j-1];
	    flag = 1;
	    continue;
	  }
	  if( src[(i-1)*cols + j].r > 0.0 ) {
	    src[i*cols + j] = src[(i-1)*cols + j];
	    flag = 1;
	    continue;
	  }
	  if( src[(i-1)*cols + j-1].r > 0.0 ) {
	    src[i*cols + j] = src[(i-1)*cols + j-1];
	    flag = 1;
	    continue;
	  }
	  if( j<cols-1 && src[(i-1)*cols + j+1].r > 0.0 ) {
	    src[i*cols + j] = src[(i-1)*cols + j+1];
	    flag = 1;
	    continue;
	  }
	}
      }
    }
  }

  // do one pass of blending
  blend(src, rows, cols);

  // return the image
  return(src);
}


/*
  Creates a material map that is covered by colored rectangles.

  The backgound is first chosen as a random color.  Then anywhere from
  5 to 50 rectangles are drawn into the image, with a higher
  probability in the middle of the range.  Each rectangle can be from
  2 pixels to rows/2 pixels on a side, uniformly distributed.

  Each rectangle has a 25% chance of being the same color as the last one.

  Returns an FPixel array of the given number of rows and columns

*/

FPixel *makeMaterialMap_squares(int rows, int cols) {
  FPixel *src;
  float rgb[3];
  int N, i;
  int x0, y0, x1, y1;

  // make the background image
  src = makeMaterialMap_solid(rows, cols);

  // get a number between 1 and 50, inclusive
  N = 5 + rand()%10 + rand()%10 + rand()%10 + rand()%10 + rand()%10;

  for(i=0;i<N;i++) {
    // pick a center point, which can be anywhere in the image
    int cx = rand() % cols;
    int cy = rand() % rows;
    int dx = rand() % (cols/2);
    int dy = rand() % (rows/2);
    int r, c;
    
    x0 = cx - dx/2;
    x0 = x0 < 0 ? 0 : x0;
    
    y0 = cy - dy/2;
    y0 = y0 < 0 ? 0 : y0;

    x1 = cx + dx/2;
    x1 = x1 > cols ? cols : x1;

    y1 = cy + dy/2;
    y1 = y1 > rows ? rows : y1;

    pickMaterialColor( rgb );
    
    for(r=y0;r<y1;r++) {
      for(c=x0;c<x1;c++) {
	src[r*cols + c].r = rgb[0];
	src[r*cols + c].g = rgb[1];
	src[r*cols + c].b = rgb[2];
      }
    }
  }

  return src;
}
