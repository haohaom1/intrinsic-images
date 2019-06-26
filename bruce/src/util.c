/*
  Bruce A. Maxwell
  Fall 2018

  Utility color routines
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
  From TVS 5

  Hue in radians [0, 2PI]
  Saturation, Value in [0, 1]

  R, G, B returned as [0, 1]
 */
#include <stdio.h>
#include <math.h>

void hsv2rgb(const float hsv[], float rgb[]);
void hsv2rgb(const float hsv[], float rgb[]) {
  float h = hsv[0];
  float s = hsv[1];
  float v = hsv[2];

  // put the hue in the proper range [0, 2PI]
  while (h<0.0) h += 2.0*M_PI;
  while (h>2.0*M_PI) h-= 2.0*M_PI;

  // compute the various values
  int h1 = (int)(floor(h/(M_PI/3.0f))) % 6;
  float f = h/(M_PI/3.0f) - floor(h/(M_PI/3.0f));
  float p = v*(1-s);
  float q = v*(1-f*s);
  float t = v*(1-(1-f)*s);
  float r=0.0f, g=0.0f, b=0.0f;

  // switch on the hue value to find the proper rotation
  switch (h1) {
  case 0:
    r = v;
    g = t;
    b = p;
    break;
  case 1:
    r = q;
    g = v;
    b = p;
    break;
  case 2:
    r = p;
    g = v;
    b = t;
    break;
  case 3:
    r = p;
    g = q;
    b = v;
    break;
  case 4:
    r = t;
    g = p;
    b = v;
    break;
  default:
    r = v;
    g = p;
    b = q;
    break;
  }
  rgb[0] = r;
  rgb[1] = g;
  rgb[2] = b;

  return;
}
