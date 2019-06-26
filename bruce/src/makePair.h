
// numerical utilities
double gaussDist(void);

// color utilities
void hsv2rgb(const float hsv[], float rgb[]);

// make material map functions
void pickIlluminationColor( float rgb_ambient[], float rgb_direct[] );
void pickMaterialColor(float rgb[]);
void blend(FPixel *src, int rows, int cols); // execute simple Gaussian approx


FPixel *makeMaterialMap_twoColorLinearSplit(int rows, int cols);
FPixel *makeMaterialMap_solid(int rows, int cols);
FPixel *makeMaterialMap_randomPixel(int rows, int cols);
FPixel *makeMaterialMap_manySeed(int rows, int cols);
FPixel *makeMaterialMap_squares(int rows, int cols);


