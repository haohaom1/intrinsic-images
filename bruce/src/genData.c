/*
  Bruce A. Maxwell

  Generates seven csv data files

  1. sRGB_shadow.csv
  2. sRGB_random.csv
  3. linear_shadow.csv
  4. linear_random.csv
  5. log_shadow.csv
  6. log_random.csv
  7. lighting.csv: the ambient and direct lighting used to generate the corresponding shadow input

  Each CSV file has 12 numbers per row
  Which 12 is this?

  The command-line argument is how many examples to put in each file
*/

int main( int argc, char *argv[] ) {
  int i;
  

  for(i=0;i<N;i++) {

    // generate a light source

    // generate two colors

    // generate two more colors, make some of them be darker versions of the brighter colors

    // calculate the four output colors in linear space

    // convert to sRGB

    // convert to log RGB

    // write out the lines of the CSV files

  }
  
  return(0);
}
