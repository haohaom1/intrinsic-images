'''
  Shafat Rahman
  Macbeth linearity test (converted to python from Bruce A. Maxwell's macbeth.cpp)

  Given six locations which correspond to the grey line on a macbeth
  chart, it outputs the average intensities and the
  expected intensities from the Macbeth chart.

  The six locations are assumed to be in a text file as 
  x1 y1
  x2 y2 ...

  It also provides a best fit line and r^2 value.

  usage: python3 macbeth.py <image file> <label file>
'''


import sys
import cv2
import math

def linregress(y,x,N):
  sumx = 0.0
  sumy = 0.0
  sumx2 = 0.0
  sumy2 = 0.0
  sumxy = 0.0

  for i in range(N):
    sumx += x[i]
    sumy += y[i]
    sumx2 += x[i] * x[i]
    sumy2 += y[i] * y[i]
    sumxy += x[i] * y[i]

  slope = (N * sumxy - sumx*sumy) / (N * sumx2 - sumx*sumx)
  intercept = (sumy - slope * sumx) / N
  r2 = ( N * sumxy - sumx*sumy ) / math.sqrt( (N * sumx2 - sumx*sumx)* ( N*sumy2 - sumy*sumy ) )

  print("Measured = MacBeth * %.3f + %.3f  (%.3f)\n", slope, intercept, r2)

def main():
  #macbeth = [9.5, 8, 6.5, 5, 3.5, 2] # Munsell Notation Value
  macbeth = [90.0, 59.1, 36.2, 19.5, 9.0, 3.1] #Y channel
  points = []
  x=[]
  y=[]

  if len(sys.argv)< 3 :
    print("usage: {} <image file> <label file>\n".format(sys.argv[0]) )
    return

  f = open(sys.argv[2], "r")
  
  if f==None:
    print("Unable to open label file {}\n".format(sys.argv[2]) )
    return
  
  for i in range(6):
    x = f.readline().strip("\n").split(" ")
    print(x)
    for j in range(len(x)):
      x[j] = int(x[j])
    points.append(x)
    print("Location {}: ({})\n".format(i, x) )
  f.close()


  src = cv2.imread( sys.argv[1], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  if src.any() == None:
    print("Unable to read image file {}\n", sys.argv[1] )
    return

  
  # print("depth {} channels {}\n".format(src.depth(), src.channels()) )
  print("shape {}".format(src.shape) )

  print("\nMacBeth,Measured\n")
  for i in range(6):
    # calculate an average around each location
    sum = 0.0

    for j in range(points[i][1] - 3, points[i][1] + 4):
      for k in range (points[i][0] - 3, points[i][0] + 4):
        rgb = src[j, k]
        sum += rgb[0]
        # sum += rgb[1]
        # sum += rgb[2]
        # if i == 0 or i == 1:
        #   print(rgb[1])

    print(points[i][0])
    print(points[i][1])
    sum /= 1.0 * 49
    y.append(sum)
    print("%.2f,%.2f\n", macbeth[i], sum)

  # calculate the best fit
  linregress( y, macbeth, 6 )

if __name__ == "__main__":
  main()