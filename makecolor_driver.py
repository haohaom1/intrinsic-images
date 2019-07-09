import os

os.system("python3 makecolor.py stripe 30000 train 512 512")
os.system("python3 makecolor.py perlin 30000 train 512 512")
os.system("python3 makecolor.py fractal 30000 train 512 512")
os.system("python3 makecolor.py random 30000 train 512 512")
