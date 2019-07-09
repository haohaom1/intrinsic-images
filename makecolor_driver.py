import os

os.system("python3 makecolor.py stripe 8000 train 512 512")
os.system("python3 makecolor.py perlin 8000 train 512 512")
os.system("python3 makecolor.py fractal 8000 train 512 512")
os.system("python3 makecolor.py random 8000 train 512 512")
