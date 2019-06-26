"""
	* Maan Qraitem
	* Colby College 
	* Linear Images research 
	* Purpose: Checks if any of the linear images can't be opened and processed.
"""

from subprocess import Popen, PIPE
import os

folderToCheck = '../Dataset_Converted/linear'
fileExtension = '.tiff'

def checkImage(fn):
	proc = Popen(['identify', '-verbose', fn], stdout=PIPE, stderr=PIPE)
	out, err = proc.communicate()
	exitcode = proc.returncode
	return exitcode, out, err


for file in os.listdir(folderToCheck):
	if file.endswith(fileExtension):
		filePath = os.path.join(folderToCheck, file)
		code, output, error = checkImage(filePath)
		if str(code) !="0" or str(error, "utf-8") != "":
			print("ERROR " + filePath)
		else:
			print("OK " + filePath)

print("-------------- DONE --------------")
