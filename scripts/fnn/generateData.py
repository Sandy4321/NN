# generateData.py
# Daniel E. Worrall -- 16  Oct 2014

"""Generate parity function training examples"""
#
# Imports
#

import numpy as np                    # to do more sophisticated maths
import optparse                       # multiple input options
import pylab
import os                             # for file writing
from parity import Parity             # import parity function

#
# Input options 
#

p = optparse.OptionParser()
p.add_option("-f", "--file", 
                action="store", 
                type="string", 
                dest="outFile", 
                metavar="FILE", 
                help="data output address (default = '../data/NN.data')")
p.add_option("-d", "--dimensions", 
                action="store", 
                type="int",
                dest="N", 
                metavar="D",
                help="number of dimensions (default = 8)")
p.add_option("-m", "--mode", 
                action="store",
                type="string",
                dest="mode", 
                metavar="M",
                help="function type (default = XOR)")

p.set_defaults(outFile="../data/NN.data")
p.set_defaults(N=8)
p.set_defaults(mode="XOR")

opts, args = p.parse_args()
N = opts.N
outFile = opts.outFile
mode = opts.mode

del opts, args

#
#Object instantiations
#

p = Parity()
  
# Randomly generate system matrices S = {Phi, H, Q, R}
# Generate some real random matrices (NxN) - restricted to real, square matrices    

# Storage - initialise from zero
Pstore = [0] * (2**N) * 2  # matrix of parity data and input values

#
#Generate data 
#

if mode == "XOR":
    # Generate parity data
    
    # Data format length (number of bits)
    fmt = "0" + repr(N) + "b"
    fmt2 = '02b'
    for k in range(1,2**N+1):
        
        # First we generate a string with the binary value without the prefix
        # "0b", we then proceed to space the binary digits out such that each
        # digit can be interpreted as a separate vector entry input by the
        # neural network
        str2write = format(k-1, fmt)
        Pstore[2*k - 2] = " ".join(str2write)
        str2write = format(p.parity(k-1) + 1, fmt2)
        Pstore[2*k - 1] = " ".join(str2write)
else:
    pass

#
# Save data and create directory if it doesn't exist
#

# Check if output file exists
if outFile[-1] == '/':
    outFile = outFile[:-1]

baseName = os.path.basename(outFile)
dirName = os.path.dirname(outFile)

# If not then create file
if not os.path.isdir(dirName):
    os.makedirs(dirName)
   
# Print .data header followed by python list contents
f = file(outFile,'w') 
s = repr(2**N) + " " + repr(N) + " 2\n" 
f.write(s)
for item in Pstore:
    print >> f, item
f.close()

print "Data printed to file %r" % outFile


























