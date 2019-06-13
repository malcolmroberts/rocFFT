#!/usr/bin/python3

# a timing script for FFTs and convolutions using OpenMP

import sys, getopt
import numpy as np
from math import *
import subprocess
import os
import re # regexp package
import shutil
import tempfile

usage = '''A timing script for rocfft

Usage:
\ttiming.py
\t\t-D <-1,1>   default: -1 (forward).  Direction of transform
\t\t-I          make transform in-place
\t\t-N <int>    number of tests per problem size
\t\t-o <string> name of output file
\t\t-R          set transform to be real/complex or complex/real
\t\t-w <string> set working directory for rocfft-rider
\t\t-x <int>    minimum problem size in x direction
\t\t-X <int>    maximum problem size in x direction'''

def runcase(workingdir, xval, direction, rcfft, inplace, ntrial):
    progname = "rocfft-rider"
    prog = os.path.join(workingdir, progname)


        
    
    cmd = []
    cmd.append(prog)

    cmd.append("-p")
    cmd.append("10")

    cmd.append("-x")
    cmd.append(str(xval))

    cmd.append("-N")
    cmd.append(str(ntrial))


    ttype = -1
    itype = ""
    otype = ""
    if rcfft:
        if (direction == -1):
            ttype = 2
            itype = 2
            otype = 3
        if (direction == 1):
            ttype = 3
            itype = 3
            otype = 2
    else:
        itype = 0
        otype = 0
        if (direction == -1):
            ttype = 0
        if (direction == 1):
            ttype = 1
    cmd.append("--transformType")
    cmd.append(str(ttype))

    cmd.append("--inArrType")
    cmd.append(str(itype))

    cmd.append("--outArrType")
    cmd.append(str(otype))
    
    
    print(cmd)
   
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")

    proc = subprocess.Popen(cmd, cwd=os.path.join(workingdir,"..",".."), stdout=fout, stderr=ferr, env=os.environ.copy())
    proc.wait()
    rc = proc.returncode

    seconds = []
    
    if rc == 0:
        
        fout.seek(0)
        cout = fout.read()

        ferr.seek(0)
        cerr = ferr.read()

        searchstr = "Execution gpu time: "
        for line in cout.split("\n"):
            #print(line)
            if line.startswith(searchstr):
                # Line ends with "ms", so remove that.
                ms_string = line[len(searchstr):-2]
                #print(ms_string)
                for val in ms_string.split():
                    #print(val)
                    seconds.append(1e-3 * float(val))
                
        #print("seconds: ", seconds)
    else:
        print("\twell, that didn't work")
        print(rc)
        print(" ".join(cmd))
        return []
                
    fout.close()
    ferr.close()
    
    return seconds
    

def main(argv):
    dryrun = False
    workingdir = "."
    xmin = 2
    xmax = 1024
    ntrial = 10
    outfilename = "timing.dat"
    direction = -1
    rcfft = False
    inplace = False
    
    try:
        opts, args = getopt.getopt(argv,"hdD:IN:o:Rw:x:X:")
    except getopt.GetoptError:
        print("error in parsing arguments.")
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h"):
            print(usage)
            exit(0)
        elif opt in ("-d"):
            dryrun = True
        elif opt in ("-D"):
            if(int(arg) in [-1,1]):
                direction = int(arg)
            else:
                print("invalid direction: " + arg)
                print(usage)
                sys.exit(1)
        elif opt in ("-I"):
            inplace = True
        elif opt in ("-o"):
            outfilename = arg
        elif opt in ("-R"):
            rcfft = True
        elif opt in ("-w"):
            workingdir = arg
        elif opt in ("-N"):
            ntrial = int(arg)
        elif opt in ("-x"):
            xmin = int(arg)
        elif opt in ("-X"):
            xmax = int(arg)
            
    print("workingdir: "+ workingdir)
    print("outfilename: "+ outfilename)
    print("ntrial: " + str(ntrial))
    print("xmin: "+ str(xmin) + " xmax: " + str(xmax))
    print("direction: " + str(direction))
    print("real/complex FFT? " + str(rcfft))
    print("in-place? " + str(inplace))

    progname = "rocfft-rider"
    prog = os.path.join(workingdir, progname)
    if not os.path.isfile(prog):
        print("**** Error: unable to find " + prog)
        sys.exit(1)
    
    with open(outfilename, 'w+') as outfile:
        # TODO: add metadata to output file
        xval = xmin
        while(xval <= xmax):
            print(xval)
            outfile.write(str(xval))
            seconds = runcase(workingdir, xval, direction, rcfft, inplace, ntrial)
            #print(seconds)
            outfile.write("\t")
            outfile.write(str(len(seconds)))
            for second in seconds:
                outfile.write("\t")
                outfile.write(str(second))
            outfile.write("\n")
            xval *= 2
        
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
                        
