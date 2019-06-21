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
\t\t-d <1,2,3>  default: 1dimension of transform
\t\t-x <int>    minimum problem size in x direction
\t\t-X <int>    maximum problem size in x direction
\t\t-x <int>    minimum problem size in x direction
\t\t-X <int>    maximum problem size in x direction
\t\t-b <int>    batch size'''

def runcase(workingdir, xval, yval, zval, direction, rcfft, inplace, ntrial, precision, nbatch,
            logfilename):
    progname = "rocfft-rider"
    prog = os.path.join(workingdir, progname)
    
    cmd = []
    cmd.append(prog)

    cmd.append("-p")
    cmd.append("10")

    cmd.append("-x")
    cmd.append(str(xval))

    cmd.append("-y")
    cmd.append(str(yval))

    cmd.append("-z")
    cmd.append(str(zval))

    cmd.append("-N")
    cmd.append(str(ntrial))

    if precision == "double":
        cmd.append("--double")

    cmd.append("-b")
    cmd.append(str(nbatch))
        
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

    print(logfilename)
    fout = open(logfilename, 'w+')
    proc = subprocess.Popen(cmd, cwd=os.path.join(workingdir,"..",".."), stdout=fout, stderr=fout,
                            env=os.environ.copy())
    proc.wait()
    rc = proc.returncode
    seconds = []
    
    if rc == 0:
        
        fout.seek(0)
        cout = fout.read()

        # ferr.seek(0)
        # cerr = ferr.read()

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
    
    return seconds
    

def main(argv):
    workingdir = "."
    dimension = 1
    xmin = 2
    xmax = 1024
    ymin = 2
    ymax = 1024
    zmin = 2
    zmax = 1024
    ntrial = 10
    outfilename = "timing.dat"
    direction = -1
    rcfft = False
    inplace = False
    precision = "float"
    nbatch = 1
    
    try:
        opts, args = getopt.getopt(argv,"hb:d:D:IN:o:Rw:x:X:y:Y:z:Z:f:")
    except getopt.GetoptError:
        print("error in parsing arguments.")
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h"):
            print(usage)
            exit(0)
        elif opt in ("-d"):
            dimension = int(arg)
            if not dimension in {1,2,3}:
                print("invalid dimension")
                print(usage)
                sys.exit(1)
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
        elif opt in ("-y"):
            xmin = int(arg)
        elif opt in ("-Y"):
            xmax = int(arg)
        elif opt in ("-z"):
            xmin = int(arg)
        elif opt in ("-Z"):
            xmax = int(arg)
        elif opt in ("-b"):
            nbatch = int(arg)
        elif opt in ("-f"):
            precition = arg
            if precision not in ["float", "double"]:
                wirte("precision must be float or double")

            
    print("workingdir: "+ workingdir)
    print("outfilename: "+ outfilename)
    print("ntrial: " + str(ntrial))
    print("xmin: "+ str(xmin) + " xmax: " + str(xmax))
    print("ymin: "+ str(ymin) + " ymax: " + str(ymax))
    print("zmin: "+ str(zmin) + " zmax: " + str(zmax))
    print("direction: " + str(direction))
    print("real/complex FFT? " + str(rcfft))
    print("in-place? " + str(inplace))
    print("batch-size: " + str(nbatch))

    progname = "rocfft-rider"
    prog = os.path.join(workingdir, progname)
    if not os.path.isfile(prog):
        print("**** Error: unable to find " + prog)
        sys.exit(1)


    with open(outfilename, 'w+') as outfile:
        # TODO: add metadata to output file
        xval = xmin
        yval = ymin if dimension > 1 else 1
        zval = zmin if dimension > 2 else 1
        while(xval <= xmax and yval < ymax and zval < zmax):
            print(xval)
            outfile.write(str(xval))
            logfilename = outfilename + ".log"
            seconds = runcase(workingdir, xval, yval, zval, direction, rcfft, inplace, ntrial,
                              precision, nbatch, logfilename)
            #print(seconds)
            outfile.write("\t")
            outfile.write(str(len(seconds)))
            for second in seconds:
                outfile.write("\t")
                outfile.write(str(second))
            outfile.write("\n")
            xval *= 2
            if dimension > 1:
                yval *= 2
            if dimension > 2:
                zval *= 2
        
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
                        
