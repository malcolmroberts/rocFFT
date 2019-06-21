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

usage = '''A timing script for rocfft the generates lots of data

Usage:

\talltime.py
\t\t-A          working directory A
\t\t-B          working directory B (optional)
\t\t-a          label for directory A
\t\t-b          label for directory B
\t\t-T          do not perform FFTs; just compile generated data into PDFs
\t\t-o          output directory
'''

def runcase(workingdir, xval, yval, zval, direction, rcfft, inplace, ntrial, nbatch):
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

    cmd.append("-n")
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
    dirA = "."
    dirB = None
    dryrun = False
    labelA = ""
    labelB = ""
    nbatch = 1
    outdir = "."
    
    try:
        opts, args = getopt.getopt(argv,"hA:B:Ta:b:o:")
    except getopt.GetoptError:
        print("error in parsing arguments.")
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h"):
            print(usage)
            exit(0)
        elif opt in ("-A"):
            dirA = arg
        elif opt in ("-B"):
            dirB = arg
        elif opt in ("-a"):
            labelA = arg
        elif opt in ("-b"):
            labelB = arg
        elif opt in ("-o"):
            outdir = arg
        elif opt in ("-T"):
            dryrun = True


    print("dirA: "+ dirA)
    print("labelA: "+ labelA)
    dirlist = [dirA]
    if not dirB == None:
        print("dirB: "+ dirB)
        print("labelB: "+ labelB)
        dirlist.append(dirB)
    print("outdir: " + outdir)

    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    pdflist = []
    captionlist = []
    for dimension in 1, 2, 3:
        sizes = [[536870912, 1]]
        #sizes = [[1024, 1]]
        batchsizes = [1]
        if dimension == 1:
            sizes.append([32768,100000])
        if dimension == 2:
            sizes = [[32768, 1]]
            sizes.append([32768,100])
            maxsize = 32768
        if dimension == 3:
            sizes = [[1024, 1]]
            sizes.append([1024, 10])
            
        for precision in "float", "double":
            for datatype in "c2c", "r2c":
                for maxsize, nbatch in sizes:
                    filelist = []
                    labellist = []
                    for direction in -1, 1:
                        for wdir in dirlist:
                            cmd = ["./timing.py"]

                            outfile = "dirA" if wdir == dirA else "dirB"


                            cmd.append("-w")
                            cmd.append(wdir)

                            cmd.append("-b")
                            cmd.append(str(nbatch))


                            cmd.append("-x")
                            cmd.append("2")
                            cmd.append("-X")
                            cmd.append(str(maxsize))

                            if dimension > 1:
                                cmd.append("-y")
                                cmd.append("2")
                                cmd.append("-Y")
                                cmd.append(str(maxsize))

                            if dimension > 2:
                                cmd.append("-z")
                                cmd.append("2")
                                cmd.append("-Z")
                                cmd.append(str(maxsize))

                            cmd.append("-D")
                            cmd.append(str(direction))
                            if direction == 1:
                                outfile += "inv"

                            cmd.append("-d")
                            cmd.append(str(dimension))

                            if datatype == "r2c":
                                cmd.append("-R")
                            outfile += datatype

                            outfile += str(dimension)
                            outfile += precision
                            outfile += "n" + str(nbatch)
                            outfile += ".dat"
                            outfile = os.path.join(outdir, outfile)
                            filelist.append(outfile)

                            label = ""
                            if wdir == dirA:
                                label += "dirA" if labelA == "" else labelA
                            else:
                                label += "dirB" if labelB == "" else labelB
                            label += " direct" if (direction == -1) else " inverse"

                            labellist.append(label)

                            cmd.append("-o")
                            cmd.append(outfile)

                            print(" ".join(cmd))
                            if not dryrun:
                                fout = tempfile.TemporaryFile(mode="w+")
                                ferr = tempfile.TemporaryFile(mode="w+")

                                proc = subprocess.Popen(cmd, stdout=fout, stderr=ferr,
                                                        env=os.environ.copy())
                                proc.wait()
                                rc = proc.returncode
                                if rc != 0:
                                    print("****fail****")
                    asycmd = ["asy", "-f", "pdf", "datagraphs.asy"]
                    asycmd.append("-u")
                    asycmd.append('filenames="' + ",".join(filelist) + '"')

                    asycmd.append("-u")
                    asycmd.append('legendlist="' + ",".join(labellist) + '"')

                    if dirB != None:
                        asycmd.append("-u")
                        asycmd.append('speedup=true')

                    
                    
                    asycmd.append("-o")

                    outpdf = "time" + str(dimension) + datatype + precision + "n" + str(nbatch) + ".pdf"
                    #outpdf = os.path.join(outdir,outpdf)
                    asycmd.append(os.path.join(outdir,outpdf))

                    print(" ".join(asycmd))

                    asyproc =  subprocess.Popen(asycmd, env=os.environ.copy())
                    asyproc.wait()
                    asyrc = asyproc.returncode
                    if asyrc != 0:
                        print("****asy fail****")
                    else:
                        caption = "Dimension: " + str(dimension)
                        caption += ", type: "+ ("complex" if datatype == "c2c" else "real/complex")
                        caption += ", precision: "+ precision
                        caption += ", batch size: "+ str(nbatch)

                        pdflist.append([outpdf, caption ])

    for stuff in pdflist:
        print(stuff)

    header = '''\documentclass[12pt]{article}
\\usepackage{graphicx}
\\usepackage{url}
\\author{Malcolm Roberts}

\\begin{document}
'''
    footer = '''

\\end{document}
'''
    texstring = header

    texstring += "\\begin{tabular}{lll}"
    texstring += "dirA: &\\url{"+ dirA+"} & " + labelA +"\\\\\n"
    if not dirB == None:
        texstring += "dirB: &\\url{"+ dirB+"} & " + labelB + "\\\\\n"
    texstring += "\\end{tabular}"
        
    for pdffile, caption in pdflist:
        texstring += '''\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=\\textwidth]{'''
        texstring += pdffile
        texstring += '''}
  \\caption{''' + caption + '''}
\\end{figure}
'''

    texstring += footer
   
    fname = os.path.join(outdir, 'figs.tex')

    with open(fname, 'w') as outfile:
        outfile.write(texstring)

    fout = open(os.path.join(outdir, "texcmd.log"), 'w+')
    ferr = open(os.path.join(outdir, "texcmd.err"), 'w+')
                    
    latexcmd = ["latexmk", "-pdf", 'figs.tex']
    print(" ".join(latexcmd))
    texproc =  subprocess.Popen(latexcmd, cwd=outdir, stdout=fout, stderr=ferr,
                                env=os.environ.copy())
    texproc.wait()
    fout.close()
    ferr.close()
    texrc = asyproc.returncode
    if texrc != 0:
        print("****tex fail****")
                                                                
    

if __name__ == "__main__":
    main(sys.argv[1:])
                        
