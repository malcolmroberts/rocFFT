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
\t\t-o          output directory
\t\t-S          plot speedup (default: 1, disabled: 0)
\t\t-t          data type: time (default) or gflops
\t\t-s          short run
\t\t-T          do not perform FFTs; just generate document
\t\t-f          document format: pdf (default) or docx
'''

def main(argv):
    dirA = "."
    dirB = None
    dryrun = False
    labelA = None
    labelB = None
    nbatch = 1
    outdir = "."
    speedup = True
    datatype = "time"
    shortrun = False
    docformat = "pdf"
    
    try:
        opts, args = getopt.getopt(argv,"hA:f:B:Tt:a:b:o:S:s")
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
        elif opt in ("-s"):
            shortrun = True
        elif opt in ("-S"):
            if int(arg) == 0:
                speedup = False
            if int(arg) == 1:
                speedup = True
        elif opt in ("-t"):
            if arg not in ["time", "gflops"]:
                print("data type must be time or gflops")
                print(usage)
                sys.exit(1)
            datatype = arg
        elif opt in ("-f"):
            goodvals = ["pdf", "docx"]
            if arg not in goodvals:
                print("error: format must in " + " ".join(goodvals))
                print(usage)
                sys.exit(1)
            docformat = arg

    if labelA == None:
        labelA = dirA
        
    print("dirA: "+ dirA)
    print("labelA: "+ labelA)
    dirlist = [dirA]
    if not dirB == None:
        if labelB == None:
            labelB = dirB

        print("dirB: "+ dirB)
        print("labelB: "+ labelB)
        dirlist.append(dirB)
    print("outdir: " + outdir)
    if shortrun:
        print("short run")
    print("output format: " + docformat)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    pdflist = []
    for dimension in 1, 2, 3:
        pdflist.append([])
        
        sizes = None
        aspectratios = None
        if dimension == 1:
            if shortrun:
                sizes = [[1048576, 1]]
                sizes.append([1024, 100000])
            else:
                sizes = [[536870912, 1]]
                sizes.append([32768, 100000])
            aspectratios = [[1]]
        if dimension == 2:
            if shortrun:
                sizes = [[1024, 1]]
                sizes.append([32768, 1000])
            else:
                sizes = [[1024, 1]]
                sizes.append([32768, 1000])
            aspectratios = [[1], [16]]
        if dimension == 3:
            if shortrun:
                sizes = [[64, 1]]
                sizes.append([64, 100])
            else:
                sizes = [[1024, 1]]
                sizes.append([1024, 100])
            aspectratios = [[1, 1], [1, 16],  [8, 8]]
            
        for precision in "float", "double":
            for ffttype in "c2c", "r2c":
                for inplace in True, False:
                    for maxsize, nbatch in sizes:
                        for ratio in aspectratios:
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
                                        cmd.append(str(2 * ratio[0]))
                                        cmd.append("-Y")
                                        cmd.append(str(maxsize))

                                    if dimension > 2: # FIXME: add ratios
                                        cmd.append("-z")
                                        cmd.append(str(2 * ratio[1]))
                                        cmd.append("-Z")
                                        cmd.append(str(maxsize))

                                    cmd.append("-D")
                                    cmd.append(str(direction))
                                    if direction == 1:
                                        outfile += "inv"

                                    cmd.append("-d")
                                    cmd.append(str(dimension))

                                    if dimension > 1:
                                        outfile += "ratio" + "_" + str(ratio[0])
                                        if dimension > 2:
                                            outfile += "_" + str(ratio[1])
                                    
                                    cmd.append("-t")
                                    cmd.append(datatype)

                                    if ffttype == "r2c":
                                        cmd.append("-R")
                                    outfile += ffttype

                                    if inplace:
                                        cmd.append("-I")
                                        outfile += "inplace"
                                    else:
                                        outfile += "outofplace"

                                    outfile += str(dimension)
                                    outfile += precision
                                    outfile += "n" + str(nbatch)
                                    outfile += ".dat"
                                    outfile = os.path.join(outdir, outfile)
                                    filelist.append(outfile)

                                    label = ""
                                    if wdir == dirA:
                                        label += labelA
                                    else:
                                        label += labelB
                                    label += " direct" if (direction == -1) else " inverse"
                                    #label += " in-place " if inplace else " out-of-place "
                                    
                                    # if dimension > 1:
                                    #     label += " aspect ratio: 1:" + str(ratio[0])
                                    #     if dimension > 2:
                                    #         label += ":" + str(ratio[1])
                                                                            
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

                            if dirB != None and speedup:
                                asycmd.append("-u")
                                asycmd.append('speedup=true')
                            else:
                                asycmd.append("-u")
                                asycmd.append('speedup=false')

                            if datatype == "gflops":
                                asycmd.append("-u")
                                asycmd.append('ylabel="GFLOPs"')

                            asycmd.append("-o")

                            outpdf = "time" + str(dimension) + ffttype + precision + "n" + str(nbatch)
                            outpdf += "inplace" if inplace else "outofplace"
                            if dimension > 1:
                                outpdf += "ratio" + "_" + str(ratio[0])
                                if dimension > 2:
                                    outpdf += "_" + str(ratio[1])

                            outpdf += ".pdf"
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
                                caption += ", type: "+ ("complex" if ffttype == "c2c" else "real/complex")
                                caption += ", in-place" if inplace else ", out-of-place"
                                caption += ", precision: "+ precision
                                caption += ", batch size: "+ str(nbatch)
                                if dimension > 1:
                                    caption += ", aspect ratio 1:"  + str(ratio[0])
                                    if dimension > 2:
                                        caption += ":" + str(ratio[1])


                                pdflist[-1].append([outpdf, caption ])

    maketex(pdflist, dirA, dirB, labelA, labelB, outdir)

def maketex(pdflist, dirA, dirB, labelA, labelB, outdir):
    
    header = '''\documentclass[12pt]{article}
\\usepackage{graphicx}
\\usepackage{url}
\\author{Malcolm Roberts}
\\begin{document}
'''
    texstring = header

    texstring += "\\begin{tabular}{ll}"
    texstring += labelA +" &\\url{"+ dirA+"} \\\\\n"
    if not dirB == None:
        texstring += labelB +" &\\url{"+ dirB+"} \\\\\n"
    texstring += "\\end{tabular}"

    texstring += "\\begin{tabular}{ll}"
    texstring += labelA +" &\\url{"+ dirA+"} \\\\\n"
    if not dirB == None:
        texstring += labelB +" &\\url{"+ dirB+"} \\\\\n"
    texstring += "\\end{tabular}"

    ## FIXME: how to deal with this?


    for i in range(len(pdflist)):
        dimension = i + 1
        print(dimension)
        texstring += "\n\\section{Dimension " + str(dimension) + "}\n"
        for outpdf, caption in pdflist[i]:
            print(outpdf, caption)
            texstring += '''
\\centering
\\begin{figure}[htbp]
   \\includegraphics[width=\\textwidth]{'''
            texstring += outpdf
            texstring += '''}
   \\caption{''' + caption + '''}
\\end{figure}
\\newpage

'''
    texstring += "\n\\end{document}\n"
   
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
    texrc = texproc.returncode
    if texrc != 0:
        print("****tex fail****")

if __name__ == "__main__":
    main(sys.argv[1:])
                        
