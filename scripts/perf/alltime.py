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
\t\t-g          generate graphs via Asymptote: 0(default) or 1
\t\t-d          device number (default: 0)
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
    devicenum = 0
    doAsy = True
    
    try:
        opts, args = getopt.getopt(argv,"hA:f:B:Tt:a:b:o:S:sg:")
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
        elif opt in ("-g"):
            if int(arg) == 0:
                doAsy = False
            if int(arg) == 1:
                doAsy = True
        elif opt in ("-d"):
            devicenum = int(arg)
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
    if not binaryisok(dirA, "rocfft-rider"):
        print("unable to find " + "rocfft-rider" + " in " + dirA)
        print("please specify with -A")
        sys.exit(1)
        
    dirlist = [dirA]
    if not dirB == None:
        if labelB == None:
            labelB = dirB

        print("dirB: "+ dirB)
        print("labelB: "+ labelB)
        if not binaryisok(dirB, "rocfft-rider"):
            print("unable to find " + "rocfft-rider" + " in " + dirB)
            print("please specify with -B")
            sys.exit(1)

        dirlist.append(dirB)
    print("outdir: " + outdir)
    if shortrun:
        print("short run")
    print("output format: " + docformat)
    print("device number: " + str(devicenum))
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not dryrun:
        import getspecs
        specs = "Host info:\n"
        specs += "\tcpu info: " + getspecs.getcpu() + "\n"
        specs += "\tram: " + getspecs.getram() + "\n"
        specs += "\tdistro: " + getspecs.getdistro() + "\n"
        specs += "\tkernel version: " + getspecs.getkernel() + "\n"
        specs += "\trocm version: " + getspecs.getrocmversion() + "\n"
        specs += "Device info:\n"
        specs += "\tdevice: " + getspecs.getdeviceinfo(devicenum) + "\n"
        specs += "\tvbios version: " + getspecs.getvbios(devicenum) + "\n"
        specs += "\tvram: " + getspecs.getvram(devicenum) + "\n"
        specs += "\tperformance level: " + getspecs.getperflevel(devicenum) + "\n"
        specs += "\tsystem clock: " + getspecs.getsclk(devicenum) + "\n"
        specs += "\tmemory clock: " + getspecs.getmclk(devicenum) + "\n"

        with open(os.path.join(outdir, "specs.txt"), "w+") as f:
            f.write(specs)
        
    pdflist = []
    dimensions = [1, 2, 3]
    for dimension in dimensions:
        pdflist.append([])

        # Sizes firmat: minvalue, maxvalue, batch size, ratio between elements, aspect ratio(s)
        sizes = []
        if dimension == 1:
            if shortrun:
                sizes.append( [1024, 16384,   1, 2, [1]] )
                sizes.append( [729,   2187,   1, 3, [1]] )
                sizes.append( [625,  15625,   1, 5, [1]] )
                sizes.append( [2,     1024, 100, 2, [1]] )
            else:
                sizes.append( [1024, 536870912, 1,      2, [1]] )
                sizes.append( [729,  129140163, 1,      3, [1]] )
                sizes.append( [625,  244140625, 1,      5, [1]] )
                sizes.append( [2,        32768, 100000, 2, [1]] )
        if dimension == 2:
            if shortrun:
                sizes.append( [64,  8192,  1, 2, [1]] )
                sizes.append( [64,  8192,  1, 2, [2]] )
                sizes.append( [2,  1024,  10, 2, [1]] )
            else:
                sizes.append( [16, 32768,    1, 2, [1]] )
                sizes.append( [2,   1024, 1000, 2, [1]] )
        if dimension == 3:
            if shortrun:
                sizes.append( [2,  64, 1, 2, [1,1]] )
                sizes.append( [2, 128, 1, 2, [1,2]] )
            else:
                sizes.append( [128, 1024,  1, 2, [1, 1]] )
                sizes.append( [2,   1024, 10, 2, [1, 1]] )
                sizes.append( [2,   1024, 10, 2, [1,16]] )
                sizes.append( [2,   1024, 10, 2, [8, 8]] )

            
        for precision in "float", "double":
            for ffttype in "c2c", "r2c":
                for inplace in True, False:
                    for minsize, maxsize, nbatch, radix, ratio in sizes:
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
                                cmd.append(str(minsize))
                                cmd.append("-X")
                                cmd.append(str(maxsize))

                                if dimension > 1:
                                    cmd.append("-y")
                                    cmd.append(str(minsize))
                                    cmd.append("-Y")
                                    cmd.append(str(maxsize))

                                if dimension > 2:
                                    cmd.append("-z")
                                    cmd.append(str(minsize))
                                    cmd.append("-Z")
                                    cmd.append(str(maxsize))

                                cmd.append("-r")
                                cmd.append(str(radix))

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

                                outfile += "_radix" + str(radix)

                                outfile += "_dim" + str(dimension)
                                outfile += "_" + precision
                                outfile += "_n" + str(nbatch)
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

                        asycmd = ["asy"]
                        if docformat == "pdf":
                            asycmd.append("-f")
                            asycmd.append("pdf")
                        if docformat == "docx":
                            asycmd.append("-f")
                            asycmd.append("png")
                            asycmd.append("-render")
                            asycmd.append("16")
                        asycmd.append("datagraphs.asy")

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
                        outpdf += "_radix" + str(radix)

                        if docformat == "pdf":
                            outpdf += ".pdf"

                        if docformat == "docx":
                            outpdf += ".png"

                        #outpdf = os.path.join(outdir,outpdf)
                        asycmd.append(os.path.join(outdir,outpdf))

                        print(" ".join(asycmd))
                    
                        if doAsy:
                            asyproc =  subprocess.Popen(asycmd, env=os.environ.copy())
                            asyproc.wait()
                            asyrc = asyproc.returncode
                            if asyrc != 0:
                                print("****asy fail****")

                        caption = "Dimension: " + str(dimension)
                        caption += ", type: "+ ("complex" if ffttype == "c2c" else "real/complex")
                        caption += ", in-place" if inplace else ", out-of-place"
                        caption += ", precision: "+ precision
                        caption += ", batch size: "+ str(nbatch)
                        caption += ", radix: "+ str(radix)
                        if dimension > 1:
                            caption += ", aspect ratio 1:"  + str(ratio[0])
                            if dimension > 2:
                                caption += ":" + str(ratio[1])

                        pdflist[-1].append([outpdf, caption ])

    if docformat == "pdf":
        maketex(pdflist, dirA, dirB, labelA, labelB, outdir)    
    if docformat == "docx":
        makedocx(pdflist, dirA, dirB, labelA, labelB, outdir)    

def binaryisok(dirname, progname):
    prog = os.path.join(dirname, progname)
    return os.path.isfile(prog)
        
def maketex(pdflist, dirA, dirB, labelA, labelB, outdir):
    
    header = '''\documentclass[12pt]{article}
\\usepackage{graphicx}
\\usepackage{url}
\\author{Malcolm Roberts}
\\begin{document}
'''
    texstring = header

    texstring += "\n\\section{Introduction}\n"
    
    texstring += "Each data point represents the median of 10 values, with error bars showing the 95\\% confidence interval for the median.\n\n"

    texstring += "\\vspace{1cm}\n"
    
    texstring += "\\begin{tabular}{ll}"
    texstring += labelA +" &\\url{"+ dirA+"} \\\\\n"
    if not dirB == None:
        texstring += labelB +" &\\url{"+ dirB+"} \\\\\n"
    texstring += "\\end{tabular}\n\n"

    texstring += "\\vspace{1cm}\n"
    
    specfilename = os.path.join(outdir, "specs.txt")
    if os.path.isfile(specfilename):
        specs = ""
        with open(specfilename, "r") as f:
            specs = f.read()

        for line in specs.split("\n"):
            if line.startswith("Host info"):
                texstring += "\\noindent " + line
                texstring += "\\begin{itemize}\n"
            elif line.startswith("Device info"):
                texstring += "\\end{itemize}\n"
                texstring += line 
                texstring += "\\begin{itemize}\n"
            else:
                if line.strip() != "":
                    texstring += "\\item " + line + "\n"
        texstring += "\\end{itemize}\n"
        texstring += "\n"
        
    texstring += "\\clearpage\n"
        
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

'''
            
        texstring += "\\clearpage\n"
            
            
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

def makedocx(pdflist, dirA, dirB, labelA, labelB, outdir):
    import docx

    document = docx.Document()

    document.add_heading('rocFFT benchmarks', 0)

    document.add_paragraph('Each data point represents the median of 10 values, with error bars showing the 95% confidence interval for the median')

    specfilename = os.path.join(outdir, "specs.txt")
    if os.path.isfile(specfilename):
        with open(specfilename, "r") as f:
            specs = f.read()
        for line in specs.split("\n"):
            document.add_paragraph(line)

    
    for i in range(len(pdflist)):
        dimension = i + 1
        print(dimension)
        document.add_heading('Dimension ' + str(dimension), level=1)
        for outpdf, caption in pdflist[i]:
            print(outpdf, caption)
            document.add_picture(os.path.join(outdir,outpdf), width=docx.shared.Inches(4))
            document.add_paragraph(caption)
                         
    document.save(os.path.join(outdir,'figs.docx'))
    
if __name__ == "__main__":
    main(sys.argv[1:])
                        
