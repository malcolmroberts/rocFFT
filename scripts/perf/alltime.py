#!/usr/bin/python3

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
\t\t-N          Number of samples (default: 10)
'''

def nextpow(val, radix):
    x = 1
    while(x <= val):
        x *= radix
    return x

class rundata:
    
    def __init__(self, wdir, diridx, label,
                 dimension, minsize, maxsize, nbatch, radix, ratio, ffttype,
                 direction):
        self.wdir = wdir
        self.diridx = diridx
        self.dimension = dimension
        self.minsize = minsize
        self.maxsize = maxsize
        self.nbatch = nbatch
        self.radix = radix
        self.ratio = ratio
        self.ffttype = ffttype
        self.precision = "double"
        self.inplace = True
        self.direction = -1
        self.label = label
            

    def outfilename(self, outdir):
        outfile = "dir" + str(self.diridx)
        if self.direction == 1:
            outfile += "_inv"
        if self.dimension > 1:
            outfile += "_ratio" + "_" + str(self.ratio[0])
        if self.dimension > 2:
            outfile += "_" + str(self.ratio[1])
        outfile += "_" + self.ffttype
        if self.inplace:
            outfile += "_inplace"
        else:
            outfile += "_outofplace"
        outfile += "_radix" + str(self.radix)
        outfile += "_dim" + str(self.dimension)
        outfile += "_" + self.precision
        outfile += "_n" + str(self.nbatch)
        outfile += ".dat"
        outfile = os.path.join(outdir, outfile)
        return outfile
        
    def runcmd(self, outdir, nsample):
        cmd = ["./timing.py"]
        
        cmd.append("-w")
        cmd.append(self.wdir)

        cmd.append("-N")
        cmd.append(str(nsample))
        
        cmd.append("-b")
        cmd.append(str(self.nbatch))
        
        cmd.append("-x")
        cmd.append(str(self.minsize))
        cmd.append("-X")
        cmd.append(str(self.maxsize))

        if self.dimension > 1:
            cmd.append("-y")
            cmd.append(str(self.minsize))
            cmd.append("-Y")
            cmd.append(str(self.maxsize))

        if self.dimension > 2:
            cmd.append("-z")
            cmd.append(str(self.minsize))
            cmd.append("-Z")
            cmd.append(str(self.maxsize))

        cmd.append("-r")
        cmd.append(str(self.radix))

        cmd.append("-D")
        cmd.append(str(self.direction))
        
        cmd.append("-d")
        cmd.append(str(self.dimension))

        cmd.append("-f")
        cmd.append(self.precision)
        
        if self.ffttype == "r2c":
            cmd.append("-R")
            
        cmd.append("-o")
        cmd.append(self.outfilename(outdir))
            
        return cmd

    def executerun(self, outdir, nsample):
        fout = tempfile.TemporaryFile(mode="w+")
        ferr = tempfile.TemporaryFile(mode="w+")
            
        proc = subprocess.Popen(self.runcmd(outdir, nsample), stdout=fout, stderr=ferr,
                                env=os.environ.copy())
        proc.wait()
        rc = proc.returncode
        if rc != 0:
            print("****fail****")
        return rc


class figure:
    def __init__(self, name, caption):
        self.name = name
        self.runs = []
        self.caption = caption
    
    def inputfiles(self, outdir):
        files = []
        for run in self.runs:
            files.append(run.outfilename(outdir))
        return files

    def labels(self):
        labels = []
        for run in self.runs:
            labels.append(run.label)
        return labels
    
    def filename(self, outdir, docformat):
        outfigure = self.name
        outfigure += ".pdf"
        # if docformat == "pdf":
        #     outfigure += ".pdf"
        # if docformat == "docx":
        #     outfigure += ".png"
        return os.path.join(outdir, outfigure)

        
    def asycmd(self, outdir, docformat, datatype, ncompare):
        asycmd = ["asy"]
        
        asycmd.append("-f")
        asycmd.append("pdf")
        # if docformat == "pdf":
        #     asycmd.append("-f")
        #     asycmd.append("pdf")
        # if docformat == "docx":
        #     asycmd.append("-f")
        #     asycmd.append("png")
        #     asycmd.append("-render")
        #     asycmd.append("8")
        asycmd.append("datagraphs.asy")

        asycmd.append("-u")
        asycmd.append('filenames="' + ",".join(self.inputfiles(outdir)) + '"')

        asycmd.append("-u")
        asycmd.append('legendlist="' + ",".join(self.labels()) + '"')

        # if dirB != None and speedup: # disabled for now
        #     asycmd.append("-u")
        #     asycmd.append('speedup=true')
        # else:
        #     asycmd.append("-u")
        #     asycmd.append('speedup=false')

        asycmd.append("-u")
        asycmd.append('speedup=' + str(ncompare))
            
        if datatype == "gflops":
            asycmd.append("-u")
            asycmd.append('ylabel="GFLOPs"')

        asycmd.append("-o")
        asycmd.append(self.filename(outdir, docformat) )
                    
        return asycmd

    def executeasy(self, outdir, docformat, datatype, ncompare):
        asyproc =  subprocess.Popen(self.asycmd(outdir, docformat, datatype, ncompare),
                                    env=os.environ.copy())
        asyproc.wait()
        asyrc = asyproc.returncode
        if asyrc != 0:
            print("****asy fail****")
        return asyrc

        
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
    nsample = 10
    
    try:
        opts, args = getopt.getopt(argv,"hA:f:B:Tt:a:b:o:S:sg:d:N:")
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
        elif opt in ("-N"):
            nsample = int(arg)
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
    
    if not dryrun and not binaryisok(dirA, "rocfft-rider"):
        print("unable to find " + "rocfft-rider" + " in " + dirA)
        print("please specify with -A")
        sys.exit(1)
        
    dirlist = [[dirA, labelA]]
    if not dirB == None:
        if labelB == None:
            labelB = dirB

        print("dirB: "+ dirB)
        print("labelB: "+ labelB)
        if not dryrun and not binaryisok(dirB, "rocfft-rider"):
            print("unable to find " + "rocfft-rider" + " in " + dirB)
            print("please specify with -B")
            sys.exit(1)

        dirlist.append([dirB, labelB])
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

    figs = []


    dimension = 1

    # FFT directions
    forwards = -1
    backwards = 1
    
    nbatch = 1

    min1d = 256 if shortrun else 1024
    max1d = 4000 if shortrun else 536870912
    
    fig = figure("1d_c2c", "1D complex transforms")
    for radix in [2, 3, 5, 7]:
        for idx, lwdir in enumerate(dirlist):
            wdir = lwdir[0]
            label = lwdir[1]
            fig.runs.append( rundata(wdir, idx, label + "radix " + str(radix) ,
                                     dimension, nextpow(min1d, radix), max1d, nbatch, radix, [], "c2c",
                                     forwards) )
    figs.append(fig)

    fig = figure("1d_r2c", "1D real-to-complex transforms")
    for radix in [2, 3, 5, 7]:
        for idx, lwdir in enumerate(dirlist):
            wdir = lwdir[0]
            label = lwdir[1]
            fig.runs.append( rundata(wdir, idx, label + "radix " + str(radix) ,
                                     dimension, nextpow(min1d, radix), max1d, nbatch, radix, [], "r2c",
                                     forwards) )
    figs.append(fig)

    
    nbatch = 100 if shortrun else 100000
    min1d = 8
    max1d = 1024 if shortrun else 32768
    fig = figure("1d_batch_c2c", "1D complex transforms batch size "+ str(nbatch))
    for radix in [2, 3, 5]:
        for idx, lwdir in enumerate(dirlist):
            wdir = lwdir[0]
            label = lwdir[1]
            fig.runs.append( rundata(wdir, idx, label + "radix " + str(radix) ,
                                     dimension, nextpow(min1d, radix), max1d, nbatch, radix, [], "c2c",
                                     forwards) )
            
    figs.append(fig)

    fig = figure("1d_batch_r2c", "1D real-to-complex transforms batch size " + str(nbatch))
    for radix in [2, 3, 5]:
        for idx, lwdir in enumerate(dirlist):
            wdir = lwdir[0]
            label = lwdir[1]
            fig.runs.append( rundata(wdir, idx, label + "radix " + str(radix) ,
                                     dimension, nextpow(min1d, radix), max1d, nbatch, radix, [], "r2c",
                                     forwards) )
    figs.append(fig)

    dimension = 2

    nbatch = 1
    min2d = 64 if shortrun else 128
    max2d = 8192 if shortrun else 32768

    fig = figure("2d_c2c", "2D complex transforms")
    for radix in [2, 3, 5]:
        for idx, lwdir in enumerate(dirlist):
            wdir = lwdir[0]
            label = lwdir[1]
            fig.runs.append( rundata(wdir, idx, label + "radix " + str(radix), dimension,
                                     nextpow(min2d, radix), max2d, nbatch, radix, [1], "c2c",
                                     forwards) )
    figs.append(fig)

    fig = figure("2d_r2c", "2D real-to-complex transforms")
    for radix in [2, 3, 5]:
        for idx, lwdir in enumerate(dirlist):
            wdir = lwdir[0]
            label = lwdir[1]
            fig.runs.append( rundata(wdir, idx, label + "radix " + str(radix), dimension,
                                     nextpow(min2d, radix), max2d, nbatch, radix, [1], "r2c",
                                     forwards) )
    figs.append(fig)

   
    fig = figure("2d_c2r", "2D complex-to-real transforms")
    for radix in [2, 3, 5]:
        for idx, lwdir in enumerate(dirlist):
            wdir = lwdir[0]
            label = lwdir[1]
            fig.runs.append( rundata(wdir, idx, label + "radix " + str(radix), dimension,
                                     nextpow(min2d, radix), max2d, nbatch, radix, [1], "r2c",
                                     backwards) )
    figs.append(fig)


    fig = figure("2d_c2c_r2", "2D complex transforms with aspect ratio N:2N")
    for idx, lwdir in enumerate(dirlist):
        wdir = lwdir[0]
        label = lwdir[1]
        fig.runs.append( rundata(wdir, idx, label + "radix 2",
                                 dimension, min2d, max2d, nbatch, 2, [2], "c2c",
                                 forwards) )
    figs.append(fig)

    fig = figure("2d_r2c_r2", "2D real-to-complex transforms with aspect ratio N:2N")
    for idx, lwdir in enumerate(dirlist):
        fig.runs.append( rundata(wdir, idx, label + "radix 2",
                                 dimension, min2d, max2d, nbatch, 2, [2], "r2c",
                                 forwards) )
    figs.append(fig)

    
    dimension = 3
    min3d = 16
    max3d = 128 if shortrun else 1024

    fig = figure("3d_c2c", "3D complex transforms")
    for radix in [2, 3, 5]:
        for idx, lwdir in enumerate(dirlist):
            wdir = lwdir[0]
            label = lwdir[1]
            fig.runs.append( rundata(wdir, idx, label + "radix " + str(radix), dimension,
                                     nextpow(min3d, radix), max3d, nbatch, radix, [1,1], "c2c",
                                     forwards) )
    figs.append(fig)

    fig = figure("3d_r2c", "3D real-to-complex transforms")
    for radix in [2, 3, 5]:
        for idx, lwdir in enumerate(dirlist):
            wdir = lwdir[0]
            label = lwdir[1]
            fig.runs.append( rundata(wdir, idx, label + "radix " + str(radix), dimension,
                                     nextpow(min3d, radix), max3d, nbatch, radix, [1,1], "r2c",
                                     forwards) )
    figs.append(fig)

    fig = figure("3d_c2r", "3D complex-to-real transforms")
    for radix in [2, 3, 5]:
        for idx, lwdir in enumerate(dirlist):
            wdir = lwdir[0]
            label = lwdir[1]
            fig.runs.append( rundata(wdir, idx, label + "radix " + str(radix), dimension,
                                     nextpow(min3d, radix), max3d, nbatch, radix, [1,1], "r2c",
                                     backwards) )
    figs.append(fig)

    
    fig = figure("3d_c2c_aspect", "3D complex transforms with aspect ratio N:N:16N")
    for idx, lwdir in enumerate(dirlist):
        wdir = lwdir[0]
        label = lwdir[1]
        fig.runs.append( rundata(wdir, idx, label + "radix 2",
                                 dimension, min3d, max3d, nbatch, 2, [1,16], "c2c",
                                 forwards) )
    figs.append(fig)

    fig = figure("3d_r2c_aspect", "3D real-to-complex transforms with aspect ratio N:N:16N")
    for idx, lwdir in enumerate(dirlist):
        wdir = lwdir[0]
        label = lwdir[1]
        fig.runs.append( rundata(wdir, idx, label + "radix 2",
                                 dimension, min3d, max3d, nbatch, 2, [1,16], "r2c",
                                 forwards) )
    figs.append(fig)



    for fig in figs:
        print(fig.name)
        for run in fig.runs:
            print(" ".join(run.runcmd(outdir, nsample)))
            if not dryrun:
                run.executerun(outdir, nsample)

        ncompare = len(dirlist) if speedup else 0
        print(fig.labels())
        print(fig.asycmd(outdir, docformat, datatype, ncompare))
        fig.executeasy(outdir, docformat, datatype, ncompare)

    if docformat == "pdf":
        maketex(figs, outdir, nsample)    
    if docformat == "docx":
        makedocx(figs, outdir, nsample)    

def binaryisok(dirname, progname):
    prog = os.path.join(dirname, progname)
    return os.path.isfile(prog)
        
def maketex(figs, outdir, nsample):
    
    header = '''\documentclass[12pt]{article}
\\usepackage{graphicx}
\\usepackage{url}
\\author{Malcolm Roberts}
\\begin{document}
'''
    texstring = header

    texstring += "\n\\section{Introduction}\n"
    
    texstring += "Each data point represents the median of " + str(nsample) + " values, with error bars showing the 95\\% confidence interval for the median.  All transforms are double-precision, in-place, and forward.\n\n"


    
    texstring += "\\vspace{1cm}\n"
    
    # texstring += "\\begin{tabular}{ll}"
    # texstring += labelA +" &\\url{"+ dirA+"} \\\\\n"
    # if not dirB == None:
    #     texstring += labelB +" &\\url{"+ dirB+"} \\\\\n"
    # texstring += "\\end{tabular}\n\n"

    # texstring += "\\vspace{1cm}\n"
    
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

    texstring += "\n\\section{Figures}\n"
    
    for fig in figs:
        print(fig.filename(outdir, "pdf"))
        print(fig.caption)
        texstring += '''
\\centering
\\begin{figure}[htbp]
   \\includegraphics[width=\\textwidth]{'''
        texstring += fig.filename("", "pdf")
        texstring += '''}
   \\caption{''' + fig.caption + '''}
\\end{figure}
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

def pdf2emf(pdfname):
    svgname = pdfname.replace(".pdf",".svg")
    cmd_pdf2svg = ["pdf2svg", pdfname, svgname]
    proc = subprocess.Popen(cmd_pdf2svg, env=os.environ.copy())
    proc.wait()
    if proc.returncode != 0:
        print("pdf2svg failed!")
        sys.exit(1)

    emfname = pdfname.replace(".pdf",".emf")
    cmd_svg2emf = ["inkscape", svgname, "-M", emfname]
    proc = subprocess.Popen(cmd_svg2emf, env=os.environ.copy())
    proc.wait()
    if proc.returncode != 0:
        print("svg2emf failed!")
        sys.exit(1)
    
    return emfname
        
def makedocx(figs, outdir, nsample):
    import docx

    document = docx.Document()

    document.add_heading('rocFFT benchmarks', 0)

    document.add_paragraph("Each data point represents the median of " + str(nsample) + " values, with error bars showing the 95% confidence interval for the median.  Transforms are double-precision, forward, and in-place.")

    specfilename = os.path.join(outdir, "specs.txt")
    if os.path.isfile(specfilename):
        with open(specfilename, "r") as f:
            specs = f.read()
        for line in specs.split("\n"):
            document.add_paragraph(line)

    for fig in figs:
        print(fig.filename(outdir, "docx"))
        print(fig.caption)
        emfname = pdf2emf(fig.filename(outdir, "docx"))
        # NB: emf support does not work; adding the filename as a placeholder
        document.add_paragraph(emfname)
        #document.add_picture(emfname, width=docx.shared.Inches(6))
        document.add_paragraph(fig.caption)
                         
    document.save(os.path.join(outdir,'figs.docx'))
    
if __name__ == "__main__":
    main(sys.argv[1:])
                        
