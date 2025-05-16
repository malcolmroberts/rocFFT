# Copyright (C) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# FFT problem generators...
#
# A generator must implement a single method
#
#     generate_problems(self)
#
# This method yields problems, which are instances of 'Problem'.
#

import itertools
import logging
import json

from dataclasses import dataclass, field
from pathlib import Path as path
from typing import Dict, List, Mapping, Generator
from perflib.utils import sjoin

top = path(__file__).resolve().parent.parent


def mktag(tag, dimension, precision, direction, inplace, real):
    t = [
        tag,
        str(dimension) + 'D', precision, {
            -1: 'forward',
            1: 'backward'
        }[direction], {
            True: 'real',
            False: 'complex'
        }[real], {
            True: 'in-place',
            False: 'out-of-place'
        }[inplace]
    ]
    return "_".join(t)


@dataclass
class Problem:
    length: List[int]
    direction: int = -1
    nbatch: int = 1
    precision: str = "single"
    real: bool = False

    istride: List[int] = None
    ostride: List[int] = None
    idist: int = 0
    odist: int = 0
    inplace: bool = False

    nranks: int = 1
    imgrid: List[int] = None
    omgrid: List[int] = None
    
    gpus_per_rank: int = 1
    ingrid: List[int] = None
    outgrid: List[int] = None
    
    min_wgs: int = 64
    max_wgs: int = 512
    
    full_token: bool = False
    tag: str = None
    meta: Dict[str, str] = field(default_factory=dict)

    def toJSON(self):
        tuning_dict = self.__dict__
        del tuning_dict['tag']
        del tuning_dict['meta']
        return tuning_dict


@dataclass
class VerbatimGenerator:
    problems: List[Problem]

    def generate_problems(self):
        for p in self.problems:
            yield p


@dataclass
class FilteredProblemGenerator:
    dimension: List[int] = field(default_factory=lambda: [1, 2, 3])
    direction: List[int] = field(default_factory=lambda: [-1, 1])
    inplace: List[bool] = field(default_factory=lambda: [True, False])
    real: List[bool] = field(default_factory=lambda: [True, False])
    precision: List[str] = field(default_factory=lambda: ["single", "double"])

    maxnodes: int = 0
    nranks: int = 1

    gpuspernode: int = 0
    gpusperrank: int = 1

    # FIXME: the plan here is to loop over all of the relevant combinations of gpus and ranks.
    # If gpusperrank is specified, then we can still loop over maxnodes.
    # If nranks is specified, then we can still loop over gpuspernode.
    
    def __call__(self, generator):
        self.generator = generator
        return self

    # Determine if val is a perfect power with exponent dim.
    def is_pow(val, dim):
        if dim == 0:
            return True
        if dim == 1:
            return True
        if dim == 2:
            return sympy.ntheory.primetest.is_square(val)
        cutoff = math.ceil(pow(val, 1 / dim))
        for idx in range(2, cutoff + 1):
            if idx ** dim == val:
                return True
        return False
                
    def generate_problems(self):
        import sympy

        hybrid = [False]
        if self.gpuspernode > 1 and self.maxnodes > 1:
            print("We have two different scaling experiments")
            # So we loop over the two possibilities, one is hybrid, one is traditional
            hybrid.append(True)
            
        gpudivs = sympy.divisors(self.gpuspernode) if self.gpuspernode > 0 else [self.gpusperrank]
        rankdivs = sympy.divisors(self.maxnodes) if self.maxnodes > 0 else [self.nranks]

        # NB: for weak scaling, we need constant data per GPU, which means that, for example, 3D
        # problems will use nubmers of GPUs that are cubes.  This implies some relationship between
        # gpus per node and max nodes.  For example, with 6 gpus per node, we could have max nodes
        # equal to 36, so then we get 1 gpu, then 36*6=216 gpus.  Powers-of-two are, as usual, much
        # nicer to deal with.
        
        for ishybrid in hybrid:
            gpus_ranks = []
            if not ishybrid:
                # Traditional parallelism, ie one GPU per rank.
                for ngpu in gpudivs:
                    gpus_ranks.append([1, ngpu])
                for nranks in rankdivs[1:]:
                    gpus_ranks.append([1, self.gpuspernode * nranks])
            else:
                # Hybrid parallelism, ie more than one GPU per rank.
                for ngpu in gpudivs:
                    gpus_ranks.append([ngpu, 1])
                for nranks in rankdivs[1:]:
                    gpus_ranks.append([self.gpuspernode, nranks])
            print(gpus_ranks)
                    
            for problem in self.generator.generate_problems():
                for gr in gpus_ranks:
                    ngpus = gr[0] * gr[1]
                    if problem.meta.get('scaling') == 'weak':
                        if not is_pow(ngpus, len(problem.length)):
                            continue

                    problem.gpusperrank = gr[0]
                    if problem.gpusperrank > 1:
                        problem.ingrid = [1] * len(problem.length)
                        problem.ingrid[0] = problem.gpusperrank
                        problem.outgrid = [1] * len(problem.length)
                        problem.outgrid[len(problem.length) - 1] = problem.gpusperrank

                    problem.nranks = gr[1]
                    if problem.nranks > 1:
                        problem.imgrid = [1] * len(problem.length)
                        problem.imgrid[0] = problem.nranks
                        problem.omgrid = [1] * len(problem.length)
                        problem.omgrid[len(problem.length) - 1] = problem.nranks

                    if ishybrid:
                        problem.tag += "_hybrid"
                        # FIXME: check that this actually creates a different file.
                        
                    if len(problem.length) in self.dimension \
                       and problem.direction in self.direction \
                       and problem.inplace in self.inplace \
                       and problem.real in self.real \
                       and problem.precision in self.precision:
                        yield problem


@dataclass
class RadixProblemGenerator:
    direction: List[int] = field(default_factory=lambda: [-1, 1])
    inplace: List[bool] = field(default_factory=lambda: [True, False])
    real: List[bool] = field(default_factory=lambda: [True, False])
    precision: List[str] = field(default_factory=lambda: ["single", "double"])
    dimension: int = 1
    xmin: int = 2
    xmax: int = 1024
    ymin: int = 2
    ymax: int = 1024
    zmin: int = 2
    zmax: int = 1024
    radix: int = 2
    nbatch: int = 1

    def generate_problems(self):
        for direction, precision, real, inplace in itertools.product(
                self.direction, self.precision, self.real, self.inplace):
            xval, yval, zval = self.xmin, self.ymin, self.zmin
            while xval <= self.xmax and yval <= self.ymax and zval <= self.zmax:
                length = [xval]
                if self.dimension > 1:
                    length.append(yval)
                if self.dimension > 2:
                    length.append(zval)

                yield Problem(length,
                              nbatch=self.nbatch,
                              direction=direction,
                              inplace=inplace,
                              real=real,
                              precision=precision,
                              tag=mktag('radix' + str(self.radix),
                                        self.dimension, precision, direction,
                                        inplace, real))

                xval *= self.radix
                if self.dimension > 1:
                    yval *= self.radix
                if self.dimension > 2:
                    zval *= self.radix


@dataclass
class FileProblemGenerator:
    problem_file: str

    inplace: List[bool] = field(default_factory=lambda: [False])
    real: List[bool] = field(default_factory=lambda: [False])
    precision: List[str] = field(default_factory=lambda: ["float"])

    def __post_init__(self):
        self.table = []
        with open(self.problem_file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.isspace():
                    continue
                nbatch = 1
                lengthBatch = line.replace(' ', '').split(',nbatch=')
                if len(lengthBatch) > 1:
                    nbatch = int(lengthBatch[1])
                line = lengthBatch[0]
                length = [int(x) for x in line.split(',')]
                self.table.append([length, nbatch])
        print(self.table)

    def generate_problems(self):
        for length, nbatch in self.table:
            for precision, real, inplace in itertools.product(
                    self.precision, self.real, self.inplace):
                ingrid = [1] * len(length)
                outgrid = [1] * len(length)
                imgrid = [1] * len(length)
                omgrid = [1] * len(length)
                yield Problem(length,
                              nbatch=nbatch,
                              direction=-1,
                              inplace=inplace,
                              real=real,
                              precision=precision,
                              ingrid=ingrid,
                              outgrid=outgrid,
                              imgrid=imgrid,
                              omgrid=omgrid)


@dataclass
class TableProblemGenerator:
    table: None
    inplace: List[bool] = field(default_factory=lambda: [False])
    real: List[bool] = field(default_factory=lambda: [False])
    precision: List[str] = field(default_factory=lambda: ["float"])

    def generate_problems(self):
        for length, nbatch in self.table:
            for precision, real, inplace in itertools.product(
                    self.precision, self.real, self.inplace):
                ingrid = [1] * len(length)
                outgrid = [1] * len(length)
                imgrid = [1] * len(length)
                omgrid = [1] * len(length)
                yield Problem(length,
                              nbatch=nbatch,
                              direction=-1,
                              inplace=inplace,
                              real=real,
                              precision=precision,
                              ingrid=ingrid,
                              outgrid=outgrid,
                              imgrid=imgrid,
                              omgrid=omgrid)


def suite_file(base):
    """Try to find a suite file using 'base' as the name part of the path."""
    p = path(base)
    if p.exists():
        return p
    p = p.with_suffix('.py')
    if p.exists():
        return p
    p = top / p.name
    if p.exists():
        return p
    raise ValueError(f"Unable to locate suite file '{base}'.")


def load_suite(suite, fname=None):
    """Load performance suite from suites.py."""

    tdef = top / 'suites.py'
    if fname is not None:
        tdef = suite_file(fname)
    logging.info(f'loading suites from {tdef}')
    code = compile(tdef.read_text(), str(tdef), 'exec')
    ns = {}
    exec(code, ns)
    return ns[suite]


@dataclass
class SuiteProblemGenerator:
    suite_names: List[str]
    suites: Mapping[str, Generator[Problem, None,
                                   None]] = field(default_factory=dict)

    def __post_init__(self):
        for name in self.suite_names:
            fname = None
            if ':' in name:
                fname, name = name.split(':')
            self.suites[name] = load_suite(name, fname)

    def generate_problems(self):
        for g in self.suites.values():
            yield from g()
