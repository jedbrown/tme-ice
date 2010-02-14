#!/usr/bin/python

from optparse import OptionParser
from pylab import *
import pprint, re, itertools
from collections import namedtuple
from funcparserlib.lexer import Spec, make_tokenizer
from funcparserlib.contrib.common import sometok,unarg
from funcparserlib.parser import maybe, many, oneplus, finished, skip, forward_decl, SyntaxError

import pdb

class SNES(object):
    def __init__(self, indent, reason, res, level=None):
        self.indent = indent
        self.reason = reason
        self.res = res
        self.level = None
        self._name = None
    def name(self, name=None):
        if name:
            self._name = name
        elif self.level:
            return 'Level %d (%d nodes)'%(self.level.level, self.level.count)
        else:
            return 'Unknown'
    def __repr__(self):
        return 'SNES(indent=%r, reason=%r, res=%r, level=%r)' % (self.indent, self.reason, self.res, self.level)

def Dict(**args): return args
KSP    = namedtuple('KSP', 'reason res')
SNESIt = namedtuple('SNESIt', 'indent res ksp')
Event  = namedtuple('Event', 'name count time flops Mflops')
Stage  = namedtuple('Stage', 'name events')

class Run(object):
    def __init__(self, levels, solves, exename, petsc_arch, hostname, np, stages, options, **args):
        self.levels     = levels
        self.solves     = solves
        self.exename    = exename
        self.petsc_arch = petsc_arch
        self.hostname   = hostname
        self.np         = np
        self.stages     = stages
        self.options    = options
        self.__dict__.update(args)
    def __repr__(self):
        return 'Run(%s)' % ', '.join('%s=%r' % (k,v) for (k,v) in self.__dict__.items())

def span(pred):
    def go(lst):
        for i in xrange(len(lst)):
            if not pred(lst[i]):
                return lst[:i], lst[i:]
        return (lst,[])
    return go
def groupBy(f,lst):
    'Group input, f(elem) == None is agnostic'
    chunks = []
    next = lst
    while next:
        cur = f(next[0])
        a, next = span(lambda x: (f(x) is None) or (f(x) == cur))(next)
        chunks.append(a)
    return chunks

class Level(object):
    def __init__(self,level,Lx,Ly,Lz,M,N,P,count,hx,hy,hz):
        scope = locals() # Bind the free variables in this scope
        def mkprops(cvt,props):
            for p in props.split():
                setattr(self,p,cvt(scope[p]))
        mkprops(float,'Lx Ly Lz hx hy hz')
        mkprops(int,'level M N P count')
        # Input validation
        assert(self.M*self.N*self.P == self.count)
        fuzzy_equals = lambda a,b: abs(a-b) / (abs(a)+abs(b)) < 1e-3
        for (l,m,h) in map(lambda s: s.split(),['Lx M hx','Ly N hy']):
            assert(fuzzy_equals(getattr(self,l)/getattr(self,m), getattr(self,h)))
        assert(fuzzy_equals(self.Lz/(self.P-1), self.hz))
    def __repr__(self):
        return ('Level(level=%r, Lx=%r, Ly=%r, Lz=%r, M=%r, N=%r, P=%r, count=%r, hx=%r, hy=%r, hz=%r)'
                % tuple(getattr(self,p) for p in 'level Lx Ly Lz M N P count hx hy hz'.split()))

def tokenize(str):
    'str -> Sequence(Token)'
    MSpec = lambda t,r: Spec(t,r,re.MULTILINE)
    specs = [
        MSpec('level', r'^Level \d+.*$'),
        MSpec('snes_monitor', r'^\s+\d+ SNES Function norm.*$'),
        MSpec('snes_converged', r'^\s*Nonlinear solve converged due to \w+$'),
        MSpec('snes_diverged', r'^\s*Nonlinear solve did not converge due to \w+$'),
        MSpec('ksp_monitor', r'^\s+\d+ KSP Residual norm.*$'),
        MSpec('ksp_converged', r'^\s*Linear solve converged due to \w+$'),
        MSpec('ksp_diverged', r'^\s*Linear solve did not converge due to \w+$'),
        MSpec('event', r'^\S{1,16}\s+\d+ \d\.\d \d\.\d{4}e[-+]\d\d \d\.\d \d\.\d\de[-+]\d\d \d\.\d (\d\.\de[-+]\d\d ){3}.*$'),
        MSpec('stage', r'^--- Event Stage \d+: .*$'),
        MSpec('memory_usage', r'^Memory usage is given in bytes:'),
        MSpec('summary_begin', r'^---------------------------------------------- PETSc Performance Summary: ----------------------------------------------$'),
        MSpec('hostline', r'^\S+ on a \S+ named \S+ with \d+ processors?, by .*$'),
        MSpec('option_table_begin', r'^#PETSc Option Table entries:$'),
        MSpec('option_table_entry', r'^-\w+(\s+\w+)?$'),
        MSpec('option_table_end', r'^#End of? PETSc Option Table entries$'),
        Spec('nl', r'[\r\n]+'),
        MSpec('other', r'^.*$') # Catches all lines that we don't understand
        ]
    ignored = 'nl other'.split()
    t = make_tokenizer(specs)
    return [x for x in t(str) if x.type not in ignored]

def parse(seq):
    'Sequence(Token) -> object'
    Host = namedtuple('Host', 'exename arch host np')
    LogSummary = namedtuple('LogSummary', 'host stages options')
    def mkLevel(s):
        rfloat = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
        rint   = r'[-+]?\d+'
        capture = lambda m: '('+m+')'
        within_space = lambda m: r'\s*'+m+r'\s*'
        cfloat, cint = map(lambda m: within_space(capture(m)),[rfloat,rint])
        x = within_space('x')
        m = re.match('Level'+cint+r'domain size \(m\)'+cfloat+x+cfloat+x+cfloat
                     +', num elements'+cint+x+cint+x+cint+r'\('+cint
                     +r'\), size \(m\)'+cfloat+x+cfloat+x+cfloat, s)
        return Level(*m.groups())
    def mkSNESIt(s):
        resline, ksp = s[0], s[1]
        res = float(resline.strip().split()[4])
        indent = len(re.match(r'^( *)(?:  | \d|\d\d)\d', resline).groups()[0])/2
        return SNESIt(indent,res,ksp)
    def mkKSPIt(s):
        return float(s.strip().split()[4])
    def mkKSP(s):
        return KSP(reason=('UNKNOWN' if len(s)==1 else s[1]), res=s[0])
    def mkSNES(s):
        res = s[0]
        reason = s[1]
        indent = res[0].indent
        for it in res[1:]:
            if it.indent != indent:
                raise RuntimeError('SNES monitors changed levels, perhaps -snes_converged_reason is missing:\n\tstarted with: %s\n\tunexpected: %s' %(res[0],it))
        return SNES(level=None, indent=indent, reason=s[1], res=s[0])
    def mkEvent(s):
        s = s.split()
        return Event(name=s[0], count=int(s[1]), time=float(s[3]), flops=float(s[5]), Mflops=float(s[-1]))
    def mkStage(stageheader, events):
        name = re.match(r'^--- Event Stage \d+: (.*)', stageheader).groups()[0]
        return Stage(name, events)
    def mkOption(s):
        return re.match(r'^(-\w+)(?:\s+(.+))?$',s).groups()
    def mkRun(levels, solves, log):
        for x in solves:
            x.level = levels[-1-x.indent]
        h = log.host
        return Run(levels, solves, h.exename, h.arch, h.host, h.np, log.stages, log.options)
    def mkHost(s):
        (exename, arch, host, np) = re.match(r'^(\S+) on a (\S+) named (\S+) with (\d+) processors?, by .*$', s).groups()
        return Host(exename, arch, host, int(np))

    level = sometok('level') >> mkLevel
    kspit = sometok('ksp_monitor')   >> mkKSPIt
    ksp_converged = sometok('ksp_converged') >> (lambda s: s.strip().split()[5])
    ksp_diverged = sometok('ksp_diverged') >> (lambda s: s.strip().split()[7])
    ksp   = many(kspit) + maybe(ksp_converged | ksp_diverged) >> mkKSP
    snesit = sometok('snes_monitor') + maybe(ksp) >> mkSNESIt
    snes_converged = sometok('snes_converged') >> (lambda s: s.strip().split()[5])
    snes_diverged = sometok('snes_diverged') >> (lambda s: s.strip().split()[7])
    snes  = oneplus(snesit) + (snes_converged | snes_diverged) >> mkSNES
    event = sometok('event') >> mkEvent
    stage = sometok('stage') + many(event) >> unarg(mkStage)
    memory_usage = sometok('memory_usage') + many(sometok('stage')) # No plans for memory usage
    option_table_entry = sometok('option_table_entry') >> mkOption
    option_table = skip(sometok('option_table_begin')) + many(option_table_entry) + skip(sometok('option_table_end')) >> dict
    host = sometok('hostline') >> mkHost
    log_summary = skip(sometok('summary_begin')) + host + many(stage) + skip(memory_usage) + option_table >> unarg(LogSummary)
    petsc_log = many(level) + many(snes) + log_summary + skip(finished) >> unarg(mkRun)
    return petsc_log.parse(seq)

def read_file(fname):
    with open(fname) as f:
        s = f.read()
    return s

def plot_loglog(arr,xcol,ycol,color,marker,name,loc):
    x = arr[:,xcol]
    y = arr[:,ycol]
    rm,rb = polyfit(log(x),log(y),1)
    plot(x,exp(log(x)*rm+rb),color,linewidth=1)
    loglog(x,y,color[0]+marker,label='%s slope=%5.3f'%(name,rm),markersize=12)
    #text(3.5e6,loc,'slope=%5.3f'%rm)
    #text(3.5e6,loc,'slope=%5.3f'%rm)

def set_sizes_talk():
    golden = (sqrt(5)-1)/2
    fig_width = 15;
    fig_size = (fig_width,fig_width*golden)
    rcParams.update({'axes.titlesize': 18,
                     'axes.labelsize': 18,
                     'text.fontsize': 24,
                     'legend.fontsize': 16,
                     #'legend.markerscale' : 8,
                     'xtick.labelsize': 18,
                     'ytick.labelsize': 18,
                     'text.usetex': True,
                     'figure.figsize': fig_size})
    subplots_adjust(left=0.08,right=0.975,bottom=0.08,top=0.94)

def set_sizes_poster():
    golden = (sqrt(5)-1)/2
    fig_width = 10;
    fig_size = (fig_width,fig_width*golden)
    rcParams.update({'axes.titlesize': 18,
                     'axes.labelsize': 18,
                     'text.fontsize': 24,
                     'lines.linewidth': 3,
                     'legend.fontsize': 16,
                     #'legend.markerscale' : 8,
                     'xtick.labelsize': 18,
                     'ytick.labelsize': 18,
                     'text.usetex': True,
                     'figure.figsize': fig_size})
    subplots_adjust(left=0.09,right=0.975,bottom=0.11,top=0.93)

def set_sizes_paper():
    golden = (sqrt(5)-1)/2
    fig_width = 8;
    fig_size = (fig_width,fig_width*golden)
    rcParams.update({'axes.titlesize': 16,
                     'axes.labelsize': 16,
                     'text.fontsize': 16,
                     'lines.linewidth': 3,
                     'legend.fontsize': 16,
                     #'legend.markerscale' : 8,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'text.usetex': True,
                     'figure.figsize': fig_size})
    subplots_adjust(left=0.09,right=0.975,bottom=0.11,top=0.93)

def plot_snes_convergence(solves,sequence=False,withksp=False):
    marker = itertools.cycle(list('osv^<>*D'))
    offset = 0
    for s in solves:
        res = array([[i,x.res] for (i,x) in enumerate(s.res)])
        name = s.name()
        semilogy(offset+res[:,0],res[:,1],'-'+next(marker),label=name)
        if withksp:
            for (k,ksp) in enumerate([r.ksp for r in s.res]):
                if not ksp.res:
                    continue # Skip when there are no linear iterations (SNES converged)
                kres = array(list(enumerate(ksp.res)))
                x = offset+k+kres[:,0]/(kres.shape[0]-1)
                semilogy(x,kres[:,1],'k:x',label=None)
        if sequence: offset += res.shape[0]-1
    ylabel('Nonlinear residual')
    xlabel('Newton iteration')
    legend()

def plot_snes(opts, logfiles):
    '''Plots the nonlinear convergence for a single file.'''
    if len(logfiles) != 1:
        raise RuntimeError('Must supply exactly one file for SNES plotting')
    run = parse(tokenize(read_file(logfiles[0])))
    plot_snes_convergence(run.solves[1:], sequence=True, withksp=True)

def segment(logfiles):
    '''turn a flat semicolon-delimited list
           ['a', 'b', ';', 'c', 'd', 'e']
       into a generator of lists
           [['a', 'b'], ['c', 'd', 'e']]'''
    return filter(lambda x: x != [';'], groupBy(lambda x: x==';', logfiles))

def plot_algorithmic(opts, logfiles):
    '''Plots algorithmic scalability for several file series.  File
    series are separated by ';' (must be escaped from the shell).  Each
    file series is expected to be a list of log files with common
    algorithm and increasing problem size.  The result is a log-linear
    plot of iteration count for each series.
    '''
    marker = itertools.cycle(list('osv^<>*D'))
    plotter = semilogx
    for logs in segment(logfiles):
        series = [parse(tokenize(read_file(fname))) for fname in logs]
        solves = [s.solves[-1] for s in series]
        its = array([(s.level.count, mean([len(r.ksp.res) for r in s.res][:-1])) for s in solves])
        plotter(its[:,0],its[:,1],'-'+next(marker),label=logs[0])
    ylabel('Average Krylov iterations per Newton at finest level')
    xlabel('Number of degrees of freedom')
    legend(loc='upper left')

def plot_strong(opts, logfiles):
    ''

def main():
    parser = OptionParser()
    parser.add_option('-f', '--format', help='|png|eps', dest='format', type='string', default=None)
    parser.add_option('-m', '--mode', help='talk|poster|paper', dest='mode', type='string', default='talk')
    parser.add_option('-t', '--type', help='snes|algorithmic|weak|strong', dest='type', type='string', default='snes')
    parser.add_option('-o', '--output', help='Output filename', dest='output', default=None)
    opts, logfiles = parser.parse_args()
    {'talk' : set_sizes_talk, 'poster' : set_sizes_poster, 'paper' : set_sizes_paper}[opts.mode]()
    print 'Plotting %s using format %s from files: %s' % (opts.type, opts.format, ' '.join(logfiles))
    if format: rcParams.update({'backend': format})
    {'snes': plot_snes, 'algorithmic' : plot_algorithmic}[opts.type](opts, logfiles)
    if opts.output:
        savefig(opts.output)
    else:
        show()

if __name__ == "__main__":
    main()
