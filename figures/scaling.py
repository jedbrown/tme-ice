#!/usr/bin/python

from optparse import OptionParser
from pylab import *
import pprint
import re
from collections import namedtuple
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

#SNES   = namedtuple('SNES', 'level indent reason res')

KSP    = namedtuple('KSP', 'reason res')
Solve  = namedtuple('Solve', 'level residuals')
Sample = namedtuple('Sample', 'fname solves')
SNESIt = namedtuple('SNESIt', 'indent res ksp')
Event  = namedtuple('Event', 'name count time flops Mflops')
Stage  = namedtuple('Stage', 'name events')
Run    = namedtuple('Run', 'levels solves stages options')

const = lambda x: lambda _: x
instanceof = lambda t: lambda x: isinstance(x,t)
residuals = lambda pred: lambda solve: [x.res for x in solve.residuals if pred(x)]
def onlevel(lev):
    def go(res):
        its = []
        i = 0
        while i < len(res) and (not isinstance(res[i],SNES) or res[i].level != lev): i+=1 # skip initial segment
        while i < len(res) and (not isinstance(res[i],SNES) or res[i].level == lev): # grab relevant segment
            its.append(res[i])
            i+=1
        return its
    return go

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

from funcparserlib.lexer import Spec, make_tokenizer
def tokenize(str):
    'str -> Sequence(Token)'
    MSpec = lambda t,r: Spec(t,r,re.MULTILINE)
    specs = [
        MSpec('level', r'^Level \d+.*$'),
        MSpec('snes_monitor', r'^\s+\d+ SNES Function norm.*$'),
        MSpec('snes_converged', r'^\s*Nonlinear solve converged due to.*$'),
        MSpec('ksp_monitor', r'^\s+\d+ KSP Residual norm.*$'),
        MSpec('ksp_converged', r'^\s*Linear solve converged due to.*$'),
        MSpec('event', r'^\S{1,16}\s+\d+ \d\.\d \d\.\d{4}e[-+]\d\d \d\.\d \d\.\d\de[-+]\d\d \d\.\d (\d\.\de[-+]\d\d ){3}.*$'),
        MSpec('stage', r'^--- Event Stage \d+: .*$'),
        MSpec('memory_usage', r'^Memory usage is given in bytes:'),
        MSpec('option_table_begin', r'^#PETSc Option Table entries:$'),
        MSpec('option_table_entry', r'^-\w+(\s+\w+)?$'),
        MSpec('option_table_end', r'^#End of? PETSc Option Table entries$'),
        Spec('nl', r'[\r\n]+'),
        MSpec('other', r'^.*$') # Catches all lines that we don't understand
        ]
    ignored = 'nl other'.split()
    t = make_tokenizer(specs)
    print t(str)
    return [x for x in t(str) if x.type not in ignored]

from funcparserlib.contrib.common import n,op,op_,sometok
from funcparserlib.parser import maybe, many, oneplus, finished, skip, forward_decl, SyntaxError
def parse(seq):
    'Sequence(Token) -> object'
    LogSummary = namedtuple('LogSummary', 'stages options')
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
    def mkStage(s):
        name = re.match(r'^--- Event Stage \d+: (.*)', s[0]).groups()[0]
        events = s[1]
        return Stage(name, events)
    def mkOption(s):
        return re.match(r'^(-\w+)(?:\s+(.+))?$',s).groups()
    def mkLogSummary(s):
        return LogSummary(stages=s[0], options=s[1])
    def mkRun(s):
        levels = s[0]
        solves = s[1]
        log    = s[2]
        for x in solves:
            x.level = levels[-1-x.indent]
        return Run(levels, solves, log.stages, log.options)

    level = sometok('level') >> mkLevel
    kspit = sometok('ksp_monitor')   >> mkKSPIt
    ksp_converged = sometok('ksp_converged') >> (lambda s: s.strip().split()[5])
    ksp   = many(kspit) + maybe(ksp_converged) >> mkKSP
    snesit = sometok('snes_monitor') + maybe(ksp) >> mkSNESIt
    snes_converged = sometok('snes_converged') >> (lambda s: s.strip().split()[5])
    snes  = oneplus(snesit) + snes_converged >> mkSNES
    event = sometok('event') >> mkEvent
    stage = sometok('stage') + many(event) >> mkStage
    memory_usage = sometok('memory_usage') + many(sometok('stage')) # No plans for memory usage
    option_table_entry = sometok('option_table_entry') >> mkOption
    option_table = skip(sometok('option_table_begin')) + many(option_table_entry) + skip(sometok('option_table_end')) >> dict
    log_summary = many(stage) + skip(memory_usage) + option_table >> mkLogSummary
    petsc_log = many(level) + many(snes) + log_summary + skip(finished) >> mkRun
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
    subplots_adjust(left=0.06,right=0.975,bottom=0.08,top=0.94)

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
    marker = list('osv^<>*D').__iter__()
    offset = 0
    for s in solves:
        res = array([[i,x.res] for (i,x) in enumerate(s.res)])
        name = 'Level %d (%d)' % (s.level.level, s.level.count)
        name = s.name()
        semilogy(offset+res[:,0],res[:,1],'-'+marker.next(),label=name)
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
    show()

def plot_poisson(format):
    if format:
        rcParams.update({'backend': format})
    plot_loglog(libmesh_poisson,1,4,'r','o',"Libmesh $Q_2$",600)
    plot_loglog(dohp_poisson_q3,1,4,'g','s',"Dohp $Q_3$",1000)
    plot_loglog(dohp_poisson_q5,1,4,'b','^',"Dohp $Q_5$",1200)
    plot_loglog(dohp_poisson_q7,1,4,'k','*',"Dohp $Q_7$",1400)
    #plot_loglog(dummy_direct_mumps,1,3,'k','o',"MUMPS direct solve",1600)
    #title('Scaling of 3D Poisson solvers with algebraic multigrid preconditioning')
    xlim(2e4,3e6)
    ylim(1,4e2)
    ylabel('Linear solve time (seconds)')
    xlabel('Degrees of freedom')
    legend(loc='upper left',numpoints=1)
    if format:
        savefig('timing.'+format)
    else:
        show()

def main1 ():
    parser = OptionParser()
    parser.add_option('-f', '--format', help='|png|eps', dest='format', type='string', default=None)
    parser.add_option('-m', '--mode', help='talk|poster', dest='mode', type='string', default='talk')
    parser.add_option('-t', '--type', help='poisson|stokes', dest='type', type='string', default='poisson')
    opts, args = parser.parse_args()
    {'talk' : set_sizes_talk, 'poster' : set_sizes_poster, 'paper' : set_sizes_paper}[opts.mode]()
    print 'Plotting ', opts.type, 'using format', opts.format
    {'poisson': plot_poisson, 'stokes' : plot_stokes}[opts.type](opts.format)

def main2():
    asm = read_file('z.10km.asm20.seq.rtol-2.log')
    toks = tokenize(asm)
    p = parse(toks)
    plot_snes_convergence(p.solves[1:],sequence=True,withksp=True)

if __name__ == "__main__":
    main2()
