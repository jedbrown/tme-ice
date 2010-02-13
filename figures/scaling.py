#!/usr/bin/python

from optparse import OptionParser
from pylab import *
import pprint
import re
from collections import namedtuple
import pdb

class Nonlinear(object):
    def __init__(self,initial,final,ksp):
        self.initial = initial
        self.final = final
        self.ksp = ksp
    def before(self, next):
        return self.final == next.initial

KSP    = namedtuple('KSP', 'step res')
SNES   = namedtuple('SNES', 'level step res')
Solve  = namedtuple('Solve', 'level residuals')
Sample = namedtuple('Sample', 'fname solves')

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
        return ('Level(level=%d,Lx=%g,Ly=%g,Lz=%g,M=%d,N=%d,P=%d,count=%d,hx=%g,hy=%g,hz=%g)'
                % tuple(getattr(self,p) for p in 'level Lx Ly Lz M N P count hx hy hz'.split()))

def parse_file(fname):
    def parse_levels(lines):
        levels = []
        for line in lines:
            # Fucking Python doesn't have fucking sscanf!
            rfloat = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
            rint   = r'[-+]?\d+'
            capture = lambda m: '('+m+')'
            within_space = lambda m: r'\s*'+m+r'\s*'
            cfloat, cint = map(lambda m: within_space(capture(m)),[rfloat,rint])
            x = within_space('x')
            m = re.match('Level'+cint+r'domain size \(m\)'+cfloat+x+cfloat+x+cfloat
                         +', num elements'+cint+x+cint+x+cint+r'\('+cint
                         +r'\), size \(m\)'+cfloat+x+cfloat+x+cfloat, line)
            if not m: return levels
            levels.append(Level(*m.groups()))
        raise RuntimeError('File contains only Level statements?')
    def pred(line): return re.match('\s+\d+ (KSP Residual|SNES Function)',line)
    def parse_line(line):
        a = line.strip().split()
        it,t,res = [int(a[0]),a[1],float(a[4])]
        clev = len(levels) - line.find(a[0])/2
        return {'KSP': KSP(it,res),
                'SNES': SNES(clev,it,res)}[t]
    with open(fname) as f:
        lines = f.readlines()
        levels = dict([(l.level,l) for l in parse_levels(lines)])
        iterations = [parse_line(line) for line in lines if pred(line)]
    solves = groupBy(lambda x: None if isinstance(x,KSP) else x.level, iterations)
    solves = [Solve(levels[s[0].level],s) for s in solves]
    pdb.set_trace()
    return Sample(fname,solves)

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
    b = parse_file('z.log')
    print b
    pdb.set_trace()

if __name__ == "__main__":
    main2()

# px=poisson[:,1]
# py=poisson[:,3]
# sx=stokes[1:,1]
# sy=stokes[1:,6]
# loglog(px,py,'rs',label='Poisson')
# loglog(sx,sy,'ob',label='Stokes')
# plot_fit(px,py)
# rpm,rpb=polyfit(log(px),log(py),1)
# plot(px,exp(log(px)*rpm+rpb),'k',linewidth=2)
# rsm,rsb=polyfit(log(sx),log(sy),1)
# plot(sx,exp(log(sx)*rsm+rsb),'k',linewidth=2)
# text(3.5e6,200,'slope=1.273')
# text(3.5e6,400,'slope=1.005')
# show()
