"""Vector PJSS architecture: data, costly joint fits, cheap screening, feedback."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif', 'pdf.fonttype':42,
                     'svg.fonttype':'none', 'font.size':8})
fig = plt.figure(figsize=(6.9, 1.85))
ax = fig.add_axes([0,0,1,1])
ax.set_xlim(0,497);ax.set_ylim(0,137);ax.axis('off')
INK = '#222222'
BLUE = '#1763a6'
ORANGE = '#b75d21'
GRAY = '#69747c'

boxes = [
    (1, 79, 'INPUT', 'Noisy samples\nTime step + terms', '#f2f4f5', GRAY),
    (94, 86, 'INITIAL FIT', 'Starting equations\nDenoise + fit', '#fff1e6', ORANGE),
    (194, 104, 'PROFILE & SCREEN', 'Adjust trajectory locally\nRank term combinations', '#e8f1f8', BLUE),
    (312, 96, 'REFIT & SELECT', 'Top 10 per round\nFit + complexity cost', '#fff1e6', ORANGE),
    (422, 74, 'FINAL MODEL', 'Refit all samples\nSelected equations', '#f2f4f5', GRAY),
]
y, h = 77, 39
for x,w,title,body,fill,edge in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=3',
                               facecolor=fill,edgecolor=edge,linewidth=.8))
    ax.text(x+w/2,y+h-9,title,ha='center',va='center',fontsize=7.7,
            fontweight='bold',color=edge)
    ax.text(x+w/2,y+13.5,body,ha='center',va='center',fontsize=7.25,
            linespacing=1.22,color=INK)

for (x,w,*_), (xn,*_) in zip(boxes[:-1],boxes[1:]):
    ax.add_patch(FancyArrowPatch((x+w+1,y+h/2),(xn-2,y+h/2),
                 arrowstyle='-|>',mutation_scale=7.5,linewidth=.9,color=GRAY))
ax.text(246,126,'CHEAP SCREEN',ha='center',va='center',
        fontsize=7.3,fontweight='bold',color=BLUE)
ax.text(137,126,'NONLINEAR FIT',ha='center',va='center',fontsize=7.3,color=ORANGE)
ax.text(360,126,'NONLINEAR FITS',ha='center',va='center',fontsize=7.3,color=ORANGE)
ax.text(415,106.5,'stop',ha='center',va='center',fontsize=6.5,color=GRAY)

# The selected state/coefficient fit updates the next local expansion, not the input data.
verts=[(360,77),(360,63),(246,63),(246,76)]
ax.add_patch(FancyArrowPatch(path=MPath(verts,[MPath.MOVETO,MPath.LINETO,MPath.LINETO,MPath.LINETO]),
                           arrowstyle='-|>',mutation_scale=8,linewidth=.9,color=BLUE))
ax.text(249,51,'Use the best fit',ha='center',va='center',fontsize=7.3,color=INK)
ax.text(249,39,'At most two rounds',ha='center',va='center',fontsize=7.3,color=INK)

# Actual saved evidence illustrates the input/output contract, without refitting.
with np.load(HERE/'evidence/lorenz.npz') as data:
    observed=data['observed'][-200:,0]
    xi=data['xi']
ax.text(77,57,'Recorded input: Lorenz x',ha='center',fontsize=7.2,color=INK)
trace=fig.add_axes([15/497,12/137,130/497,34/137])
trace.scatter(np.arange(1800,2000),observed,s=1.1,color=GRAY,alpha=.65)
trace.set_xlim(1800,1999)
trace.set_xticks([1800,1999]);trace.set_xticklabels(['1800','1999'],fontsize=6)
trace.set_yticks([-10,10]);trace.tick_params(axis='y',labelsize=6,pad=1)
trace.tick_params(axis='x',pad=1,length=2)
trace.spines[['top','right']].set_visible(False)
for spine in trace.spines.values():spine.set_linewidth(.45);spine.set_color(GRAY)
ax.text(80,1,'Training sample',ha='center',va='bottom',fontsize=6,color=INK)
ax.add_patch(FancyArrowPatch((40,60),(40,76),arrowstyle='-|>',
                           mutation_scale=7,linewidth=.8,color=GRAY))
ax.add_patch(FancyArrowPatch((459,77),(459,61),arrowstyle='-|>',
                           mutation_scale=7,linewidth=.8,color=GRAY))
ax.text(412,52,'Discovered Lorenz equations',ha='center',fontsize=7.2,color=INK)
# Use the same polynomial ordering as the cached discovery coefficients.
from eqdisc.core import NAMES
index={name:i for i,name in enumerate(NAMES)}
terms=[('x',0),('y',0),('x',1),('y',1),('x*z',1),('z',2),('x*y',2)]
a,b,c,d,e,f,g=[xi[index[name],j] for name,j in terms]
equations=[rf'$\dot x={a:.2f}x{b:+.2f}y$',
           rf'$\dot y={c:.2f}x{d:+.2f}y{e:+.2f}xz$',
           rf'$\dot z={f:.2f}z{g:+.2f}xy$']
for line,yy in zip(equations,[38,24,10]):
    ax.text(412,yy,line,ha='center',va='center',fontsize=7.4,color=INK)

for ext in ('pdf','svg','png'):
    fig.savefig(FIG/f'architecture.{ext}',dpi=220,facecolor='white',
                metadata={'Creator':'PJSS figure generator'} if ext=='pdf' else None)
plt.close(fig)
print('Architecture saved as PDF, SVG and PNG.')
