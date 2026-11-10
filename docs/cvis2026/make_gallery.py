"""Broader attractor gallery from three fresh examples and six saved benchmark fits.

Gallery choice is illustrative, explicitly includes known support failures, and
does not estimate any population recovery rate. Historical fits are never rerun.
"""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from eqdisc.core import library, simulate
from make_illustrations import save, BLUE, INK, ORANGE, EVIDENCE

NAMES = ['Lorenz', 'Rossler', 'Chen', 'Arneodo', 'Finance',
         'HindmarshRose', 'DequanLi', 'GuckenheimerHolmes', 'AtmosphericRegime']
protocol = {
    'purpose': 'Illustrate breadth and two known support failures, not estimate recovery rates',
    'systems_in_order': NAMES,
    'fresh': ['Lorenz', 'Rossler', 'Chen'],
    'saved_benchmark_selection': 'Lowest-index saved run per named system with n=2000 and noise=0.1',
    'selection_rationale': 'Six additional systems outside development benchmarks, including two known support-failure examples; choices are post-hoc',
    'historical_models': 'Saved frozen-model coefficients in results_dysts/test_final.json; not newly rerun fits',
    'historical_initial_state': 'A common clean reference-attractor point for each reference/learned pair; this is not forecast evaluation',
    'historical_integrations': {'solver':'DOP853', 'rtol':1e-8,'atol':1e-10,
        'samples':8000,'burn_samples':1000,'escape_bound':'20 times maximum absolute reference state per coordinate, with floor 1'},
    'axes': 'Reference limits shared by both overlaid paths; out-of-view learned samples masked and labeled',
    'failure_handling': 'Retain incomplete or escaped integrations and label them; never substitute another run'
}
gallery_dir = EVIDENCE / 'gallery'
gallery_dir.mkdir(exist_ok=True)
(gallery_dir/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
records = json.loads((ROOT/'results_dysts/test_final.json').read_text())['records']
meta = json.loads((ROOT/'data/dysts_test.json').read_text())['systems']
classes = json.loads((ROOT/'data/dysts_classes.json').read_text())['systems']


def historical(name):
    row = min((r for r in records if r['system']==name and r['n']==2000 and r['noise']==.1),key=lambda r:r['index'])
    assert row['status']=='ok' and meta[name].get('overlap') is None
    xi_true=np.asarray(meta[name]['xi']);xi=np.asarray(row['xi']);dt=meta[name]['dt']
    warm=simulate(xi_true,classes[name]['ic'],dt,3000)
    assert warm is not None
    x0=warm[-1]
    reference=simulate(xi_true,x0,dt,8000,burn=1000)
    assert reference is not None
    scale=np.maximum(np.max(np.abs(reference),axis=0),1.)
    def escape(t,x):return 20.-np.max(np.abs(x)/scale)
    escape.terminal=True;escape.direction=-1
    times=np.arange(9000)*dt
    sol=solve_ivp(lambda t,x:library(x[None])[0]@xi,(0,times[-1]),x0,
                  t_eval=times,method='DOP853',rtol=1e-8,atol=1e-10,events=escape)
    complete=bool(sol.success and sol.y.shape[1]==len(times))
    # An incomplete integration is retained before its escape, not replaced.
    learned=sol.y.T[1000:] if complete else sol.y.T
    info={'system':name,'source':'saved frozen dysts fit','index':row['index'],
          'noise':.1,'n_fit':2000,'dt':dt,'exact':bool(row['final']['exact']),
          'coef_err':row['final']['coef_err'],'new_system':True,
          'integration_complete':complete,'integrator_message':sol.message,
          'samples_before_stop':sol.y.shape[1]}
    return reference,learned,xi_true,xi,x0,info


panels=[]
for name in NAMES:
    if name in protocol['fresh']:
        d=dict(np.load(EVIDENCE/f'{name.lower()}.npz'))
        s=json.loads((EVIDENCE/f'{name.lower()}.json').read_text())
        ref,learned=d['attractor_true'],d['attractor_learned']
        info={'system':name,'source':'fresh v1.2 fixed illustration',
              'noise':.1,'n_fit':2000,'dt':s['dt'],'exact':bool(s['exact']),
              'coef_err':s['coef_err'],'new_system':False,'integration_complete':bool(len(learned)==8000)}
        xi_true,xi,x0=d['xi_true'],d['xi'],d['initial']
    else:
        cache=gallery_dir/f'{name}.npz';summary=gallery_dir/f'{name}.json'
        if cache.exists():
            d=dict(np.load(cache));info=json.loads(summary.read_text())
            ref,learned,xi_true,xi,x0=[d[k] for k in ['reference','learned','xi_true','xi','initial']]
        else:
            print('INTEGRATING SAVED FIT',name,flush=True)
            ref,learned,xi_true,xi,x0,info=historical(name)
            np.savez_compressed(cache,reference=ref,learned=learned,xi_true=xi_true,xi=xi,initial=x0)
            summary.write_text(json.dumps(info,indent=2),encoding='utf-8')
    panels.append((ref,learned,info))
    print(name,'exact',info['exact'],'complete',info['integration_complete'],flush=True)

fig=plt.figure(figsize=(6.75,2.25))
for i,(ref,learned,info) in enumerate(panels):
    # Five panels above four centred panels preserve all systems while leaving
    # space in the manuscript for interpretation of the plots.
    cx=.1+.2*i if i<5 else .2+.2*(i-5)
    bottom=.59 if i<5 else .11
    ax=fig.add_axes([cx-.0925,bottom,.185,.31],projection='3d')
    lo=ref.min(0);hi=ref.max(0);pad=.08*np.maximum(hi-lo,1e-6)
    lo-=pad;hi+=pad
    clipped=learned.copy()
    outside=np.any((clipped<lo)|(clipped>hi),axis=1)
    clipped[outside]=np.nan
    # Drawing paths separately prevents joining unrelated trajectory segments.
    ax.plot(*ref[::2].T,color='#85898d',lw=.32,alpha=.65,rasterized=True)
    ax.plot(*clipped[::2].T,color=BLUE,lw=.37,alpha=.82,rasterized=True)
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2])
    label='Exact support' if info['exact'] else 'Support mismatch'
    if not info['integration_complete']:label+='; escaped'
    suffix=' *' if info['new_system'] else ''
    pretty={'Rossler':'Rössler','HindmarshRose':'Hindmarsh-\nRose',
            'GuckenheimerHolmes':'Guckenheimer-\nHolmes','AtmosphericRegime':'Atmospheric\nregime'}.get(info['system'],info['system'])
    fig.text(cx,.96 if i<5 else .475,pretty+suffix,fontsize=7.1,
             ha='center',va='center',linespacing=1.02)
    fig.text(cx,.56 if i<5 else .08,label,ha='center',va='center',fontsize=6.0,
             color=INK if info['exact'] else ORANGE)
    ax.view_init(elev=24,azim=-62)
    ax.set_box_aspect([1,1,.8])
    ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
    # Fixed inset labels stay clear of the status line in the dense gallery.
    ax.set_xlabel('');ax.set_ylabel('');ax.set_zlabel('')
    for xx,yy,label_axis in [(.10,.17,'$x$'),(.90,.20,'$y$'),(.96,.65,'$z$')]:
        ax.text2D(xx,yy,label_axis,transform=ax.transAxes,fontsize=6)
    ax.grid(False)
    for a in [ax.xaxis,ax.yaxis,ax.zaxis]:a.pane.fill=False
fig.legend(handles=[Line2D([0],[0],color='#85898d',lw=1.4,label='Reference'),
                    Line2D([0],[0],color=BLUE,lw=1.4,label='Discovered')],
           loc='lower center',bbox_to_anchor=(.5,-.025),ncol=2,frameon=False,fontsize=7)
save(fig,'attractor_gallery')
(gallery_dir/'manifest.json').write_text(json.dumps([p[2] for p in panels],indent=2),encoding='utf-8')
print('GALLERY COMPLETE',flush=True)
