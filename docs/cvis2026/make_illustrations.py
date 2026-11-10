"""Reproducible CVIS figures: fresh fixed cases and saved benchmark summaries.

Run from the repository root. Discovery runs once per system and is cached;
subsequent invocations redraw figures from saved numerical evidence.
"""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from threadpoolctl import threadpool_limits
from scipy.signal import savgol_filter
from eqdisc import discover
from eqdisc import joint_discover as jd
from eqdisc.core import SYSTEMS, X0, DT, NAMES, EXP_ARR, simulate, metrics, xi_to_scaled

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
EVIDENCE = HERE / 'evidence'
FIG.mkdir(exist_ok=True)
EVIDENCE.mkdir(exist_ok=True)
P = json.loads((HERE / 'illustration-protocol.json').read_text())
BLUE, ORANGE, INK, MUTED = '#1763a6', '#d16b32', '#222222', '#80858a'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5,
                     'axes.titlesize': 10, 'axes.labelsize': 8.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'legend.fontsize': 8, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': .6, 'lines.linewidth': 1.2})


def terminal_state(observations, xi, dt, sigma):
    """Causal states-only LM, following archive/forecast_select.states_fit."""
    scale = np.sqrt(np.mean(observations ** 2, axis=0))
    yo = observations[-40:] / scale
    sig = np.maximum(np.asarray(sigma) / scale, 1e-7)
    C = xi_to_scaled(xi, scale)
    sub = jd.auto_substeps(observations / scale, dt)
    nf = (len(yo)-1)*sub + 1
    oi = np.arange(len(yo))*sub
    y = np.zeros((nf, 3)); y[oi] = yo
    mask = np.zeros(nf, bool); mask[oi] = True
    smooth = savgol_filter(yo, 11, 3, axis=0)
    grid = np.arange(nf)/sub
    u = np.column_stack([np.interp(grid, np.arange(len(yo)), smooth[:, j]) for j in range(3)])
    params = np.zeros(0, dtype=int)
    mu = 1e-3
    for _ in range(30):
        ab, auc, acc, gu, gc, cost = jd._normal(u, C, params, y, mask, dt/sub, sig, .03, 2.)
        try:
            du, _ = jd._schur_solve(ab, auc, acc, gu, gc, mu)
        except (ValueError, np.linalg.LinAlgError):
            mu *= 10
            continue
        un = u + du.reshape(u.shape)
        rd, rm, _, _ = jd._residuals(un, C, y, mask, dt/sub, sig, .03)
        new = jd._hsum(rd, 2.) + float(np.sum(rm**2))
        if np.isfinite(new) and new < cost:
            u = un
            mu = max(mu/3, 1e-9)
            if (cost-new)/max(cost, 1e-300) < 1e-8:
                break
        else:
            mu *= 4
    return u[-1]*scale


def run_cases():
    cases = []
    rng = np.random.default_rng(P['seed'])
    n, h = P['training_samples'], P['forecast_samples']
    for name in P['systems_in_order']:
        # Draw all noise in fixed system order even when caches already exist.
        truth = simulate(SYSTEMS[name], X0[name], DT[name], n+h, burn=P['burn_samples'])
        assert truth is not None
        std = truth[:n].std(axis=0)
        noise = .1*std*rng.standard_normal((n,3))
        observed = truth[:n] + noise
        cache = EVIDENCE / f'{name}.npz'
        report = EVIDENCE / f'{name}.json'
        if not cache.exists():
            print('FITTING', name, flush=True)
            start = time.perf_counter()
            with threadpool_limits(limits=1):
                result = discover(observed, DT[name], confidence=False)
            elapsed = time.perf_counter()-start
            xi = result.xi
            initial = terminal_state(observed, xi, DT[name], result.info['sigma'])
            pred = simulate(xi, initial, DT[name], h+1)
            if pred is None:
                pred = np.full((h+1,3), np.nan)
                pred[0] = initial
            # Exclude the initial point: first scored target is one step after origin.
            ref = truth[n:n+h]
            err = (pred[1:]-ref)/std
            summary = {'system':name, 'seed':P['seed'], 'dt':DT[name],
                       'method_version':'1.2', 'n_train':n, 'n_forecast':h,
                       'noise':.1, 'seconds':elapsed,
                       **metrics(xi, SYSTEMS[name]),
                       'ambiguous':bool(result.ambiguous), 'gap':float(result.gap),
                       'nrmse_first100':float(np.sqrt(np.mean(err[:100]**2))),
                       'nrmse_all400':float(np.sqrt(np.mean(err**2))),
                       'equations':result.equations(digits=5),
                       'initialization':'states-only fit, final 40 training observations'}
            learned_attr = simulate(xi, initial, DT[name], 8000, burn=1000)
            true_attr = simulate(SYSTEMS[name], initial, DT[name], 8000, burn=1000)
            np.savez_compressed(cache, truth=truth, observed=observed, xi=xi,
                                xi_true=SYSTEMS[name], initial=initial,
                                forecast=pred, std=std,
                                attractor_learned=learned_attr if learned_attr is not None else np.empty((0,3)),
                                attractor_true=true_attr if true_attr is not None else np.empty((0,3)))
            report.write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(json.dumps(summary), flush=True)
        cases.append((name, dict(np.load(cache)), json.loads(report.read_text())))
    return cases


def save(fig, name):
    fig.savefig(FIG/f'{name}.pdf', bbox_inches='tight', pad_inches=.035)
    fig.savefig(FIG/f'{name}.png', dpi=180, bbox_inches='tight', pad_inches=.035)
    plt.close(fig)


def prediction_figure(cases):
    fig, axes = plt.subplots(3, 2, figsize=(6.75, 2.5),
                             gridspec_kw={'width_ratios':[2.55,1]}, layout='constrained')
    n, h = P['training_samples'], P['forecast_samples']
    for row,(name,d,s) in enumerate(cases):
        dt=s['dt']; ax,err_ax=axes[row]
        past=150
        t=(np.arange(n-past,n+h)-(n-1))*dt
        ax.axvspan(-past*dt,0,color='#edf0f2',zorder=0)
        ax.plot(t,d['truth'][n-past:n+h,0],color=INK,label='Clean reference',lw=1.)
        ax.scatter((np.arange(n-past,n)-(n-1))*dt,d['observed'][-past:,0],
                   s=3,color=MUTED,alpha=.45,label='Training observations',rasterized=True)
        ax.plot(np.arange(h+1)*dt,d['forecast'][:,0],color=BLUE,lw=1.3,label='PJSS forecast')
        ax.axvline(0,color=MUTED,ls=':',lw=.7)
        ax.set_ylabel(f'{name.capitalize()}\n$x$')
        ax.set_yticks([-10,10])
        ax.set_xlim(-past*dt,h*dt)
        ax.grid(alpha=.15)
        er=np.sqrt(np.mean(((d['forecast'][1:]-d['truth'][n:])/d['std'])**2,axis=1))
        err_ax.plot(np.arange(1,h+1)*dt,er,color=BLUE)
        err_ax.set_ylim(bottom=0)
        err_ax.set_xlim(0,h*dt)
        err_ax.grid(alpha=.15)
        err_ax.text(.04,.84,f'RMSE = {s["nrmse_all400"]:.3f}',transform=err_ax.transAxes,fontsize=7.5)
    axes[0,0].set_title('(a) Observed training tail and autonomous prediction',loc='left')
    axes[0,1].set_title('(b) Normalized error',loc='left')
    axes[-1,0].set_xlabel('Time from forecast origin (system units)')
    axes[-1,1].set_xlabel('Lead time')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    save(fig,'predictions')


def attractor_figure(cases):
    fig=plt.figure(figsize=(6.75,3.7),layout='constrained')
    for col,(name,d,s) in enumerate(cases):
        true=d['attractor_true']; learned=d['attractor_learned']
        if len(learned):
            allx=np.vstack([true,learned]); lim=np.column_stack([allx.min(0),allx.max(0)])
        else: lim=np.column_stack([true.min(0),true.max(0)])
        for row,(x,color,label) in enumerate([(true,INK,'Reference'),(learned,BLUE,'Discovered')]):
            ax=fig.add_subplot(2,3,row*3+col+1,projection='3d')
            if len(x):ax.plot(*x.T,color=color,lw=.30,alpha=.8,rasterized=True)
            else:ax.text2D(.05,.5,'Integration failed',transform=ax.transAxes,color=ORANGE)
            ax.set_title(f'{name.capitalize()} | {label}',fontsize=9,pad=0)
            ax.view_init(elev=22,azim=-62)
            ax.set_xlim(*lim[0]);ax.set_ylim(*lim[1]);ax.set_zlim(*lim[2])
            ax.set_xlabel('$x$',labelpad=-8);ax.set_ylabel('$y$',labelpad=-8);ax.set_zlabel('$z$',labelpad=-8)
            ax.tick_params(labelsize=5.5,pad=-3)
            ax.set_box_aspect([1,1,.8])
            ax.grid(False)
            for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False
    save(fig,'attractors')


def support_figure(cases):
    fig,ax=plt.subplots(figsize=(6.75,1.85),layout='constrained')
    # 0 correctly absent, 1 correctly selected, 2 false positive, 3 missed.
    cmap=ListedColormap(['#f0f1f2',BLUE,ORANGE,'#a53f55'])
    grids=[]
    labels=[]
    for name,d,s in cases:
        true=d['xi_true']!=0; got=d['xi']!=0
        code=np.zeros_like(true,dtype=int)
        code[true & got]=1;code[~true & got]=2;code[true & ~got]=3
        grids.append(code.T)
        labels.extend([name.capitalize()+'  $\\dot '+coord+'$' for coord in ('x','y','z')])
    ax.imshow(np.vstack(grids),aspect='auto',cmap=cmap,vmin=0,vmax=3,interpolation='nearest')
    ax.set_xticks(np.arange(20),['$'+x.replace('*',r'\,')+'$' for x in NAMES],fontsize=7.5,rotation=45)
    ax.set_yticks(np.arange(9),labels,fontsize=7.2)
    for boundary in [2.5,5.5]:ax.axhline(boundary,color='white',lw=1.5)
    ax.tick_params(length=0,pad=2)
    for sp in ax.spines.values():sp.set_visible(False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=cmap(i),label=l) for i,l in enumerate(
        ['Correctly absent','Correctly selected','Extra term','Missed term'])],
        loc='outside upper center',ncol=4,frameon=False,fontsize=7)
    save(fig,'support')


def benchmark_figures():
    records=json.loads((ROOT/'results_dysts/test_final.json').read_text())['records']
    meta=json.loads((ROOT/'data/dysts_test.json').read_text())['systems']
    noises=[.01,.05,.1,.2]
    fig,ax=plt.subplots(figsize=(3.25,2.5),layout='constrained')
    for n,col,mk in [(500,ORANGE,'s'),(2000,BLUE,'o')]:
        rates=[np.mean([r['final']['exact'] for r in records if r['n']==n and r['noise']==eta]) for eta in noises]
        ax.plot([100*x for x in noises],rates,color=col,marker=mk,label=f'$N={n}$')
    ax.set_xticks([1,5,10,20]);ax.set_xlabel('Measurement noise (% of state std)')
    ax.set_ylabel('Exact support recovery');ax.set_ylim(.5,1.01);ax.grid(alpha=.18)
    ax.legend(frameon=False,loc='lower left')
    save(fig,'recovery_noise')
    # Per-system recovery without comparison against historical weak baselines.
    names=sorted(set(r['system'] for r in records),key=lambda name:(np.mean([r['final']['exact'] for r in records if r['system']==name]),name))
    vals=[np.mean([r['final']['exact'] for r in records if r['system']==name]) for name in names]
    fig,ax=plt.subplots(figsize=(3.25,1.85),layout='constrained')
    ax.plot(np.arange(1,78),vals,color=BLUE,lw=1.)
    ax.scatter(np.arange(1,78),vals,color=BLUE,s=7)
    ax.axhline(.9,ls=':',lw=.7,color=MUTED)
    ax.set_xlabel('System rank (sorted by recovery)');ax.set_ylabel('Exact recovery (80 runs)')
    ax.set_xlim(0,78);ax.set_ylim(-.03,1.03);ax.grid(alpha=.15)
    ax.text(.05,.14,f'{sum(v>=.9 for v in vals)}/77 systems at 90% or higher',transform=ax.transAxes,fontsize=8)
    save(fig,'recovery_systems')
    # Directly recompute the two profiling arms from the recorded paired runs.
    files=['sealed','colored','hard40','dysts']
    totals={'PJSS':0,'Fixed-state proposals':0,'Forward stepwise':0,'No proposals':0}
    count=0
    for name in files:
        old=json.loads((ROOT/f'results_ablation/{name}.json').read_text())['records']
        new=json.loads((ROOT/f'results_rev/abl_{name}.json').read_text())['records']
        old={r['index']:r for r in old};new={r['index']:r for r in new}
        assert old.keys()==new.keys()
        for index in old:
            variants=old[index]['variants']
            totals['PJSS']+=variants['full']['exact']
            totals['Fixed-state proposals']+=new[index]['variants']['no_schur']['exact']
            totals['Forward stepwise']+=variants['greedy']['exact']
            totals['No proposals']+=variants['no_proposals']['exact']
            count+=1
    assert count==576 and totals['PJSS']==490 and totals['Fixed-state proposals']==480
    fig,ax=plt.subplots(figsize=(3.25,2.3),layout='constrained')
    labels=list(totals)
    ax.barh(labels,list(totals.values()),color=[BLUE,'#9cabb8','#9cabb8','#9cabb8'],height=.58)
    ax.invert_yaxis();ax.set_xlim(0,576);ax.set_xlabel('Exact recoveries (576 runs)')
    ax.tick_params(axis='y',length=0,labelsize=8)
    for j,v in enumerate(totals.values()):ax.text(v+6,j,str(int(v)),va='center',fontsize=8)
    ax.grid(axis='x',alpha=.13)
    save(fig,'ablation')
    (EVIDENCE/'ablation_counts.json').write_text(json.dumps({'n':count,'counts':totals},indent=2))


def write_numbers(cases):
    lines=[]
    def macro(k,v):lines.append('\\newcommand{\\'+k+'}{'+str(v)+'}')
    for name,d,s in cases:
        key=name.capitalize()
        macro(key+'Coef',f'{100*s["coef_err"]:.2f}')
        macro(key+'Short',f'{s["nrmse_first100"]:.3f}')
        macro(key+'Long',f'{s["nrmse_all400"]:.3f}')
        macro(key+'Exact',int(s['exact']))
    # An actual fitted Lorenz equation, with every selected coefficient shown.
    d=cases[0][1]
    for j,var in enumerate('xyz'):
        terms=[]
        for m in np.flatnonzero(d['xi'][:,j]):
            c=d['xi'][m,j]
            feature=NAMES[m].replace('*','').replace('^2','^2').replace('^3','^3')
            terms.append(f'{c:+.3f}'+('' if feature=='1' else feature))
        macro('LorenzEq'+var.upper(),' '.join(terms).lstrip('+'))
    (HERE/'illustration-numbers.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    cases=run_cases()
    prediction_figure(cases)
    attractor_figure(cases)
    support_figure(cases)
    benchmark_figures()
    write_numbers(cases)
    print('FIGURES COMPLETE',flush=True)
