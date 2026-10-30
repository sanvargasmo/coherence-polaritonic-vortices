"""Exact Gaussian-removed core tracks and matched entanglement/OAM scans."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from polaritonic_vortices import ExactCoherentEvolution,SimulationConfig
from polaritonic_vortices.paper_core import track_paper_core


def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=140,bbox_inches='tight',facecolor='white')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def occupation(e,time,sub):
    rho=e.reduced_seed(time,sub)
    pairs=e.pairs; index={pair:i for i,pair in enumerate(pairs)}
    value=float(np.dot(np.diag(rho).real,[sum(pair) for pair in pairs]))
    off=0 if sub=='cavity' else 2
    for axis,alpha in enumerate(e.displacement(time)[off:off+2]):
        a=np.zeros((len(pairs),len(pairs)),complex)
        for j,pair in enumerate(pairs):
            if pair[axis]:
                lower=list(pair);lower[axis]-=1
                a[index[tuple(lower)],j]=np.sqrt(pair[axis])
        value+=abs(alpha)**2+2*np.real(alpha.conjugate()*np.trace(rho@a))
    return float(value)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/paper_core'))
    parser.add_argument('--samples',type=int,default=801)
    parser.add_argument('--tmax',type=float,default=20.)
    parser.add_argument('--cases',type=float,nargs='+',default=[0.,.005,.5,1.1])
    parser.add_argument('--reuse',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'mathtext.fontset':'dejavuserif'})
    ts=np.linspace(0,args.tmax,args.samples)
    evolutions={a:ExactCoherentEvolution(SimulationConfig(mean_photon_number=a)) for a in args.cases}
    e0=next(iter(evolutions.values()));tg=2*np.pi/e0.g
    report={'definition':'F_k(r,t)=w^2 exp(|r|^2/w^2) D_k(r,t); track continuous local minima',
            'envelope':'fixed at original origin, as in the paper; no Taylor truncation',
            'state_evolution':'exact coherent displacement plus conserved total-excitation <=2 seed; all undisplaced coherent manifolds retained',
            'frame':'common rotating frame','time_window_ps':[0,args.tmax],'tg_ps':tg,
            'alpha2_definition':'Per Cartesian coherent background before vortex imprint; NOT final mean photon number',
            'search_radius_over_w':12.,'max_accepted_step_over_w':.5,'cases':{}}
    tracks={}; series={}
    for a,e in evolutions.items():
        entry={'initial_mean_photons':occupation(e,0,'cavity'),'initial_mean_excitons':occupation(e,0,'exciton')}
        for sub in ['cavity','exciton']:
            cache=out/f'core_alpha2_{a:g}_{sub}.npz'
            track=dict(np.load(cache)) if args.reuse and cache.exists() else track_paper_core(e,ts,sub)
            if track['time_ps'].shape != ts.shape or not np.allclose(track['time_ps'],ts,rtol=0,atol=1e-12):
                raise ValueError('Cached trajectory sampling differs; rerun without --reuse')
            tracks[a,sub]=track;np.savez_compressed(cache,**track)
            with (out/f'core_alpha2_{a:g}_{sub}.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['time_ps','x_over_w','y_over_w','gradient_residual','hessian_min','candidate_count','branch_break'])
                writer.writerows(zip(track['time_ps'],*track['xy_over_w'].T,track['gradient_residual'],track['hessian_min'],track['candidate_count'],track['branch_break']))
            valid=np.isfinite(track['xy_over_w']).all(axis=1)
            initial=np.array([getattr(e.physical,sub+'_x'),getattr(e.physical,sub+'_y')])/e.physical.oscillator_width
            entry[sub]={'accepted_fraction':float(valid.mean()),'max_gradient_residual':float(np.nanmax(track['gradient_residual'])) if valid.any() else None,
                        'no_candidate_samples':int(np.sum(track['candidate_count']==0)),
                        'continuity_rejected_samples':int(np.sum((~valid)&(track['candidate_count']>0))),
                        'initial_core_error':float(np.linalg.norm(track['xy_over_w'][0]-initial)),
                        'minimum_hessian_factor':float(np.nanmin(track['hessian_min'])) if valid.any() else None,
                        'multiple_candidate_fraction':float(np.mean(track['candidate_count']>1))}
            print('Tracked',a,sub,entry[sub],flush=True)
        scaled=np.linspace(0,4,1601);ot=scaled*tg
        entropy=np.array([e.entropy(t) for t in ot])
        lc=np.array([e.angular_momentum(t,'cavity') for t in ot]);lx=np.array([e.angular_momentum(t,'exciton') for t in ot])
        series[a]=(scaled,entropy,lc,lx)
        np.savez_compressed(out/f'observables_alpha2_{a:g}.npz',time_ps=ot,scaled_time=scaled,linear_entropy=entropy,lz_cavity=lc,lz_exciton=lx)
        with (out/f'observables_alpha2_{a:g}.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['time_ps','t_over_tg','linear_entropy','lz_cavity_over_hbar','lz_exciton_over_hbar','lz_total_over_hbar']);writer.writerows(zip(ot,scaled,entropy,lc,lx,lc+lx))
        entry.update(initial_entropy=float(entropy[0]),total_lz_variation=float(np.ptp(lc+lx)),min_entropy=float(entropy.min()),max_entropy=float(entropy.max()))
        report['cases'][f'{a:g}']=entry
    colors={'cavity':'#B83C62','exciton':'#493E76'}
    def trajectory_figure(cases,name):
        fig,axs=plt.subplots(2,len(cases),figsize=(4.1*len(cases),7.8),squeeze=False,layout='constrained')
        for col,a in enumerate(cases):
            for row,sub in enumerate(['cavity','exciton']):
                track=tracks[a,sub];xy=track['xy_over_w'].copy()
                # Never draw a segment across reacquired branches or NaN gaps.
                plot=xy.copy();plot[track['branch_break']]=np.nan
                ax=axs[row,col];ax.plot(plot[:,0],plot[:,1],color=colors[sub],lw=1.3)
                ax.scatter(*xy[0],marker='x',s=45,c='black',zorder=5)
                symbol='C' if sub=='cavity' else 'X'
                initial_mean=report['cases'][f'{a:g}']['initial_mean_photons' if sub=='cavity' else 'initial_mean_excitons']
                ax.set(title=rf'{sub.capitalize()} · $|\alpha|^2={a:g}$'+'\n'+rf'$\langle N_{symbol}(0)\rangle={initial_mean:.3f}$',xlabel=r'$x/w$',ylabel=r'$y/w$',aspect='equal',xlim=(-3,3),ylim=(-3,3));ax.grid(alpha=.15)
        fig.suptitle('Core tracks after removing the fixed Gaussian envelope',fontsize=17)
        fig.supxlabel(f'Exact evolution · 0–{args.tmax:g} ps · × initial core · gaps: no accepted continuous local minimum · view: ±3w',fontsize=10)
        save(fig,out,name)
    trajectory_figure(args.cases,'trajectories_all')
    if len(args.cases)==4:
        trajectory_figure(args.cases[:2],'trajectories_low_background')
        trajectory_figure(args.cases[2:],'trajectories_higher_background')
    def observable_figure(cases,name):
        fig,axs=plt.subplots(2,len(cases),figsize=(4.1*len(cases),6.3),sharex=True,squeeze=False,layout='constrained')
        for col,a in enumerate(cases):
            scaled,entropy,lc,lx=series[a]
            axs[0,col].plot(scaled,entropy,color='#78952F',lw=1.6)
            initial_mean=report['cases'][f'{a:g}']['initial_mean_photons']
            axs[0,col].set(title=rf'$|\alpha|^2={a:g},\quad\langle N_C(0)\rangle={initial_mean:.3f}$',ylabel=r'$S_L=1-\mathrm{Tr}(\rho_C^2)$',ylim=(-.02,.7))
            for values,c,label,style in [(lc,colors['cavity'],'Cavity','-'),(lx,colors['exciton'],'Exciton','-'),(lc+lx,'#389AB4','Total','--')]:axs[1,col].plot(scaled,values,color=c,label=label,ls=style,lw=1.4)
            axs[1,col].set(xlabel=r'$t/t_g$',ylabel=r'$\langle L_z\rangle/\hbar$',xlim=(0,4))
            for ax in axs[:,col]:ax.grid(alpha=.15)
        handles,labels=axs[1,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
        fig.suptitle('Light–matter entanglement and angular-momentum exchange',fontsize=17)
        save(fig,out,name)
    observable_figure(args.cases,'entropy_angular_momentum_all')
    if len(args.cases)>1:observable_figure([args.cases[0],args.cases[-1]],'entropy_angular_momentum_selected')
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Figures and data saved to',out,flush=True)


if __name__=='__main__':main()
