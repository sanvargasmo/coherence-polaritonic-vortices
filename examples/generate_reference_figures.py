"""Regenerate corrected plots from the cutoff-free coherent-vortex solution.

Run from the repository root. PNG/PDF pairs, sampled data and validation.json
are written to results/figures. All spatial plots use the documented common
rotating frame. Alpha labels describe each coherent background, not the mean
occupation of the final vortex-modified state.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties
from polaritonic_vortices import ExactCoherentEvolution, SimulationConfig
from polaritonic_vortices.density_minima import track_density_minimum, DensityProfile


def save(fig, directory, name):
    fig.savefig(directory/(name+'.png'),dpi=190,bbox_inches='tight',facecolor='white')
    fig.savefig(directory/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/figures'))
    parser.add_argument('--font-path',type=Path)
    parser.add_argument('--reuse-tracks',action='store_true',help='Reuse previously verified tracking samples for layout-only edits')
    args=parser.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    if args.font_path:
        fontManager.addfont(str(args.font_path))
        font=FontProperties(fname=str(args.font_path)).get_name()
    else:
        font='DejaVu Serif'
    plt.rcParams.update({'font.family':font,'font.size':12,'axes.titlesize':15,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.labelsize':12,'legend.fontsize':10,
                         'mathtext.fontset':'dejavuserif'})
    cavity_color='#B83C62'; exciton_color='#493E76'; green='#78952F'
    evolutions={a:ExactCoherentEvolution(SimulationConfig(mean_photon_number=a)) for a in [0.,.005,.5,1.1]}
    scaled=np.linspace(0,4,801)
    e0=evolutions[0.]; tg=2*np.pi/e0.g
    times=scaled*tg
    series={}; report={'frame':'common rotating at (Ecal_C+Ecal_X)/(2*hbar)',
                       'delta_ps_inv':e0.delta,'tg_ps':tg,
                       'method':'Exact coherent displacement plus conserved total-excitation <=2 seed',
                       'alpha_label':'|alpha|^2 per Cartesian coherent background, before vortex factor',
                       'cases':{}}
    for a,e in evolutions.items():
        entropy=np.array([e.entropy(t) for t in times])
        lc=np.array([e.angular_momentum(t,'cavity') for t in times])
        lx=np.array([e.angular_momentum(t,'exciton') for t in times])
        series[a]=(entropy,lc,lx)
        np.savez(out/f'observables_alpha2_{a:g}.npz',scaled_time=scaled,time_ps=times,entropy=entropy,lz_cavity=lc,lz_exciton=lx)
        report['cases'][str(a)]={'initial_entropy':float(entropy[0]),
            'total_lz_variation':float(np.ptp(lc+lx)),
            'max_seed_norm_error':float(max(abs(np.vdot(e.seed_matrix(t),e.seed_matrix(t)).real-1) for t in times[::10]))}
    def draw_column(ax,e,a):
        entropy,lc,lx=series[a]
        ax[0].plot(scaled,entropy,color=green,lw=2)
        ax[0].set(title=rf'$|\alpha|^2={a:g}$',ylim=(-.015,.6),ylabel=r'Linear entropy $S_L$')
        ax[1].plot(scaled,lc,color=cavity_color,lw=2,label='Cavity')
        ax[1].plot(scaled,lx,color=exciton_color,lw=2,label='Exciton')
        ax[1].plot(scaled,lc+lx,color='#389AB4',lw=1.6,ls='--',label='Total')
        ax[1].set(xlabel=r'$t/t_g$',ylabel=r'$\langle L_z\rangle/\hbar$',xlim=(0,4))
        ax[1].legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.25),frameon=False)
        for axis in ax:
            axis.grid(alpha=.18)
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,layout='constrained')
    for i,a in enumerate([0.,1.1]): draw_column(axes[:,i],evolutions[a],a)
    fig.suptitle('Light–matter entanglement and angular-momentum exchange',fontsize=18)
    save(fig,out,'poster_entropy_lz_comparison')
    for a,slug in [(0.,'nbar0'),(1.1,'nbar11')]:
        fig,axes=plt.subplots(2,1,figsize=(7,7),sharex=True,layout='constrained')
        draw_column(axes,evolutions[a],a); save(fig,out,'poster_entropy_lz_'+slug)
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True,layout='constrained')
    draw_column(axes,evolutions[.5],.5); save(fig,out,'reference_observables')
    fig,ax=plt.subplots(figsize=(8,4.3),layout='constrained')
    ax.plot(times,series[.5][0],color=green,lw=2); ax.set(xlabel='Time (ps)',ylabel=r'$S_L$',title=r'Linear entropy: $|\alpha|^2=0.5$')
    save(fig,out,'linear_entropy_vs_time')
    fig,ax=plt.subplots(figsize=(8,4.3),layout='constrained')
    for data,col,label in [(series[.5][1],cavity_color,'Cavity'),(series[.5][2],exciton_color,'Exciton'),(series[.5][1]+series[.5][2],'#389AB4','Total')]:
        ax.plot(times,data,color=col,label=label)
    ax.set(xlabel='Time (ps)',ylabel=r'$\langle L_z\rangle/\hbar$',title=r'Angular-momentum exchange: $|\alpha|^2=0.5$'); ax.legend()
    save(fig,out,'lz_vs_time')
    print('Observable figures saved',flush=True)

    trajectory_times=np.linspace(0,20,601)
    fig,axes=plt.subplots(2,2,figsize=(10,9),sharex=True,sharey=True,layout='constrained')
    for row,a in enumerate([0.,.005]):
        e=evolutions[a]
        tracks={}
        for col,(sub,color) in enumerate([('exciton',exciton_color),('cavity',cavity_color)]):
            cache=out/f'density_minima_alpha2_{a:g}_{sub}.npz'
            track=(np.load(cache)['xy_over_w'] if args.reuse_tracks and cache.exists()
                   else track_density_minimum(e,trajectory_times,sub))
            tracks[sub]=track
            np.savez(out/f'density_minima_alpha2_{a:g}_{sub}.npz',time_ps=trajectory_times,xy_over_w=track)
            valid=np.isfinite(track).all(axis=1)
            report['cases'][str(a)][sub+'_minimum_coverage']=float(np.mean(valid))
            ax=axes[row,col]
            ax.plot(track[:,0],track[:,1],color=color,lw=1.2)
            ax.scatter(*track[0],marker='x',s=60,color='black',zorder=5)
            ax.set(title=rf'{sub.capitalize()} · $|\alpha|^2={a:g}$',xlim=(-3,3),ylim=(-3,3),xlabel=r'$x/w$',ylabel=r'$y/w$',aspect='equal')
            ax.grid(alpha=.15)
        fig2,ax2=plt.subplots(1,2,figsize=(10,4.7),layout='constrained')
        for ax,sub,color in zip(ax2,['exciton','cavity'],[exciton_color,cavity_color]):
            track=tracks[sub]; ax.plot(track[:,0],track[:,1],color=color,lw=1.2)
            ax.scatter(*track[0],marker='x',s=60,c='black')
            ax.set(title=sub.capitalize(),xlim=(-3,3),ylim=(-3,3),xlabel=r'$x/w$',ylabel=r'$y/w$',aspect='equal')
        fig2.suptitle(rf'Local density-minimum tracks: $|\alpha|^2={a:g}$')
        fig2.supxlabel('Common rotating frame · 0–20 ps · gaps indicate no accepted local minimum',fontsize=10)
        save(fig2,out,'vortex_core_trajectories_'+('nbar0' if a==0 else 'nbar0005'))
        print('Minimum tracks saved for alpha2 =',a,flush=True)
    fig.suptitle('Tracked local minima of the full spatial density',fontsize=18)
    fig.supxlabel('Common rotating frame · 0–20 ps · × initial core · gaps indicate no accepted local minimum',fontsize=10)
    save(fig,out,'vortex_core_trajectories')

    e=evolutions[.5]; w=e.physical.oscillator_width; q=np.linspace(-3.5,3.5,281)
    fractions=[0.,.125,.25]
    fig,axes=plt.subplots(2,3,figsize=(12,8),sharex=True,sharey=True,layout='constrained')
    grids={(sub,j):w*w*e.spatial_density_grid(frac*tg,sub,q*w,q*w) for sub in ['cavity','exciton'] for j,frac in enumerate(fractions)}
    vmax=max(float(g.max()) for g in grids.values())
    for row,sub in enumerate(['cavity','exciton']):
        for j,frac in enumerate(fractions):
            ax=axes[row,j]; im=ax.imshow(grids[sub,j],extent=[q[0],q[-1],q[0],q[-1]],origin='lower',vmin=0,vmax=vmax,cmap='viridis')
            ax.set(title=rf'{sub.capitalize()} · $t/t_g={frac:g}$',xlabel=(r'$x/w$' if row==1 else ''),ylabel=r'$y/w$')
            if j==0:
                x0=getattr(e.physical,sub+'_x')/w; y0=getattr(e.physical,sub+'_y')/w
                ax.scatter(x0,y0,marker='+',c='white',s=75,lw=1.4)
    fig.colorbar(im,ax=axes,label=r'$w^2D(x,y,t)$',shrink=.75)
    fig.suptitle(r'Spatial dynamics: $|\alpha|^2=0.5$',fontsize=18)
    fig.supxlabel('Common rotating frame · common color scale · + prepared initial core',fontsize=10)
    save(fig,out,'spatial_density_evolution')
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    for ax,sub in zip(axes,['cavity','exciton']):
        im=ax.imshow(grids[sub,0],extent=[q[0],q[-1],q[0],q[-1]],origin='lower',vmin=0,vmax=vmax,cmap='viridis')
        ax.scatter(getattr(e.physical,sub+'_x')/w,getattr(e.physical,sub+'_y')/w,marker='+',c='white',s=75)
        ax.set(title=sub.capitalize(),xlabel=r'$x/w$',ylabel=r'$y/w$')
    fig.colorbar(im,ax=axes,label=r'$w^2D(x,y,0)$',shrink=.8)
    fig.suptitle(r'Initial vortex densities: $|\alpha|^2=0.5$')
    save(fig,out,'initial_spatial_densities')
    for sub in ['cavity','exciton']:
        fig,ax=plt.subplots(figsize=(5.7,5),layout='constrained')
        im=ax.imshow(grids[sub,0],extent=[q[0],q[-1],q[0],q[-1]],origin='lower',vmin=0,vmax=vmax,cmap='viridis')
        ax.set(title=sub.capitalize()+r' density at $t=0$, $|\alpha|^2=0.5$',xlabel=r'$x/w$',ylabel=r'$y/w$')
        fig.colorbar(im,ax=ax,label=r'$w^2D(x,y,0)$')
        save(fig,out,sub+'_density_t0')
    # Full exact-state Fock projections: a check of why finite cutoffs matter.
    report['fock_projection_missing_probability']={}
    for cutoff in [8,12,16,20,24]:
        missing=max(max(0.,1-np.vdot(c,c).real) for t in times[::40] for c in [evolutions[1.1].fock_matrix(t,cutoff)])
        report['fock_projection_missing_probability'][str(cutoff)]=float(missing)
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
