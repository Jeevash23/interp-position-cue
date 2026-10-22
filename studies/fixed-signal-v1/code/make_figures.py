"""Publication-style summaries from frozen AI-08 analysis artifacts."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ai08-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import core as c


def main():
    out=c.ROOT/'report/figures';out.mkdir(exist_ok=True,parents=True)
    summaries={k:c.read(c.ROOT/'analysis'/k/'summary.json') for k in ['holdout','confirmation']}
    controls=c.read(c.ROOT/'analysis/control-descriptions.json')['banks']['confirmation']
    contexts=c.read(c.ROOT/'data/confirmation-qualified.json')['contexts']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'axes.titleweight':'bold','axes.titlesize':12,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained',gridspec_kw={'height_ratios':[1,1.2]})
    ax=axes[0,0];xx=np.arange(2);w=.3
    for offset,key,label,color in [(-w/2,'gain_normalized_error','Scalar gain','#697a8c'),(w/2,'transform_normalized_error','Gain + directional map','#167b96')]:
        vals=[d[key] for d in summaries.values()];bars=ax.bar(xx+offset,vals,w,label=label,color=color)
        ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=11)
    ax.set(xticks=xx,xticklabels=['Pilot holdout\n128 recipients','Confirmation\n1,152 recipients'],ylim=(0,.53),
        ylabel='Normalized squared prediction error',title='A  Modest improvement on unseen groups')
    ax.legend(frameon=False,fontsize=10,loc='upper right');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax=axes[0,1]
    for i,(key,d) in enumerate(summaries.items()):
        mean=d['improvement']['mean'];lo,hi=d['improvement']['ci95']
        ax.errorbar(mean,1-i,xerr=[[mean-lo],[hi-mean]],fmt='o',color='#167b96',capsize=5,markersize=8)
        ax.text(mean,1-i+.18,f'{mean:.4f} [{lo:.4f}, {hi:.4f}]',ha='center',fontsize=10)
    ax.axvline(0,color='#6b7280',lw=1);ax.axvline(.05,color='#aa643a',ls='--',label='Predeclared 0.05 threshold')
    ax.set(yticks=[1,0],yticklabels=['Pilot holdout','Confirmation'],ylim=(-.5,1.6),xlim=(-.012,.12),
        xlabel='Gain error minus directional-map error',title='B  Confirmation misses the effect-size rule')
    ax.legend(frameon=False,fontsize=10,loc='upper right');ax.grid(axis='x',alpha=.15)
    ax=axes[1,0];s=summaries['confirmation']
    labels=['Matching\nname order','Opposite\nname order','All positive\nsource edits','Recipient\nself edit','Random\ndirections']
    vals=[controls['positive_mention_order']['matching_mention_order']['margin_repair'],controls['positive_mention_order']['opposite_mention_order']['margin_repair'],
        s['positive_strength_margin_repair'],s['behavior']['recipient_self']['margin_repair'],s['random_margin_repair']]
    colors=['#167b96','#b76b45','#5d92a1','#447a64','#969da4'];bars=ax.bar(np.arange(5),vals,color=colors,width=.7)
    ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=11)
    ax.axhline(0,color='#6b7280',lw=1);ax.set(xticks=np.arange(5),xticklabels=labels,ylim=(-1.55,2.25),
        ylabel='Mean change in intended-answer logit margin',title='C  Signal reuse depends strongly on name order')
    ax.tick_params(axis='x',labelsize=10);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax=axes[1,1];effect=s['improvement'];matrix=np.array(effect['numerator_grid'])/np.array(effect['denominator_grid'])
    limit=max(abs(matrix.min()),abs(matrix.max()));im=ax.imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    ax.set(xticks=np.arange(12),xticklabels=[v[0] for v in contexts],yticks=np.arange(12),yticklabels=[f'N{i+1:02d}' for i in range(12)],
        ylabel='Held-out name group',title='D  Improvement varies across confirmation groups')
    ax.tick_params(axis='x',labelrotation=50,labelsize=9);ax.tick_params(axis='y',labelsize=9)
    fig.colorbar(im,ax=ax,fraction=.045,pad=.02,label='Gain error minus directional-map error')
    fig.suptitle('AI-08: a fixed internal signal changes with recipient context',fontsize=17,fontweight='bold')
    fig.savefig(out/'AI-08-results.png',dpi=180);fig.savefig(out/'AI-08-results.svg');plt.close(fig)
    c.write(out/'figure-data.json',dict(primary={k:{f:d[f] for f in ['gain_normalized_error','transform_normalized_error','improvement']} for k,d in summaries.items()},
        confirmation_behavior=dict(labels=labels,values=vals),notes='Panel B: 95% crossed name/context bootstrap intervals, 20,000 draws. Panels C/D descriptive; no multiplicity-adjusted slice inference. Errors include all 1,024 dimensions.'))
    print('Figures saved',flush=True)


if __name__=='__main__':main()
