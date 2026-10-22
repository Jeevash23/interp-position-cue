"""Static scientific summary of the frozen confirmation factorial."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ai08-v3-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import core as c


def main():
    s=c.read(c.ROOT/'analysis/confirmation/summary.json');order=['third_name@0.5','third_name@1','repeat_io@0.5','repeat_io@1']
    labels=['Third name\nHalf strength','Third name\nFull strength','Reversed role\nHalf strength','Reversed role\nFull strength']
    output=c.ROOT/'report/figures';output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained');x=np.arange(4);data={}
    for axis,kind,title in [(ax[0,0],'necessity','A  Blocking the selected pathways'),(ax[0,1],'sufficiency','B  Recreating their changes without the source edit')]:
        known=[];joint=[];lo=[];hi=[]
        for key in order:
            r=s['strata'][key];D=r['absolute']['clamp_0']['mean'];v=r['ratios'][f'joint_{kind}'];joint.append(v['estimate']);lo.append(v['estimate']-v['interval'][0]);hi.append(v['interval'][1]-v['estimate'])
            known.append(1-r['absolute']['clamp_1']['mean']/D if kind=='necessity' else r['absolute']['restore_1']['mean']/D)
        axis.bar(x-.18,known,.34,label='Five known queries',color='#929da6')
        bars=axis.bar(x+.18,joint,.34,label='Queries + two heads + one MLP',color='#187c8b',yerr=[lo,hi],capsize=3)
        axis.bar_label(bars,labels=[f'{v:.0%}' for v in joint],padding=12,fontsize=9)
        axis.axhline(.75,ls='--',lw=1,color='#b26e44',label='75% point-estimate criterion');axis.axhline(0,color='#596270',lw=.7)
        axis.set(xticks=x,xticklabels=labels,title=title,ylabel='Fraction of source-order contrast',ylim=(-.1,max(1.4,max(np.array(joint)+np.array(hi))+.2)))
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
        data[kind]=dict(known=known,joint=joint,lower_error=lo,upper_error=hi)
    ax[0,0].legend(loc='upper left',fontsize=8,frameon=False)
    for corruption,color,label in [('third_name','#187c8b','Third-name corruption'),('repeat_io','#b26e44','Reversed-role corruption')]:
        vals=[s['strata'][f'{corruption}@{dose:g}']['absolute']['clamp_0']['mean'] for dose in [.5,1.]]
        ax[1,0].plot([.5,1.],vals,'o-',color=color,label=label)
        for xx,yy in zip([.5,1.],vals):ax[1,0].annotate(f'{yy:.2f}',(xx,yy),xytext=(4,8),textcoords='offset points')
    ax[1,0].set(title='C  The source effect depends on correction strength',xlabel='Correction strength',ylabel='Matched minus opposite order (logit units)',xticks=[.5,1.],xlim=(.4,1.1));ax[1,0].legend(frameon=False,fontsize=9);ax[1,0].grid(alpha=.15)
    vals=[];lo=[];hi=[]
    for key in order:
        r=s['strata'][key]['ratios']['restore_interaction_total'];vals.append(r['estimate']);lo.append(r['estimate']-r['interval'][0]);hi.append(r['interval'][1]-r['estimate'])
    bars=ax[1,1].bar(x,vals,color='#7a5a98',yerr=[lo,hi],capsize=3)
    ax[1,1].axhline(0,color='#596270',lw=1);ax[1,1].set(xticks=x,xticklabels=labels,title='D  Combined restoration versus isolated effects',ylabel='(Combined − sum of isolated effects) / order contrast')
    ax[1,1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax[1,1].grid(axis='y',alpha=.15);ax[1,1].set_axisbelow(True)
    fig.suptitle('AI-08: additional pathways and interactions on fresh confirmation inputs',fontsize=15,fontweight='bold')
    fig.savefig(output/'AI-08-pathway-results.png',dpi=180);fig.savefig(output/'AI-08-pathway-results.svg');plt.close(fig)
    c.write(output/'figure-data.json',dict(order=order,primary=data,interaction_total=vals,
        notes='Joint-effect error bars: 99.6875% crossed-bootstrap intervals, Bonferroni family of 16. Interaction bars: descriptive 95%. Percentages are separate interventional estimands, not additive causal shares.'))
    print('Figures saved',flush=True)


if __name__=='__main__':main()
