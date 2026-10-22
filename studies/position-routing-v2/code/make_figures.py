"""Scientific figures for the prospective positional-signal explanation test."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ai08-causal-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import core as c


def main():
    summary=c.read(c.ROOT/'analysis/confirmation/summary.json');cond=summary['condition_summary'];absolute=summary['absolute_contrasts'];ratios=summary['ratios']
    output=c.ROOT/'report/figures';output.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    ax=axes[0,0];keys=['same_matched','same_opposite','foreign_matched','foreign_opposite'];vals=[cond[k]['margin_repair'] for k in keys]
    bars=ax.bar(range(4),vals,color=['#197d8d','#b36e45','#197d8d','#b36e45'],width=.65);ax.bar_label(bars,fmt='%.2f',padding=4)
    ax.axhline(0,color='#596270',lw=1);ax.set(xticks=range(4),xticklabels=['Same names\nMatched order','Same names\nOpposite order','New names\nMatched order','New names\nOpposite order'],
        ylabel='Mean change in intended-answer margin',title='A  Order effects transfer between names')
    ax.tick_params(axis='x',labelsize=9)
    ax.margins(y=.2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax=axes[0,1];baseline=absolute['same']['mean']
    keys=['q','k','v','p','z'];removed=[1-absolute[f'same_clamp_{key}']['mean']/baseline for key in keys]
    removed.append(1-(absolute['same_control0']['mean']+absolute['same_control1']['mean'])/(2*baseline))
    bars=ax.bar(range(6),removed,color=['#7b5c9a','#929da6','#929da6','#7b5c9a','#7b5c9a','#929da6'],width=.65)
    ax.bar_label(bars,labels=[f'{v:.1%}' for v in removed],padding=4,fontsize=10);ax.axhline(0,color='#596270',lw=1)
    ax.set(xticks=range(6),xticklabels=['Queries','Keys','Values','Attention\nweights','Head\noutputs','Other-head\nqueries'],ylim=(-.12,.78),
        ylabel='Fraction of order contrast removed',title='B  Query changes explain part of the effect')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax=axes[1,0];keys=['foreign_retention','query_necessity','query_sufficiency'];labels=['Retained with different names','Removed by query clamping','Recreated by queries alone']
    for i,key in enumerate(keys):
        d=ratios[key];m=d['estimate'];lo,hi=d['interval'];ax.errorbar(m,2-i,xerr=[[m-lo],[hi-m]],fmt='o',color='#197d8d',capsize=5,markersize=7)
        ax.text(m,2-i+.18,f'{m:.1%}',ha='center',fontsize=11)
    ax.axvline(.5,color='#a7643e',ls='--',label='Predeclared 50% threshold');ax.axvline(0,color='#596270',lw=1)
    ax.set(yticks=[2,1,0],yticklabels=labels,ylim=(-.5,2.55),xlim=(-.05,1.25),xlabel='Ratio to the original order contrast',title='C  Primary tests support a partial explanation')
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.legend(frameon=False,fontsize=9,loc='upper right');ax.grid(axis='x',alpha=.15)
    ax=axes[1,1];keys=['zero','same_matched','same_opposite','same_matched:query_only','same_opposite:query_only'];vals=[cond[k]['attention_io_minus_s1'] for k in keys]
    bars=ax.bar(range(5),vals,color=['#929da6','#197d8d','#b36e45','#197d8d','#b36e45'],width=.65);ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=10)
    ax.axhline(0,color='#596270',lw=1);ax.set(xticks=range(5),xticklabels=['Baseline','Matched\nsignal','Opposite\nsignal','Matched\nqueries only','Opposite\nqueries only'],
        ylabel='Attention to intended name minus subject',title='D  Queries redirect attention between names')
    ax.tick_params(axis='x',labelsize=9)
    ax.margins(y=.22);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('AI-08 causal follow-up: positional information and attention queries',fontsize=16,fontweight='bold')
    fig.savefig(output/'AI-08-causal-results.png',dpi=180);fig.savefig(output/'AI-08-causal-results.svg');plt.close(fig)
    c.write(output/'figure-data.json',dict(confirmation_recipients=512,primary_ratios=ratios,clamp_removed_fractions=dict(zip(['q','k','v','p','z','random_q'],removed)),
        notes='Panel C uses 98.3333% crossed-bootstrap intervals for a family of three ratios. Other panels are descriptive. Fractions are intervention-specific, not additive causal shares.'))
    print('Causal figures saved',flush=True)


if __name__=='__main__':main()
