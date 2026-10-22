"""Machine-readable accounting and scoped completion decision from saved evidence."""
import numpy as np
import core as c


def main():
    path=c.ROOT/'analysis/STUDY-SUMMARY.json';assert not path.exists();banks={};main_examples=0
    for bank in ['development','holdout','confirmation']:
        run=c.ROOT/'runs'/bank;rows=c.read(run/'recipients.json');sources=c.read(run/'sources.json');s=c.read(c.ROOT/'analysis'/bank/'summary.json')
        n=len(rows);ns=len(sources);main_examples+=2*n+2*ns+12*n
        with np.load(c.ROOT/'analysis'/bank/'prediction-errors.npz') as z:energy=z['energy']
        pairs=c.read(c.ROOT/'analysis'/bank/'pairs.json')
        banks[bank]=dict(recipients=n,source_pairs=ns,condition_records=13*n,prediction_pairs=len(pairs),
            unique_clean_recipient_strings=len({r['clean_text'] for r in rows}),unique_corrupted_recipient_strings=len({r['corrupted_text'] for r in rows}),
            gain_error=s['gain_normalized_error'],transform_error=s['transform_normalized_error'],absolute_improvement=s['improvement']['mean'],
            interval95=s['improvement']['ci95'],relative_improvement=s['improvement']['mean']/s['gain_normalized_error'],
            mechanism_gates=s['mechanism_gates'],continue_to_mechanism=s['continue_to_mechanism'],
            primary_energy_by_strength={str(t):float(energy[np.array([p['strength']==t for p in pairs])].sum()/energy.sum()) for t in [-.5,.5,1.]})
    qrows=c.read(c.ROOT/'runs/qualification/recipients.json')
    numeric_rows=len([r for r in qrows if len(r['tokens_clean'])==len(qrows[0]['tokens_clean'])][:4]);assert numeric_rows==4
    extra_self=len(qrows);numeric_examples=6*numeric_rows;main_examples+=extra_self+numeric_examples
    singleton=c.read(c.ROOT/'runs/model-replay/summary.json');batch=c.read(c.ROOT/'runs/batch-replay/summary.json')
    audits={name:c.read(c.ROOT/'analysis'/f'{name}.json')['status'] for name in ['EVIDENCE-AUDIT','INPUT-AUDIT','ANALYSIS-REPLAY']}
    assert set(audits.values())=={'passed'} and batch['status']=='passed' and batch['bitwise_identical']
    assert singleton['status']=='failed'  # Documented; never silently promote this check to passing.
    sensitivity=c.read(c.ROOT/'analysis/REPLAY-SENSITIVITY.json')
    cv=c.read(c.ROOT/'analysis/development/selection.json')
    out=dict(status='complete_with_documented_replay_limitation',created_utc=c.now(),banks=banks,
        total_recipient_cases=sum(v['recipients'] for v in banks.values()),total_source_pairs=sum(v['source_pairs'] for v in banks.values()),
        total_condition_records=sum(v['condition_records'] for v in banks.values()),total_prediction_pairs=sum(v['prediction_pairs'] for v in banks.values()),
        model_examples=dict(main=main_examples,extra_qualification_self=extra_self,numerical_qualification=numeric_examples,
            singleton_replay=singleton['singleton_model_examples'],original_batch_replay=batch['model_examples'],
            total=main_examples+singleton['singleton_model_examples']+batch['model_examples']),
        development_cross_validation_relative_improvement=1-cv['transform_cv_sse']/cv['gain_cv_sse'],
        decision=dict(controlled_program_complete=True,broader_mechanistic_question_resolved=False,
            causal_localization='not_triggered_under_predeclared_rule',routing_mechanism_identified=False,
            confirmation_stop_reason='Absolute normalized improvement 0.03777745 is below the predeclared 0.05 point-estimate threshold',
            pilot_stop_reason='F11/order1 clean competence 27/32 is below 0.85',no_continuation_running_or_scheduled=True),
        verification=dict(audits=audits,original_batch_replay='bitwise_identical',singleton_replay='failed_absolute_state_tolerance; retained',
            largest_sample_improvement_change=max(abs(v['changes']['improvement']) for v in sensitivity['banks'].values())),
        source_sha256=c.sha(__file__))
    c.write(path,out);print('Summary saved:',out['total_condition_records'],'condition records;',out['model_examples']['total'],'model examples',flush=True)


if __name__=='__main__':main()
