"""Exact final accounting and development-to-confirmation prediction comparison."""
import core as c


def main():
    output=c.ROOT/'analysis/STUDY-SUMMARY.json';assert not output.exists()
    phases={phase:c.read(c.ROOT/f'runs/{phase}/status.json') for phase in ['screen','development','confirmation']}
    q=c.read(c.ROOT/'runs/qualification/summary.json');r=c.read(c.ROOT/'runs/replay/summary.json')
    audits={name:c.read(c.ROOT/f'analysis/{name}.json')['status'] for name in ['INPUT-AUDIT','EVIDENCE-AUDIT','ANALYSIS-REPLAY','SERIALIZATION-EQUALITY']}
    assert all(s['status']=='complete' for s in phases.values()) and all(v=='passed' for v in audits.values())
    assert q['status']=='passed' and r['status']=='passed' and r['bitwise_identical']
    examples={phase:s['model_examples'] for phase,s in phases.items()};examples.update(qualification=q['model_examples'],replay=r['model_examples']);examples['total']=sum(examples.values())
    assert examples['total']==158336
    s=c.read(c.ROOT/'analysis/confirmation/summary.json');pred=c.read(c.ROOT/'analysis/PREDICTIONS.json');compare={}
    for key,stratum in s['strata'].items():
        compare[key]={name:dict(development=value,confirmation=stratum['ratios'][name]['estimate'],change=stratum['ratios'][name]['estimate']-value)
                      for name,value in pred['numeric_expectations'][key].items()}
    primary=['joint_necessity','joint_sufficiency','extra_necessity','extra_sufficiency']
    failures=[]
    for key,st in s['strata'].items():
        for name in primary:
            v=st['ratios'][name];point=.75 if name.startswith('joint') else .15;lower=.5 if name.startswith('joint') else 0.
            if not(v['interpretable'] and v['estimate']>=point and v['interval'][0]>lower):
                failures.append(dict(stratum=key,quantity=name,estimate=v['estimate'],interval=v['interval'],required_point=point,required_lower_bound=lower))
    c.write(output,dict(status='complete',created_utc=c.now(),study='pathway-interactions-v3',
        decision=dict(planned_followup_complete=True,strong_explanation_supported=s['strong_explanation_supported'],complete_mechanism_identified=False,
            prior_studies_unchanged=True,further_runs_running_or_scheduled=False,failed_confirmation_gates=[k for k,v in s['gates'].items() if not v],failed_primary_thresholds=failures,
            scope='Selection was frozen after a 32-case development screen. Confirmation tested the same two heads and one MLP together with known queries, both corruptions and both doses. Interpretations remain intervention-specific.'),
        unique_recipient_cases=1280,screen_cases_contained_in_development=32,recipient_panel_evaluations=sum(v['completed'] for v in phases.values()),source_pairs=96,
        condition_records={phase:v['condition_records'] for phase,v in phases.items()},total_condition_records=sum(v['condition_records'] for v in phases.values()),
        model_examples=examples,model_stage_seconds={**{k:v['seconds'] for k,v in phases.items()},'qualification':q['seconds'],'replay':r['seconds']},
        timing_scope='Measured execution stages, excluding design and CPU analysis; qualification and main panel timers start after model loading.',
        replay=dict(saved_panels=r['recipient_evaluations'],condition_records=r['condition_records'],source_pairs=r['unique_source_pairs'],bitwise_identical=r['bitwise_identical']),
        numerical=dict(qualification_max_error=q['maximum_error'],whole_path_controls_max_error=max(v.get('complete_path_max_error',0) for v in phases.values())),
        audit_status=audits,serialization_repair_preserved=True,selection=c.read(c.ROOT/'analysis/SELECTION.json'),confirmation_gates=s['gates'],
        development_confirmation_comparison=compare,source_sha256=c.sha(__file__)))
    print('Study closure accounted:',sum(v['condition_records'] for v in phases.values()),'records;',examples['total'],'model examples')


if __name__=='__main__':main()
