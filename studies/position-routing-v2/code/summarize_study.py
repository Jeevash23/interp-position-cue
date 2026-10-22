"""Closure accounting after the frozen experiments and independent audits finish."""
import core as c


def main():
    output = c.ROOT / 'analysis/STUDY-SUMMARY.json'
    assert not output.exists()
    bundle = c.read(c.ROOT / 'data/inputs.json')
    qualification = c.read(c.ROOT / 'runs/qualification/summary.json')
    replay = c.read(c.ROOT / 'runs/replay/summary.json')
    panels = {bank: c.read(c.ROOT / f'runs/{bank}/status.json')
              for bank in ['development', 'confirmation']}
    results = {bank: c.read(c.ROOT / f'analysis/{bank}/summary.json')
               for bank in panels}
    audits = {name: c.read(c.ROOT / f'analysis/{name}.json')
              for name in ['INPUT-AUDIT', 'EVIDENCE-AUDIT', 'ANALYSIS-REPLAY']}
    assert qualification['status'] == 'passed'
    assert replay['status'] == 'passed' and replay['bitwise_identical']
    assert all(a['status'] == 'passed' for a in audits.values())
    assert all(s['status'] == 'complete' and s['completed'] == s['planned']
               for s in panels.values())
    examples = dict(qualification=qualification['model_examples'],
                    **{bank: s['model_examples'] for bank, s in panels.items()},
                    exact_model_replay=replay['model_examples'])
    examples['total'] = sum(examples.values())
    assert examples['total'] == 40256
    assert len(bundle['recipients']) == 640 and len(bundle['sources']) == 96
    assert sum(s['condition_records'] for s in panels.values()) == 28800
    confirm = results['confirmation']
    assert not confirm['bounded_explanation_supported']
    assert confirm['gates']['numerical'] and confirm['gates']['competence']
    assert not confirm['gates']['primary_ratios'] and not confirm['gates']['query_specificity']
    positive = {kind: sum(v > 0 for v in confirm['absolute_contrasts']['same'][kind])
                for kind in ['name_means', 'context_means']}
    failure_details = []
    for name, threshold in [('query_necessity', .5), ('query_sufficiency', .5),
                            ('sufficiency_vs_rotation', .25)]:
        r = confirm['ratios'][name]
        assert r['estimate'] < threshold
        failure_details.append(dict(quantity=name, estimate=r['estimate'],
                                    required_point_estimate=threshold, interval=r['interval']))
    summary = dict(
        status='complete', created_utc=c.now(), study='position-routing-v2',
        source_sha256=c.sha(__file__),
        decision=dict(
            planned_followup_complete=True, strong_joint_criterion_passed=False,
            partial_causal_evidence=True, complete_mechanism_identified=False,
            claim='The order-dependent correction transfers across disjoint names. Final-position queries in five fixed literature heads causally contribute, but do not meet the predeclared strong-explanation thresholds.',
            interpretation='Ratios describe separate interventions and are not additive or uniquely identified shares of a complete mechanism.',
            failed_thresholds=failure_details, prior_fixed_signal_study_conclusion_unchanged=True,
            further_runs_started_or_scheduled=False),
        recipients=len(bundle['recipients']), sources=len(bundle['sources']),
        recipient_unique_clean_texts=len({r['clean_text'] for r in bundle['recipients']}),
        recipient_unique_corrupted_texts=len({r['corrupted_text'] for r in bundle['recipients']}),
        total_condition_records=28800,
        panels={bank: dict(recipients=r['recipients'], sources=r['sources'],
                           condition_records=r['condition_records']) for bank, r in results.items()},
        confirmation_independent_axes=dict(paired_name_units=8, lexical_contexts=8,
                                           within_cell_cases=8, fresh_names=48),
        positive_confirmation_order_contrasts=positive,
        model_examples=examples,
        model_stage_seconds=dict(qualification=qualification['seconds'],
                                 **{bank: s['seconds'] for bank, s in panels.items()},
                                 replay=replay['seconds']),
        model_stage_seconds_total=qualification['seconds'] + replay['seconds'] + sum(s['seconds'] for s in panels.values()),
        time_scope='Recorded stage times include their saved-output work. Qualification starts timing after model loading; other stages include loading. These are not total project wall time.',
        confirmation_ratios=confirm['ratios'], confirmation_gates=confirm['gates'],
        audit_status={name: a['status'] for name, a in audits.items()},
        numerical=dict(qualification='passed', independent_zero_writes_max_error=0.0,
                       model_replay_bitwise_identical=True, model_replay_recipients=replay['recipient_cases'],
                       model_replay_conditions=replay['condition_records'],
                       model_replay_sources=replay['source_pairs_reextracted'],
                       maximum_model_replay_error=replay['maximum_coordinate_error']),
        resources=dict(gpu='[6 GB NVIDIA GPU]',
                       peak_primary_gpu_allocated_bytes=max(s['peak_gpu_bytes'] for s in panels.values()),
                       peak_recorded_primary_rss_kib=max(s['maximum_rss_kib'] for s in panels.values()),
                       initial_resource_abort_model_examples=0),
        accounting='Physical model-example evaluations include padding, clean baselines, qualification and replay. The 28,800 primary condition records exclude repeated audit evaluations. No training occurred.')
    c.write(output, summary)
    print('Completed causal follow-up:', summary['recipients'], 'recipients;',
          summary['total_condition_records'], 'conditions;', examples['total'], 'model examples')


if __name__ == '__main__':
    main()
