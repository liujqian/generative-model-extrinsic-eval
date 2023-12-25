import rouge

if __name__ == '__main__':
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score("The player stood in the field looking at the batter.",
                          'A man stands in a field looking at a field.')
    print(scores)
    # def prepare_results(m, p, r, f):
    #     return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)
    #
    #
    # for aggregator in ['Avg', 'Best', 'Individual']:
    #     print('Evaluation with {}'.format(aggregator))
    #     apply_avg = aggregator == 'Avg'
    #     apply_best = aggregator == 'Best'
    #
    #     evaluator = rouge.Rouge(
    #         metrics=['rouge-l', ],
    #         max_n=4,
    #         limit_length=True,
    #         length_limit=100,
    #         length_limit_type='words',
    #         apply_avg=apply_avg,
    #         apply_best=apply_best,
    #         alpha=0.5,  # Default F1_score
    #         weight_factor=1.2,
    #         stemming=True
    #     )
    #
    #     hypothesis_1 = "A man is standing in a field looking at the camera."
    #     references_1 = [
    #         "The player stood in the field looking at the batter.",
    #         # "The coach stands along the field, looking at the goalkeeper.",
    #         # "I stood and looked across the field, peacefully.",
    #         # "Someone stands, looking around the empty field."
    #     ]
    #     all_hypothesis = [hypothesis_1]
    #     all_references = [references_1]
    #
    #     scores = evaluator.get_scores(all_hypothesis, all_references)
    #
    #     for metric, results in sorted(scores.items(), key=lambda x: x[0]):
    #         if not apply_avg and not apply_best:  # value is a type of list as we evaluate each summary vs each reference
    #             for hypothesis_id, results_per_ref in enumerate(results):
    #                 nb_references = len(results_per_ref['p'])
    #                 for reference_id in range(nb_references):
    #                     print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
    #                     print('\t' + prepare_results(metric, results_per_ref['p'][reference_id], results_per_ref['r'][reference_id],
    #                                                  results_per_ref['f'][reference_id]))
    #             print()
    #         else:
    #             print(prepare_results(metric, results['p'], results['r'], results['f']))
    #     print()
