import pandas as pd

# Open table

evaluation_filename = 'zbmath keywords evaluation_all.csv'
table = pd.read_csv(evaluation_filename,delimiter=';')

# Retrieve scores
scores_sparql = table['Score SPARQL']
scores_pywikibot = table['Score Pywikibot']
score_benchmark = table['Score Benchmark']

# Calculate tp,fp,fn,tn
def get_binary_classification(scores,benchmark):
    results = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    for line_number in range(len(scores)):

        def normalize_score(score):
            try:
                score = int((int(score) > 0))
            except:
                if score == 'NaN':
                    score = 0
            return score

        score_value = normalize_score(scores[line_number])
        benchmark_value = normalize_score(benchmark[line_number])

        if score_value == benchmark_value:
            if score_value == 1:
                results['tp'] += 1
            elif score_value == 0:
                results['tn'] += 1
        elif score_value != benchmark_value:
            if score_value == 1:
                results['fp'] += 1
            elif score_value == 0:
                results['fn'] += 1

    return {'tp': results['tp'],'fp': results['fp'],'fn': results['fn'],'tn': results['tn']}

def get_precision_recall_tnr(binary_classification):

    bc = binary_classification

    precision = bc['tp']/(bc['tp'] + bc['fp'])
    recall = bc['tp']/(bc['tp'] + bc['fn'])
    tnr = bc['tn']/(bc['tn'] + bc['fp'])

    return {'precision': precision,'recall': recall,'tnr': tnr}

print()

# for SPARQL
binary_classification_sparql = get_binary_classification(scores_sparql,score_benchmark)
print('binary_classification_sparql: ' + str(binary_classification_sparql))
precision_recall_tnr_sparql = get_precision_recall_tnr(binary_classification_sparql)
print('precision_recall_tnr_sparql: ' + str(precision_recall_tnr_sparql))

print()

# for Pywikibot
binary_classification_pywikibot = get_binary_classification(scores_pywikibot,score_benchmark)
print('binary_classification_pywikibot: ' + str(binary_classification_pywikibot))
precision_recall_tnr_pywikibot = get_precision_recall_tnr(binary_classification_pywikibot)
print('precision_recall_tnr_pywikibot: ' + str(precision_recall_tnr_pywikibot))

print('end')