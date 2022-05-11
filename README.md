# Explainable Fine-Grained Document Classification of Mathematical Documents

This manual provides descriptions to reproduce the results of the associated paper.

Document subject classification enables structuring (digital) libraries and allows readers to search for articles within a specific field.
Currently, the classification is typically provided by human domain experts.
Semi-supervised Machine or Deep Learning algorithms can support them by exploiting labeled data to predict subject classes of unclassified new documents.
However, these algorithms only work or yield useful results if the ratio of training examples per class is high.
In the case of mathematical documents, the commonly used Mathematical Subject Classification (MSC) leads to multiple challenges: The classification is 1) multi-label, 2) hierarchical, 3) fine-grained, and 4) sparsely populated with examples for the more than 5,000 classes.
In this paper, we address these challenges by using class-entity relations to enable multi-label hierarchical fine-grained category predictions for the first time while providing high explainability.
We examine relationships between fine-grained subject classes and keyword entities, mining a dataset from the zbMATH library https://zbmath.org.

<!--
This manual provides descriptions to reproduce the results for each subsection of the evaluation section of the associated paper. In this project, we address the information need of document subject category classification interpretability and explainability. Therefore, we analyze the relationships between categories (labels) and entities (features) of a document. We examine relationships between textual and mathematical subject classes and entities, mining a collection of documents from the arXiv preprint repository (NTCIR and zbMATH dataset). The results indicate that mathematical entities have the potential to provide high explainability as they are a crucial part of a STEM (Science, Technology, Engineering, and Mathematics) document.

## Evaluation Workflow

The figure below (paper Figure 1) illustrates the workflow of our experiments.

![EvaluationWorkflow](https://github.com/AnonymousCSResearcher/STEMdocClassExplainability/blob/main/Workflow_EL-ClassExplainability.png)

Entity Linking for both textual and mathematical entities and entity-category correspondence is examined as a prerequisite for classification entity explainability.
The python module dependencies (scikit-learn, LIME, SHAP, etc.) are specified in the header of the respective scripts. The numbering of the result figures and tables in the paper are referenced where they are relevant.
-->

## Requirements

Before executing the algorithms, it is necessary to install the python modules into your local virtual environment (venv) using the provided requirements.txt

## Fine-Grained QID and MSC Prediction

Data and algorithms can be found in the folder 'Fine-Grained-MSC-Class'.

The script
```
evaluation.py
```
contains all required steps in the data processing pipeline.

### 0) Load input table

After specifiying the
```
fullpath
```
of the dataset csv file, the
```
table = pd.read_csv(fullpath,delimiter=',')
```
can be read in using the python pandas module.

In our experiments, we set the parameter to
```
tot_rows = len(table)
train_split_rate = 0.7
nr_docs = int(tot_rows*train_split_rate)
```
which can be adapted.

### 1*) Dataset statistics

The dataset statistics are generated using
```
print_dataset_statistics(sorted_cls_ent_idx,sorted_ent_cls_idx)
```

### 1) Generate MSC-keyword mapping

First the MSC-keyword/keyword-MSC class-entity/entity-class (cls_ent) index can be created from the input table via
```
cls_ent_idx,ent_cls_idx = generate_msc_keyword_mapping(table,nr_docs)
```
and dumped to disk using
```
sorted_cls_ent_idx,sorted_ent_cls_idx = sort_and_save_index(cls_ent_idx,ent_cls_idx)
```
After being generated once, in subsequent script executions, the above line may be commented out and the index loaded via
```
sorted_cls_ent_idx,sorted_ent_cls_idx = load_index(outpath)
```

### 2) Predict MSCs

To predict the MSCs from the table, use
```
predict_text_mscs(table,n_gram_lengths)
```
The prediction table is saved to the specified
```
outpath + 'mscs_prediction_table.csv'
```

### 3) Evaluate MSC predictions

The core evaluation is done by
```
train_test_split(table,train_split_rate)
```
and
```
get_sparse_mscs(table)
```

<!--
Data and algorithms can be found in the folder 'ConceptClassSpaces'.

1) QID and MSC predictions can be made using
```
get_ConceptClassSpaces.py
```
2) The manual QID benchmark dataset (500 manually linked entities) is available in
```
evaluation/100docs/assessed/Math Entity Linking zbmath keywords evaluation_all.csv
```
3) The pywikibot/SPARQL predictions are evaluated agains the benchmark using
```
evaluate_MathEL_zbmath_keywords.py
```
4) MSC prediction is done by
```
predict_mscs.py
```
generating
```
mrmscs_dict.json
```
5)
After making a
```
train-test_split.py
```
the evaluation can be carried out using
```
keywords_vs_refs.py
```
outputting
```
keywords_vs_refs_mrmscs.csv
```
-->

## MSC-arXiv Category Correspondence

Data and algorithms can be found in the folder 'Classification'.

1) The data for the prediction of the MSC-arXiv category correspondence is the table in the file
```
msc_categories.csv
```

2) To get the co-ocurrence matrix with 6202 MSC rows and 156 arXiv category columns, run
```
get_cooccurrence_matrix.py
```
yielding 'coocurrence_matrix.csv'.

3) To get the uncertainty/entropy analysis results run
```
get_classification_uncertainties.py
```
or
```
get_classification_entropies.py
```
yielding 'uncertainties_arXiv.json' and 'uncertainties_MSCs.json'.

4) The multi-label prediction of arXiv categories from MSCs can be made with
```
get_multilabel_classification.py
```

5) The prediction of the categories using a LogReg classifier on TFIDF encodings (paper Table I) or the co-occurrence matrix can be made with
```
predict_categories_classifier.py
```
or
```
predict_categories_cooccurrence.py
```
respectively.

6) A comparison of the prediction results from 5) can be made running
```
compare_predictions_classifier_to_cooccurrence.py
```

## Identifier Class Semantics Distributions

Data and algorithms can be found in the folder 'IdentifierStatistics'.

0) The NTCIR-11/12 arXiv dataset can be obtained from
```
http://ntcir-math.nii.ac.jp/data
```

1) The identifier annotation index 'annotation_catalog_all.pkl' is created from the surrounding text of mathematical formula symbols in parallel multiprocesses running
```
extract_surroundingtext_multiprocess.py
```

2) The identifier statistics (paper Figure 2 and Table II) are extracted from the 'annotation_catalog_all.pkl' using
```
compare_predictions_classifier_to_cooccurrence.py
```

3) Visualization of the identifier symbol and name distributions can be made via
```
visualize_identifier_statistics.py
```

## Unsupervised Semantic Identifier Enrichment for Document Classification

Data and algorithms can be found in the folder 'Augmentations/IdentifierSemantification'.

0) For the creation of the formula identifier semantification catalog (paper Table III) see the 'Identifier Class Semantics Distributions' section with the script
```
extract_surroundingtext_multiprocess.py
```

1) The formula identifier catalog index can be analyzed for duplicates and encoded in TFID feature vectors using
```
analyze_formula_catalog.py
```

2) The encoding of the arXiv NTCIR-11/12 selection of 100 documents from each of the subject classes 'astro-ph', 'cond-mat', 'gr-qc', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'nlin', 'quant-ph', 'physics' is made via
```
arXiv12DocsText2Vec_augmented.py
```
for textual elements, and
```
arXiv12DocsMath2Vec_augmented.py
```
for mathematical elements (surrounding text of formulae).

3) The document classification is made and evaluated (paper Table IV) running
```
arXivClassification_semantified.py
```

## Category-Concept Augmentations

Data and algorithms can be found in the folder 'Augmentations'.

1) The data retrieved from https://en.wikipedia.org/w/index.php?title=Outline_of_physics&oldid=994270982 is condensed in
```
Wikipedia_Categories_Concepts_Data.py
```

2) The category concept augmentations (paper Table V) for the text classification is carried out using
```
Wikipedia_Categories_Concepts.py
```

## Wikisource Entity Linking (Wikification)

Data and algorithms can be found in the folder 'EntityLinking'.

0) The processed documents are in the folder
```
documents/with_msc
```

1) To get the class-entity index run
```
get_class_entity_index(_Wikipedia).py
```

2) The Wikipedia article name dump can be accessed via
```
WikiDump.py
```

3) The evaluation of the entity linking (paper Table VI for text comparing eval modes 1-6, and Table VII for math ranking formula concept n-grams) can be reproduced via
```
get_evaluation_entity_linking.py
```
with the binary scoring (TP, FP, FN, TN) made via
```
get_scoring_evaluation_entity_linking.py
```

4) An entropy index for 'text' or 'math' mode can be obtained via
```
get_index_entropies.py
```

## Class-Entity Explainability

Data and algorithms can be found in the folder 'Explainability'.

1) The most discriminative word features form the NTCIR-11/12 arXiv corpus (same selection of 100 documents from each of the subject classes 'astro-ph', 'cond-mat', 'gr-qc', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'nlin', 'quant-ph', 'physics') using a LIME or SHAP explainer can be identified using
```
get_most_discriminative_LIME.py
```
or
```
get_most_discriminative_SHAP.py
```
respectively

2) The most frequent word features are obtained using
```
get_most_frequent.py
```

3) Finally, both distributions (most discriminative, most frequent) are compared calculating distribution entropies (paper Table VIII) in
```
compare_most_frequent_vs_discriminative.py
```
