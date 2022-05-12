# Explainable Fine-Grained Document Classification of Mathematical Documents

This manual provides descriptions to reproduce the results of the associated paper.

Document subject classification enables structuring (digital) libraries and allows readers to search for articles within a specific field.
Currently, the classification is typically provided by human domain experts.
Semi-supervised Machine or Deep Learning algorithms can support them by exploiting labeled data to predict subject classes of unclassified new documents.
However, these algorithms only work or yield useful results if the ratio of training examples per class is high.
In the case of mathematical documents, the commonly used Mathematical Subject Classification (MSC) leads to multiple challenges: The classification is 1) multi-label, 2) hierarchical, 3) fine-grained, and 4) sparsely populated with examples for the more than 5,000 classes.
In this paper, we address these challenges by using class-entity relations to enable multi-label hierarchical fine-grained category predictions for the first time while providing high explainability.
We examine relationships between fine-grained subject classes and keyword entities, mining a dataset from the zbMATH library https://zbmath.org.

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

### 1*) Index statistics

The index statistics are generated using
```
print_index_statistics(sorted_cls_ent_idx,sorted_ent_cls_idx)
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

The train-test-split is generated using
```
train_test_split(table,train_split_rate)
```
To display sparsely populated MSCs, run
```
get_sparse_mscs(table)
```
Finally, the evaluation in comparison to the MR-MSCs and keywords baseline is made by
```
compare_mr_keyword_refs_dcgs(table)
```

## Wikisource Entity Linking (Wikification)

Data and algorithms can be found in the folder 'EntityLinking'.

### 0) Documents dataset

The processed documents are in the folder
```
documents/with_msc
```

### 1) Class-entity index

To get the class-entity index run
```
get_class_entity_index(_Wikipedia).py
```

### 2) Wikipedia article name dump

The Wikipedia article name dump can be downloaded at
```
http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz
```
adn accessed via
```
WikiDump.py
```

### 3) Scoring evaluation

The evaluation of the entity linking (paper Table VI for text comparing eval modes 1-6, and Table VII for math ranking formula concept n-grams) can be reproduced via
```
get_evaluation_entity_linking.py
```
with the binary scoring (TP, FP, FN, TN) made via
```
get_scoring_evaluation_entity_linking.py
```

### 4) Entropies

An entropy index for 'text' or 'math' mode can be obtained via
```
get_index_entropies.py
```
