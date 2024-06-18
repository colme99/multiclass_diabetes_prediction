# Multiclass Diabetes Prediction

This repository contains the Master's thesis project in Bioinformatics and Biostatistics developed by Carlos Colmenero Gómez Cambronero at the Universitat Oberta de Catalunya and Universitat de Barcelona.

## Dataset Attribution
Within the `data` folder of this repository, you will find a subset of the observations from the "Mendeley Diabetes Dataset," which was utilized to test the predictive model described in this thesis. This dataset is not owned by the author of this thesis. Full rights to the dataset belong to:

**Author:** Rashid, A.  
**Publication Year:** 2020  
**DOI:** [10.17632/wj9rwkp9c2.1](http://www.doi.org/10.17632/wj9rwkp9c2.1)

Please cite the original source when using this dataset for any purpose.

## Code

Inside the `code` folder, the file `multiclass_diabetes_prediction.ipynb` is a Jupyter notebook with the complete analysis process, whereas the files `multiclass_diabetes_prediction_app.py` contains the whole implementation of the web application.

## Reproducibility

To reproduce the results obtained in the thesis, please install the specific libraries and versions indicated in `requirements.txt`.

## Hyperparameter ranges used in the optimization

SVC:
- **C** : [0.5, 200]
- **gamma** : [0.001, 1]
- **kernel**: *linear*, *poly*, *rbf*, or *sigmoid*
- **decision_function_shape**: *ovo* (one-vs-one), *ovr* (one-vs-rest)
- **degree**: [2, 6]


XGBoost:
- **max_depth**: [3, 10]
- **learning_rate**: [0.01, 0.3]
- **n_estimators**: [10, 1000]
- **subsample**: [0.5, 1.0]
- **colsample_bytree**: [0.5, 1.0]
- **gamma**: [0, 5]
- **min_child_weight**: [1, 10]
- **reg_alpha**: [0.0, 1.0]
- **reg_lambda**: [1.0, 10.0]


LightGBM:
- **max_depth**: [3, 10]
- **learning_rate**: [0.01, 0.3]
- **n_estimators**: [10, 1000]
- **subsample**: [0.5, 1.0]
- **colsample_bytree**: [0.5, 1.0]
- **num_leaves**: [5, 50]
- **min_split_gain**: [0.01, 0.3]
- **min_child_weight**: [1, 10]
- **reg_alpha**: [0.0, 1.0]
- **reg_lambda**: [1.0, 10.0]


AdaBoost:
- **n_estimators**: [10, 1000]
- **learning_rate**: [0.05, 3.0]
- **estimator_max_depth**: [3, 10]

Random Forest:
- **max_depth**: [3, 10]
- **n_estimators**: [10, 1000]
- **max_features**: *sqrt*, *log2*
- **min_samples_leaf**: [1, 10]
- **max_samples**: [0.5, 1.0]
- **criterion**: *gini*, *entropy*

Neural Network:
- **n_steps** (number of *steps* in the neural network): [1, 5]
- **n_units** (number of units in each layer): [5, 50]
- **lr** (learning rate for the Adam optimizer): [1e-4, 1e-2]
- **drop_out_rate** (rate to randomly turn off the output of neurons): [0, 0.2]
