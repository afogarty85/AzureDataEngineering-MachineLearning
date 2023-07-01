from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.special import expit, logit
import pandas as pd
import shap
np.random.seed(0)

# gen data
X, y = make_classification(n_samples=100000, n_features=50, n_informative=4, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# create data sets
xg_train = xgb.DMatrix(data=X_train, label=y_train)
xg_test = xgb.DMatrix(data=X_test, label=y_test)

# gen model
params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    'tree_method': 'gpu_hist',
    "eval_metric": "logloss"
}

# fit
bst = xgb.train(params=params,
                dtrain=xg_train,
                num_boost_round=5000,
                verbose_eval=125,
                evals=[(xg_train, 'xg_train'), (xg_test, 'xg_test')],
                early_stopping_rounds=100)


# init explainer
explainer = shap.explainers.GPUTree(model=bst,
                                    feature_perturbation='tree_path_dependent',
                                    )

# explain data
shap_values = explainer.shap_values(X=X_test, check_additivity=False)

# average SHAP value
avg_shap_values = np.abs(shap_values).mean(0)

# transform to dict / sort
avg_shap_by_col_dict = dict(zip( [f'col_{i}' for i in range(0, 50)], np.abs(shap_values).mean(0)))
sorted(avg_shap_by_col_dict.items(), key=lambda x: x[1], reverse=True)[:5]

# matches this plot
shap.summary_plot(shap_values=shap_values,
                  features=X_test,
                  plot_type="bar")

# The model's raw prediction for the first observation.
preds = bst.predict(data=xg_test,
                    output_margin=True,  # logits if True
                        )

# get first predicted val
first_row_logit = preds[0]

# the corresponding sum of the mean + shap values -- yields logit that needs sigmoid e.g., scipy.special.expit(val)
shap_logit = explainer.expected_value + shap_values[0].sum()
np.isclose(first_row_logit, shap_logit, atol=1e-2)


# ML model predicted probabilities can be expressed as the sum of shap values over its features, plus its expected value:
np.allclose( shap_values.sum(axis=1) + explainer.expected_value,
             bst.predict(data=xg_test,
                        output_margin=True,  # logits if True
                        ),
                        atol=1e-2
             )

# generate class labels
pred_label = np.round(expit(shap_values.sum(1) + explainer.expected_value))

# cutoff point is zero log odds; which is probability 0.5; or we can use the above logic np.round(expit(...))
y_pred = (shap_values.sum(axis=1) + explainer.expected_value) > 0
misclassified = (pred_label != y_test)

# find misclassified idx
np.where(misclassified == True)

# show labels
print(misclassified[81:83])

# plot a subset of data; dotted lines are misclassified
shap.decision_plot(base_value=explainer.expected_value,
                   shap_values=shap_values[81:83, :],
                   feature_names=[f'col_{i}' for i in range(0, 50)],
                   link='logit',  # convert base + shap logit to probabilities
                   highlight=misclassified[81:83])


# plot single misclassification -- true dataframe values displayed
shap.decision_plot(base_value=explainer.expected_value,
                   shap_values=shap_values[44, :],
                   features=X_test,
                   link='logit',  # convert base + shap logit to probabilities
                   highlight=misclassified[44])