# MAIN

# INSTALLATION
# copy dataset_propensity.json to 'data' folder
# conda create -n nzz python=3.10
# conda env remove --name nzz
# pip install pandas
# pip isntall matplotlib
# pip isntall seaborn
# pip isntall plotly
# pip install nbformat
# pip isntall xgboost
# pip install shap

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from xgboost import XGBClassifier
import shap

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 100)

import os

os.getcwd()
os.listdir()

### DATA
df = pd.read_csv("../data/data_clean.csv")
print(df.shape)
print(df.columns)

### PREPARATION

# Splitting
df["buy"].value_counts()

X = df.drop(columns=["buy"])
y = df.buy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

for column in df.columns:
    if df[column].dtype == "object":
        print(f"Column: {column}")
        print(df[column].unique())
        print("\n")

# Preprocessing
column_trans = make_column_transformer(
    (
        OneHotEncoder(handle_unknown="ignore"),
        ["langNew", "mostLikedCategories", "operatingSystemNew", "preferredTimeOfDay"],
    ),  # non-numeric
    remainder="passthrough",
)

# Scaling
scaler = StandardScaler()

### MODEL


def print_classification_metrics(
    model, column_trans, scaler, X_train, X_test, y_train, y_test
):
    pipe = make_pipeline(column_trans, scaler, model)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)

    model_name = str(model).split("(")[0]
    if "Regression" in model_name:
        model_name = model_name.replace("Regression", "")

    print(f"{model_name} classification model:")
    print(f"Accuracy: {accuracy:.4f}")
    print(
        f"Precision: {precision:.4f}"
    )  # Precision that the marketing efforts are not wasted on users who are unlikely to buy
    print(
        f"Recall: {recall:.4f}"
    )  # Recall is crucial because missing out on potential subscribers could mean lost revenue opportunities.
    print(
        f"F1 Score: {f1:.4f}"
    )  # useful if you need to balance the importance of identifying buyers accurately (recall) and ensuring the predictions are correct (precision)
    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )  # Receiver Operating Characteristic is a metric for evaluating the performance of a classification model (not so useful we have inbalance data)
    print(
        f"PR-AUC: {pr_auc:.4f}"
    )  # Precision-Recall Area Under Curve to identify buyers across different decision thresholds
    print("Confusion Matrix:")
    print(cm)
    #                      Predicted Negative   Predicted Positive
    # Actual Negative          TN                   FP
    # Actual Positive          FN                   TP

    return pipe


### LOGISTIC REGRESSION
log_model = LogisticRegression(class_weight="balanced")
pipe = print_classification_metrics(
    log_model, column_trans, scaler, X_train, X_test, y_train, y_test
)

### XGBOOST CLASSIFICATION
xgb_model = XGBClassifier(
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    use_label_encoder=False,
    eval_metric="logloss",
)
pipe = print_classification_metrics(
    xgb_model, column_trans, scaler, X_train, X_test, y_train, y_test
)

### IMPORTANCE
# Get all features
xgb_model_trained = pipe.named_steps["xgbclassifier"]
importances = xgb_model_trained.feature_importances_
ohe = pipe.named_steps["columntransformer"].named_transformers_["onehotencoder"]
ohe_features = ohe.get_feature_names_out(
    ["langNew", "mostLikedCategories", "operatingSystemNew", "preferredTimeOfDay"]
)
all_features = list(ohe_features) + list(X.select_dtypes(include=[int, float]).columns)

feature_importances = pd.DataFrame({"feature": all_features, "importance": importances})
feature_importances = feature_importances.sort_values(by="importance", ascending=False)
print(feature_importances)

# Variable explenation
explainer = shap.TreeExplainer(xgb_model_trained)
X_test_transformed = pipe.named_steps["columntransformer"].transform(X_test)
X_test_transformed = pipe.named_steps["standardscaler"].transform(X_test_transformed)

shap_values = explainer.shap_values(X_test_transformed)

shap.summary_plot(
    shap_values, X_test_transformed, feature_names=all_features, max_display=30
)

# Features on the Y-axis: The list of features on the Y-axis shows the most important features at the top. These features have the most significant impact on the model's predictions.
# SHAP values on the X-axis: The X-axis represents the SHAP values, which show the impact of each feature on the model's output. A positive SHAP value indicates that the feature contributes positively towards predicting the target variable (e.g., a higher probability of a user buying), while a negative SHAP value indicates a negative contribution.
# The colors represent the feature values. Red represents higher values of the feature, and blue represents lower values. This helps to understand how the value of a feature influences its impact on the prediction.
# Each dot in the plot represents a single prediction from the test dataset. The spread of dots shows the distribution of SHAP values for that feature across all predictions.

# Understanding the Factors:
# nzz: This feature has a high impact on the model's predictions, with both high and low values (red and blue dots) influencing the SHAP values significantly (long shape).
# num_read_articles: Higher values (red dots) of this feature tend to move the SHAP values to the left, indicating that users who read more articles are less likely to buy. Conversely, lower values (blue dots) have a neutral or slightly positive impact on the prediction, contributing to a higher likelihood of buying.
# number_of_newsletters: This feature also has a significant impact, with higher values increasing the probability of buying.
# contentTypeArticles: The impact of this feature varies, but higher values tend to have a positive contribution.
# time_from_last_session: Lower values (blue dots) tend to have a positive impact on the prediction, suggesting that users who have recently visited are likely to buy.
# days_since_registration: Higher values have a positive impact, meaning users who registered a long time ago are likely to buy.
# other: This feature shows a mixed impact with both high (red) and low (blue) values affecting the prediction differently.
# mostLikedCategories_NotAvailable: This categorical feature shows that if the preferred category is not available, it negatively affects the likelihood of buying.
# num_display_articles: The number of display articles also has a variable impact, with higher values contributing both positively and negatively in different contexts.

# Model Improvements:
