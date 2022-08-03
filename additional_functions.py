# Data analysis and calculation
import pandas as pd
import numpy as np
import additional_functions

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.features import FeatureImportances
from yellowbrick.classifier import ROCAUC

# Statistical modules
import statsmodels.stats.api as sms
from scipy.stats import ttest_ind
import collections



def dist_plot(
    column_1: str, column_2: str, x_name: str, label_1: str, label_2: str
) -> None:
    """
    Takes labels of two selected columns.
    Selects rows with only a certain value and from certain columns.
    Returns graphs of the distribution of values with a specific x-axis name.
    """
    sns.set_style("ticks")
    fig, ax1 = plt.subplots(1, figsize=(16, 14))

    sns.kdeplot(
        stroke_eda.loc[(stroke_eda[column_2] == 1), column_1],
        color="seagreen",
        shade=True,
        Label=label_1,
        ax=ax1,
    )

    sns.kdeplot(
        stroke_eda.loc[(stroke_eda[column_2] == 0), column_1],
        color="darkorchid",
        shade=True,
        Label=label_2,
        ax=ax1,
    )

    ax1.set_xlabel(x_name, fontsize=14)
    ax1.set_ylabel("Density", fontsize=14)
    sns.despine()
    plt.legend(title_fontsize=13)
    plt.legend(loc="upper left")
    plt.show()


def bar_plot(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    """
    Takes labels of two seleceted columns.
    Returns a bar chart with annotated bars and hidden x and y axes.
    """

    sns.set(rc={"figure.figsize": (16, 14)})
    sns.set_style("white")

    splot = sns.barplot(x=x, y=y, data=df, palette="PRGn")

    for p in splot.patches:
        splot.annotate(
            format(p.get_height(), ".1f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=14,
        )
    widthbars = [0.6, 0.6]
    for bar, newwidth in zip(splot.patches, widthbars):
        x = bar.get_x()
        width = bar.get_width()
        centre = x + width / 2.0
        bar.set_x(centre - newwidth / 2.0)
        bar.set_width(newwidth)

    plt.tick_params(axis="both", which="both", length=0)
    plt.yticks([])
    plt.ylabel("")
    plt.xlabel("")
    plt.title(title, fontsize=18)
    plt.tick_params(labelsize=14)
    sns.despine(bottom=True, left=True)
    plt.show()


def swarm_plot(
    df: pd.DataFrame, column_1: str, column_2: str, x_name: str, y_name: str, title: str
) -> None:
    """
    Takes data frame and labels of two selected columns.
    Returns swarmplot of values with a specific x-axis and y-axis name.
    """
    sns.set(rc={"figure.figsize": (16, 14)})
    sns.set_style("ticks")

    ax = sns.swarmplot(x=column_1, y=column_2, data=df, palette="PRGn")

    sns.despine()
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(x_name, fontsize=14)
    ax.set_ylabel(y_name, fontsize=14)
    plt.show()


def bar_plot2(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    hue: str,
    legend_lab: list,
    legend_title: str,
) -> None:
    """
    Takes labels of two seleceted columns.
    Returns a bar chart with annotated bars and hidden x and y axes.
    """

    sns.set(rc={"figure.figsize": (16, 14)})
    sns.set_style("white")

    splot = sns.barplot(x=x, y=y, hue=hue, data=df, palette="PRGn")

    for p in splot.patches:
        splot.annotate(
            format(p.get_height(), ".1f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=14,
        )

    plt.tick_params(axis="both", which="both", length=0)
    plt.title(title, fontsize=18)
    plt.yticks([])
    plt.ylabel("")
    plt.xlabel("")

    plt.tick_params(labelsize=14)
    sns.despine(bottom=True, left=True)
    h, l = splot.get_legend_handles_labels()
    splot.legend(h, legend_lab, title=legend_title)
    plt.show(splot)


def base_line(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """
    Takes x and y dataframes.
    Perform cross validation with different models.
    Returns table with metrics and results
    """
    roc_auc = []
    accuracy = []
    recall = []
    precision = []
    f1_score = []

    kfold = StratifiedKFold(n_splits=5)
    classifiers = ["XGB classifier", "Random Forest", "Cat Boost", "LGBM classifier"]

    models = [
        XGBClassifier(),
        RandomForestClassifier(n_estimators=100),
        CatBoostClassifier(),
        LGBMClassifier(),
    ]

    for model in models:
        pipeline = ImPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("resample", resample),
                ("classifier", model),
            ]
        )
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=kfold,
            scoring=(
                "balanced_accuracy",
                "accuracy",
                "f1_macro",
                "recall_macro",
                "precision_macro",
                "roc_auc",
            ),
        )
        accuracy.append(result["test_accuracy"].mean())
        recall.append(result["test_recall_macro"].mean())
        precision.append(result["test_precision_macro"].mean())
        f1_score.append(result["test_f1_macro"].mean())
        roc_auc.append(result["test_roc_auc"].mean())

    base_models = pd.DataFrame(
        {
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "f1": f1_score,
            "Roc Auc": roc_auc,
        },
        index=classifiers,
    )
    base_models = base_models.style.background_gradient(cmap="PiYG_r")

    return base_models


def plot_cm(
    model: list, X_test: list, y_test: list, display_labels: str, cmap: None, title: str
) -> None:
    """
    Takes, list of x and y values, label, color and title names.
    Returns confusion  matrix plot
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.rc("font", size=12)
    plot_confusion_matrix(
        model,
        X_test,
        y_test,
        display_labels=display_labels,
        cmap=cmap,
        ax=ax,
        values_format="",
    )
    plt.title(
        title,
        fontsize=13,
        y=1.03,
    )
    plt.grid(False)
    ax.tick_params(labelsize=10)
    plt.ylabel("True label", fontsize=10)
    plt.xlabel("Predicted label", fontsize=10)
    plt.show()


def LGBM_objective(trial, X, y) -> dict:
    """
    Takes x and y and dataframes
    Performs model training.
    Returns best parameters for particular model.
    """
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        model = LGBMClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            callbacks=[
                optuna.integration.LightGBMPruningCallback(trial, "binary_logloss")
            ],
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

        return np.mean(cv_scores)


def xgboost_objective(trial, X, y) -> dict:
    """
    Takes x and y and dataframes
    Performs model training.
    Returns best parameters for particular model.
    """

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 100),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 100),
        "gama": trial.suggest_float("gama", 0, 20),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 100),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = np.empty(5)

    for idx, (train_idx, eval_idx) in enumerate(cv.split(X, y)):
        X_train, X_eval = X.iloc[train_idx], X.iloc[eval_idx]
        y_train, y_eval = y.iloc[train_idx], y.iloc[eval_idx]

        X_train = preprocessor.fit_transform(X_train)
        X_eval = preprocessor.transform(X_eval)

        model = XGBClassifier(use_label_encoder=0, **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            eval_metric="logloss",
            early_stopping_rounds=100,
            verbose=False,
        )
        preds = model.predict(X_eval)
        cv_scores[idx] = f1_score(y_eval, preds, average="macro")

    return np.mean(cv_scores)


def forest_objective(trial, X, y) -> dict:
    """
    Takes x and y and dataframes
    Performs model training.
    Returns best parameters for particular model.
    """

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_split": trial.suggest_int("max_depth", 1, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["auto", "sqrt", "log2"]
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = np.empty(5)

    for idx, (train_idx, eval_idx) in enumerate(cv.split(X, y)):
        X_train, X_eval = X.iloc[train_idx], X.iloc[eval_idx]
        y_train, y_eval = y.iloc[train_idx], y.iloc[eval_idx]

        X_train = preprocessor.fit_transform(X_train)
        X_eval = preprocessor.transform(X_eval)

        model = RandomForestClassifier(**param_grid)
        model.fit(
            X_train,
            y_train,
        )
        preds = model.predict(X_eval)
        cv_scores[idx] = f1_score(y_eval, preds, average="macro")

    return np.mean(cv_scores)


def call_conf_matrix(lgbm, xgb, forest):
    """
    Takes three models and predicts outcomes.
    Results are passed to confusion_matrix function
    And this function returns three confusion matrices subplots
    """

    y_pred_lgbm = lgbm.predict(X_test)
    y_pred_xgbm = xgb.predict(X_test)
    y_pred_tree = forest.predict(X_test)

    lgbm_cf = confusion_matrix(y_test, y_pred_lgbm)
    xgbm_cf = confusion_matrix(y_test, y_pred_xgbm)
    tree_cf = confusion_matrix(y_test, y_pred_tree)

    fig, ax = plt.subplots(1, 3, figsize=(20, 8))

    sns.heatmap(lgbm_cf, ax=ax[0], annot=True, cmap="PRGn", fmt="g")
    ax[0].set_title("LGBM \n Confusion Matrix", fontsize=14)

    sns.heatmap(xgbm_cf, ax=ax[1], annot=True, cmap="PRGn", fmt="g")
    ax[1].set_title("XGBM \n Confusion Matrix", fontsize=14)

    sns.heatmap(tree_cf, ax=ax[2], annot=True, cmap="PRGn", fmt="g")
    ax[2].set_title("Random Forest \n Confusion Matrix", fontsize=14)
    plt.show()


def LGBM_objective_multi(trial, X, y) -> dict:
    """
    Takes x and y and dataframes
    Performs model training.
    Returns best parameters for particular multiclass model.
    """

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        model = LGBMClassifier(objective="multiclass", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="multi_logloss",
            early_stopping_rounds=100,
            callbacks=[
                optuna.integration.LightGBMPruningCallback(trial, "multi_logloss")
            ],
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

        return np.mean(cv_scores)


def xgboost_objective_multi(trial, X, y):
    """
    Takes x and y and dataframes
    Performs model training.
    Returns best parameters for particular multiclass model.
    """

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 100),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 100),
        "gama": trial.suggest_float("gama", 0, 20),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 100),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = np.empty(5)

    for idx, (train_idx, eval_idx) in enumerate(cv.split(X, y)):
        X_train, X_eval = X.iloc[train_idx], X.iloc[eval_idx]
        y_train, y_eval = y.iloc[train_idx], y.iloc[eval_idx]

        X_train = preprocessor.fit_transform(X_train)
        X_eval = preprocessor.transform(X_eval)

        model = XGBClassifier(**param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            early_stopping_rounds=100,
            verbose=False,
        )
        preds = model.predict(X_eval)
        cv_scores[idx] = f1_score(y_eval, preds, average="micro")

    return np.mean(cv_scores)


def forest_objective_multi(trial, X, y) -> dict:
    """
    Takes x and y and dataframes
    Performs model training.
    Returns best parameters for particular multiclass model.
    """

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_split": trial.suggest_int("max_depth", 1, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["auto", "sqrt", "log2"]
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = np.empty(5)

    for idx, (train_idx, eval_idx) in enumerate(cv.split(X, y)):
        X_train, X_eval = X.iloc[train_idx], X.iloc[eval_idx]
        y_train, y_eval = y.iloc[train_idx], y.iloc[eval_idx]

        X_train = preprocessor.fit_transform(X_train)
        X_eval = preprocessor.transform(X_eval)

        model = RandomForestClassifier(**param_grid)
        model.fit(
            X_train,
            y_train,
        )
        preds = model.predict(X_eval)
        cv_scores[idx] = f1_score(y_eval, preds, average="micro")

    return np.mean(cv_scores)


def feature_names(module) -> list:
    """
    Takes trainde model.
    Extracts and returns feature name from preprocesseor.
    """

    cat = list(
        module.named_steps["preprocessor"]
        .transformers_[1][1]
        .named_steps["encoder"]
        .get_feature_names(categorical_features)
    )

    cat_all = ["age", "avg_glucose_level", "bmi"] + cat

    return cat_all