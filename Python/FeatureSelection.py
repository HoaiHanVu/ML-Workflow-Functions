import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif, SelectFromModel
from sklearn.inspection import permutation_importance
warnings.filterwarnings('ignore')


def fit_model(model, X, y):
    
    trained_model = model.fit(X, y)
    
    return trained_model


def calculate_metrics_reg(model, X, y):
    
    y_pred = model.predict(y)
    scoring = model.score(X, y)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    return scoring, mse, mae

def calculate_metrics_clf(trained_model, X, y):
    
    y_pred = trained_model.predict(X)
    acc = accuracy_score(y, y_pred)
    roc = roc_auc_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    return acc, roc, prec, rec, f1

def train_get_metrics_reg(model, X_train, y_train, X_test, y_test):
    
    trained_model = fit_model(model, X_train, y_train)
    scoring, mse, mae = calculate_metrics_reg(trained_model, X_test, y_test)
    
    return scoring, mse, mae

def train_get_metrics_clf(model, X_train, y_train, X_test, y_test):
    
    trained_model = fit_model(model, X_train, y_train)
    acc, roc, prec, rec, f1 = calculate_metrics_clf(trained_model, X_test, y_test)
    
    return acc, roc, prec, rec, f1

def eval_reg_model_on_features(model, X_train, y_train, X_test, y_test):
    
    scoring, mse, mae = train_get_metrics_reg(model, X_train, y_train, X_test, y_test)
    display_df = pd.DataFrame([[scoring, mse, mae, X_train.shape[1]]],
                              columns = ['Accuracy', 'MSE', 'MAE', 'Feature Count'])
    
    return display_df

def eval_clf_model_on_features(model, X_train, y_train, X_test, y_test):
    
    acc, roc, prec, rec, f1 = train_get_metrics_clf(model, X_train, y_train, X_test, y_test)
    display_df = pd.DataFrame([[acc, roc, prec, rec, f1, X_train.shape[1]]],
                              columns = ['Accuracy', 'ROC', 'Precision', 'Recall', 'F1-score', 'Feature Count'])
    
    return display_df




# Filter Method
###  Univariate Selection

def univariate_selection(X_train, y_train, model_type, k):
    
    selector = SelectKBest(model_type, k = k)
    X_new = selector.fit_transform(X_train, y_train)
    feature_idx = selector.get_support()
    features_name = X_train.columns[feature_idx]
    
    return features_name



# Wrapper Method
### Recursive feature elimination - RFE

def run_rfe(model, X_train, y_train, n):
    
    rfe = RFE(model, n_features_to_select = n)
    rfe = rfe.fit(X_train, y_train)
    features_name = X_train.columns[rfe.get_support()]
    
    return features_name


# Embedded Method
### Feature Importance
def tree_based_feature_importances(trained_model, X_train, plot = True):
    
    imp_df = pd.DataFrame({'Feature': X_train.columns,
                          'Importances': trained_model.feature_importances_}.sort_values('Importances', 
                                                                                         ascending = False))
    if plot:
        plt.figure(figsize=(10, 6))
        sb.barplot(data = rf_imp, x = 'Importances', y = 'Features')
        plt.title("Feature Importances (train set)")
        plt.show()
        
    return imp_df

def permutation_importance_features(trained_model, X, y, n, st, plot = True):
    
    result = permutation_importance(trained_model, X, y, n_repeats = 10, random_state = st)
    imp_order = result.importances_mean.argsort()
    ft_importance = pd.DataFrame(result.importances[imp_order].T,
                                 columns = X.columns[imp_order])
    imp_df = ft_importance[ft_importance.columns[::-1]]
    if plot:
        plt.figure(figsize = (10, 6))
        ax = sb.boxplot(data = imp_df, orient = 'h')
        ax.set_title("Permutation Importances")
        ax.axvline(x=0, color="k", linestyle="--")
        ax.set_xlabel("Decrease in accuracy score")
        ax.figure.tight_layout()
        plt.show()
        
    return imp_df

def run_l1_regularization(model, X_train, y_train):
    
    # Select L1 regulated features from LinearSVC output 
    selection = SelectFromModel(model)
    selection.fit(X_train, y_train)

    feature_names = X_train.columns[(selection.get_support())]
    
    return feature_names