import string
import regex
import demoji
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_theme()
import scipy.stats as st
from underthesea import word_tokenize, pos_tag, sent_tokenize
from sklearn.model_selection import train_test_split, cross_val_score
from pyvi import ViPosTagger, ViTokenizer


#Reduce Memory Usage
def reduce_memory_usage(df):
    
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype.name
        if ((col_type != 'datetime64[ns]') & (col_type != 'category')):
            if (col_type != 'object'):
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        pass
            else:
                df[col] = df[col].astype('category')
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage became: ",mem_usg," MB")
    
    return df


# Hàm lọc các cột có kiểu dữ liệu continuous
def numbers_variable(frame):
    numbers = [col for col in frame.columns if frame.dtypes[col] != object]
    return numbers

# Hàm hiển thị dữ liệu unique của các cột continuous
def display_numbers(frame, lst_numbers):
    for index, num in enumerate(lst_numbers):
        print('{}. Name var: {}, Numbers of unique: {}, Unique Value: {}'
          .format(index + 1, num, len(frame[num].unique()), frame[num].unique()[:10]))
        print()
        
# Hàm lọc các cột có kiểu dữ liệu categorical        
def objects_variable(frame):
    categorical = [col for col in frame.columns if frame.dtypes[col] == object]
    return categorical

# Hàm hiển thị dữ liệu unique của các cột categorical
def display_objects(frame, lst_objects):
    for index, cat in enumerate(lst_objects):
        print('{}. Name obj: {}, Number of unique: {}, Unique Values: {}'
              .format(index + 1, cat, len(frame[cat].unique()), frame[cat].unique()[:10]))
        
# Xây dựng các hàm phân tích dữ liệu đơn biến, hai biến đối với các thuộc tính continuous & categorcal

# Hàm phân tích đơn biến thuộc tính continuos        
def continuous_analysis(frame, var):
    print('----- {} -----'.format(var))
    print(frame[var].describe())
    Q1 = np.quantile(frame[var].dropna(), 0.25)
    Q3 = np.quantile(frame[var].dropna(), 0.75)
    IQR = Q3 - Q1
    outliers = frame.loc[(frame[var] < Q1 - 1.5*IQR) | (frame[var] > Q3 + 1.5*IQR)]
    percent_outliers = outliers.shape[0] / frame.shape[0]
    skew = frame[var].dropna().skew()
    kurtosis = frame[var].dropna().kurtosis()
    median = frame[var].dropna().median()
    miss_value = frame[var].isnull().sum()
    variance = frame[var].var()
    print('* Variance: {}'.format(variance))
    print('* Median: {}'.format(median))
    print('* Skewness: {}'.format(skew))
    print('* Kurtosis: {}'.format(kurtosis))
    print('* Percentage of outliers: {}'.format(percent_outliers))
    print('* Number of missing value: {}'.format(miss_value))
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    sb.distplot(frame[var].dropna())
    plt.subplot(1, 2, 2)
    plt.boxplot(frame[var].dropna())
    plt.show()
    print()
    
# Hàm phân tích đơn biến thuộc tính categorical  
def categorical_analysis(frame, var, ax = (10, 8)):
    print('----- {} -----'.format(var))
    print('Describe: ')
    print(frame[var].describe())
    miss_value = frame[var].isnull().sum()
    unique_val = pd.DataFrame(frame[var].value_counts())
    print('* Unique value: ')
    print(unique_val)
    print('* Mode value: {}'.format(frame[var].mode()[0]))
    print('* Number of missing value: {}'.format(miss_value))
    plt.figure(figsize=ax)
    sb.barplot(data = unique_val, x = var, y = unique_val.index)
    plt.xlabel('count of ' + var)
    plt.show()
    
# Hàm phân tích hai biến có thuộc tính categorical        
def cat_cat(frame, var1, var2, prob, stacked = False):
    from scipy.stats import chi2_contingency
    from scipy.stats import chi2
    print('----- {}  vs {} -----'.format(var1, var2))
    table = pd.crosstab(frame[var1], frame[var2])
    print(table)
    if stacked:
        plt.rcParams["figure.figsize"] = [12, 8]
        table.plot(kind = 'barh', stacked = True)
        plt.xlabel('Count')
        plt.show()
    else:
        temp = table.reset_index()
        plt.rcParams["figure.figsize"] = [12, 8]
        temp.plot(kind = 'barh', x = temp.columns[0])
        plt.xlabel('Count')
        plt.show()
    stat, p_value, dof, expected = chi2_contingency(table)
    print('----- Chi2 Hypothesis Testing -----')
    print('P-value: {}'.format(p_value))
    alpha = 1 - prob
    if p_value <= alpha:
        print('Reject H0 --> {} and {} are dependent.'.format(var1, var2))
    else:
        print('Accept H0 --> {} and {} are independent.'.format(var1, var2))
        
# Hàm phân tích hai biến có thuộc tính continuos    
def cont_cont(frame, var1, var2):
    print('----- {} vs {} -----'.format(var1, var2))
    correlation = frame[var1].corr(frame[var2])
    print('Pearson correlation between {} & {}: {}'.format(var1, var2, correlation))
    sb.pairplot(frame[[var1, var2]].dropna(), size = 5)
    plt.show()
    
# Hàm gọi tính toán Anova
def cal_anova(*arg):
    f, p = st.f_oneway(*arg)
    return f, p

# Hàm levene kiểm tra giả định Anova
def cal_levene(*arg):
    w, p_levene = st.levene(*arg)
    return w, p_levene

# Hàm xoá bỏ giá trị outlier bằng IQR
def remove_outliers(frame, col):
    Q1 = np.quantile(frame[col].dropna(), 0.25)
    Q3 = np.quantile(frame[col].dropna(), 0.75)
    IQR = Q3 - Q1
    clean_data = frame.loc[(frame[col] >= Q1 - 1.5*IQR) & (frame[col] <= Q3 + 1.5*IQR), col]
    return clean_data

# Hàm phân tích ảnh hưởng của biến phân loại lên biến output(continuous)
        
def cat_cont(frame, col, output_var, ax = (10, 6)):
    print('----- {} vs {} -----'.format(col, output_var))
    df = frame[[col, output_var]]
    plt.figure(figsize=ax)
    sb.boxplot(data = df, x = col, y = output_var)
    if len(frame[col].unique()) > 8:
        plt.xticks(rotation = 90)
    df_pivot = df.pivot(columns = col, values = output_var)
    lst = []
    for column in df_pivot.columns:
        lst.append(remove_outliers(df_pivot, column))
    fvalue, pvalue = cal_anova(*lst)
    w_levene, p_levene = cal_levene(*lst)
    print('* --- Levene hypothesis --- *')
    print('p_value: {}'.format(p_levene))
    if p_levene > 0.05:
        print('Accept H0 --> Các quần thể có phương sai bằng nhau.')
    else:
        print('Reject H0 --> Các quần thể có phương sai không bằng nhau.')
    print()
    print('* --- Anova one-way hypothesis --- *')
    print('p_value: {}'.format(pvalue))
    if pvalue <= 0.05:       
        print('Reject H0 --> Có sự khác biệt đáng kể.')
        print()
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        m_comp = pairwise_tukeyhsd(endog = df[output_var],
                          groups = df[col],
                          alpha = 0.05)
        print(m_comp)
    else:
        print('Accept H0 --> Không có sự khác biệt đáng kể')



# Hàm vẽ đường ong precision & recall theo threshold    
def plot_precision_recall_curve(model, X, y, cv):
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import precision_recall_curve
    y_scores = cross_val_predict(model, X, y, cv=cv, method = 'decision_function')
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], 'g--', label = 'Precision')
    plt.plot(thresholds, recalls[:-1], 'b--', label = 'Recall')
    plt.legend(fontsize = 15)
    plt.grid(True)
    plt.xlabel('Thresholds', fontsize = 15)
    plt.title('Precision & Recall by threshold', color = 'red', fontsize = 18)
    plt.show()
    
def ROC_curve_display(model, X, y, pred):
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
    yhat_proba = model.predict_proba(X)
    print('* Area below the curve: {}'.format(round(roc_auc_score(y, yhat_proba[:, pred]), 5)))
    print()
    fpr, tpr, thresholds = roc_curve(y, yhat_proba[:, pred])
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, marker = '.')
    plt.xlabel('False Possitve Rate', fontsize = 15)
    plt.ylabel('True Possitive Rate', fontsize = 15)
    plt.title('ROC Curve of Predict class {}'.format(pred), fontsize = 18)
    plt.grid(True)
    plt.show()
    
# Hàm vẽ biểu đồ heatmap
def correlation_heatmap(frame, lst_cont):
    matrix_corr = frame[lst_cont].corr()
    onescorr = np.ones_like(matrix_corr, bool)
    mask = np.triu(onescorr)
    adjusted_mask = mask[1:, : -1]
    adjusted_matrix_corr = matrix_corr.iloc[1:, :-1]

    fig, ax = plt.subplots(figsize = (16, 14))
    sb.set_theme()
    sb.heatmap(adjusted_matrix_corr, mask = adjusted_mask, annot = True, fmt = '.2f', cmap = 'Blues', vmin = -1, vmax = 1,
    linecolor = 'white', linewidths = 0.5);
    y_ticks = [i for i in adjusted_matrix_corr.index]
    x_ticks = [i for i in adjusted_matrix_corr.columns]
    ax.set_yticklabels(y_ticks, rotation = 0, fontsize = 12)
    ax.set_xticklabels(x_ticks, rotation = 90, fontsize = 12)
    ax.set_title('CONTINUOUS VARIABLES CORRELATIVE MATRIX\n', fontsize = 18, c = 'r')
    plt.show()


# Write & save model
def save_model(model, filename):
    import pickle
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    return

def load_model(filename):
    import pickle
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Hàm đánh giá model regression trên tập test
def eval_regression_testset(model, X_test, y_test):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    yhat_test = model.predict(X_test)
    print('----- REGRESSION MODEL PERFORMANCE IN TEST-----')
    print('* R-squared model of Test: {}'.format(round(model.score(X_test, y_test), 4)))
    print('* MSE of output and predicted: {}'.format(mean_squared_error(y_test, yhat_test)))
    print('* MAE of output and predicted: {}'.format(mean_absolute_error(y_test, yhat_test)))
    plt.figure(figsize=(14, 6))
    sb.distplot(y_test, hist = False, rug = True, kde_kws={'shade':'kde_kws'}, label = 'True label')
    sb.distplot(model.predict(X_test), hist = False, rug = True, kde_kws={'shade':'kde_kws'}, label = 'Predicted label')
    plt.legend()
    plt.show()
    
def crossval_linear_regression(model, X, y, cv):
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_val_predict
    Rcross = cross_val_score(model, X, y, cv = cv)
    MSEcross = cross_val_score(model, X, y, cv = cv, scoring = 'neg_mean_squared_error') * (-1)
    print('----- CROSS VALIDATION OF REGRESSION MODEL -----')
    print('* Rcross value: {}'.format(Rcross.tolist()))
    print('* Mean of folds: {}'.format(round(Rcross.mean(), 4)))
    print('* Std of folds: {}'.format(round(Rcross.std(), 4)))
    print()
    print('* MSE value: {}'.format(MSEcross.tolist()))
    print("* Mean's MSE of fold: {}".format(round(MSEcross.mean(), 4)))
    print("* Std's MSE of fold: {}".format(round(MSEcross.std(), 4)))
    return (Rcross, MSEcross)

# Hàm đánh giá hiệu suất model phân loại trên tập test   
def eval_clf_testset(model, X, y):
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
    from sklearn.metrics import confusion_matrix, classification_report
    yhat = model.predict(X)
    print('----- EVALUATION MODEL PERFORMANCE ON TESTING SET -----')
    print('* Testing Accuracy score: {}'.format(round(accuracy_score(y, yhat), 4)))
    print()
    print('* Confusion Matrix: ')
    print(confusion_matrix(y, yhat))
    print()
    print('* Classification Report: ')
    print(classification_report(y, yhat))

# Hàm đánh giá hiệu suất model phân loại trên tập validation
def eval_clf_valset(model, X, y):
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
    from sklearn.metrics import confusion_matrix, classification_report
    yhat = model.predict(X)
    print('----- EVALUATION MODEL PERFORMANCE ON VALIDATION SET -----')
    print('* Validation Accuracy score: {}'.format(round(accuracy_score(y, yhat), 4)))
    print()
    print('* Confusion Matrix: ')
    print(confusion_matrix(y, yhat))
    print()
    print('* Classification Report: ')
    print(classification_report(y, yhat))


# Spliting dataset into training, validating and testing
def train_val_test_split(X, y, val_size, test_size, random_st):
    """ Function that split dataset into 3 parts: training, validating and testing"""
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = test_size, 
                                                      random_state = random_st)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = val_size, 
                                                      random_state = random_st)
    return X_train, X_val, X_test, y_train, y_val, y_test



# Weighting sample for imbalanced dataset of binary classify problems
def BinarySampleWeights(total_samples, labels, method = None, beta = None):
    
    """
    This function calculate sample weights for binary classification using imbalanced dataset.
    Args:
        method: str: zen, ins, isns, ens
            None: default, w(c) = total_samples / number of samples of class c
            zen: The “balanced” heuristic is inspired by Logistic Regression in Rare Events Data, King, Zen, 2001.
            ins: Inverse Number of Samples, w(c) = 1 / number of sample in class c
            isns: Inverse of Square Root of Number of Samples, w(c) = 1 / sqrt(number of sample in class c)
            ens: Effective Number of Samples, was introduced in the CVPR’19 paper by Google: Class-Balanced Loss Based on Effective Number of Samples.
        beta: int; in [0, 1], suggest experiment with, 0.9, 0.99, 0.999, 0.9999
    """
    import numpy as np
    if method == None:
        sample_weight = np.round(total_samples / np.bincount(labels), 3)
    elif method == 'zen':
        sample_weight = np.round(total_samples / (2 * np.bincount(labels)), 3)
    elif method == 'ins':
        sample_weight = np.round(1 / np.bincount(labels), 4)
    elif method == 'isns':
        sample_weight = np.round(1 / np.bincount(labels) ** 0.5, 4)
    elif method == 'ens':
        eff_num = 1.0 - np.power(beta, np.bincount(labels))
        weights = (1.0 - beta) / eff_num
        sample_weight = weights / np.sum(weights) * 2
    return sample_weight  