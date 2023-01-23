import pandas as pd
import numpy as np
import joblib
import itertools
from datasist.structdata import detect_outliers
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    cm = np.round(cm, 2)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
def MLPredictAcc(X, y, classes , scale = False , smote = False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if (smote == True):
        sampler = SMOTE()
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    if (scale == True) :
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    models = {
        "XGB": XGBClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVC": SVC(),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(),
        "GaussianNB" : GaussianNB(),
        "Perceptron" : Perceptron(),
        "LinearSVC" : LinearSVC(),
        "SGDClassifier" : SGDClassifier(),
        "LogisticRegression" : LogisticRegression()
    }
    modell = []
    modell_acc = []
    model_built = {}
    for name, model in models.items():
        print(f'Training Model {name} \n--------------')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cf = confusion_matrix(y_test, y_pred)
        acc_svc = round(accuracy_score(y_test, y_pred) * 100,2)
        modell.append(name)
        modell_acc.append(acc_svc)
        model_built[name]=model
        plot_confusion_matrix(cf, classes, title='{} cf with acc = {} %'.format(name,acc_svc))
        print('-' * 30)
    models = pd.DataFrame(
        {
            'Model': modell,
            'Score': modell_acc ,

        })
    models = models.sort_values(by='Score', ascending=False)
    models['Score'] = models['Score'].apply(lambda x : str(x) + " %")
    modelss = pd.DataFrame({
        "index ": [p for p in range(1,len(modell_acc)+1)],
         "model" : models['Model'],
         'Score': models['Score'],
    })

    if (scale == True):
        return modelss, model_built , scaler
    else:
        return modelss, model_built
def check_category_classes(df):
    return df.select_dtypes(include='O').columns.to_list()
def check_non_category_classes(df):
    return df.select_dtypes(exclude='O').columns.to_list()
def define_column_type(df):
    numerical_column =check_non_category_classes(df)
    categorical_column = check_category_classes(df)
    print("numerical_column", numerical_column)
    print("categorical_column", categorical_column)
    return numerical_column , categorical_column
def show_value_count_category_column(df ):
    for name in define_column_type(df)[1]:
        df_count = pd.DataFrame(df[name].value_counts())
        print(df_count)
        print("*" * 50)

def allam_visualize_null_count(df):
    plt.figure(figsize=(12,8))
    print(df.isnull().sum())
    sns.heatmap(df.isnull())
def make_encoding_dict(df):
    return dict(tuple(zip(df.value_counts().index.tolist(), [i for i in range (100)])))
def detect_outlier(df,numerical):
    for col in numerical:
        outliers = detect_outliers(df, 0, [col])
        df.drop(outliers, inplace=True)
        print("len outliner in {} = {}".format(col,len(outliers)) )
def load_modelWithScaler(model_path,scaler_path ,data ,returnName=False,dictionary = None):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    dictionary = dictionary
    data = data
    prediction = model.predict(scaler.transform([data]))
    if (returnName == True):
        for name, age in dictionary.items():
            if age == prediction:
                return name
    else:
        return prediction