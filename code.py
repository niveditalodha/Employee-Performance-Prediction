import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

data = pd.read_csv('dataset/train.csv')

data.columns

print("Dimensions before removing null values")
data.shape
print("Null values?")
print(data.isnull().any())

data.head()

data.info()

dept = data.iloc[:,[1,12]].copy()
dept_per = dept.copy()
print(dept_per)

plt.figure(figsize=(15,4))
ax = sns.countplot(x="department",data=data, palette="viridis",hue="is_promoted", order = data['department'].value_counts().index)
ax.grid(False)
plt.suptitle("Department")
plt.show()

plt.figure(figsize=(15,20))
ax = sns.countplot(y="region",data=data, 
                    palette="viridis", order = data['region'].value_counts().index)
ax.grid(False)
sns.set(style="whitegrid")
plt.suptitle("Region")
plt.show()

plt.figure(figsize=(6,4))
ax = sns.countplot(x="gender",data=data, palette="viridis",hue="is_promoted", order=data['gender'].value_counts().index)
sns.set(style="whitegrid")
ax.grid(False)
plt.suptitle("Gender")
plt.show()

plt.figure(figsize=(6,4))
ax = sns.countplot(x="recruitment_channel",data=data, palette="viridis",hue="is_promoted", order=data['recruitment_channel'].value_counts().index)
ax.grid(False)
sns.set(style="whitegrid")
plt.suptitle("Recruitment Channel")
plt.show()

data.is_promoted.value_counts(normalize=True)

data.is_promoted.value_counts()

data_test = pd.read_csv('dataset/test.csv')

merged = data
merged = merged.append(data_test)
merged.shape

data_merged = pd.DataFrame(merged)
data_merged['education'].replace(np.nan, 'missing', inplace=True)
data_merged['previous_year_rating'].replace(np.nan, data_merged['previous_year_rating'].median(), inplace=True)
print(data_merged.shape)
print("Null values?")
print(data_merged.isnull().any())

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories='auto')
ohe_arr = ohe.fit_transform(data_merged[['department','region','education','gender','recruitment_channel']]).toarray()
ohe_labels = ohe.get_feature_names(['department','region','education','gender','recruitment_channel'])
ohe_df = pd.DataFrame(ohe_arr, columns= ohe_labels)
ohe_df.head()

data_merged.drop(columns= ['department','region','education','gender','recruitment_channel'], inplace=True)

data_merged.reset_index(inplace=True)
data_merged = pd.concat([data_merged,ohe_df],axis=1,join='inner')
data_merged.info()

data_merged.drop(columns= ['index'], inplace=True)
data_merged.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_merged[['no_of_trainings','age','previous_year_rating','length_of_service','awards_won?','avg_training_score']] = scaler.fit_transform(data_merged[['no_of_trainings','age','previous_year_rating','length_of_service','awards_won?','avg_training_score']])
data_merged.head()

seed_value = 12321
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
np.random.seed(seed_value)

train, test = data_merged[~data_merged['is_promoted'].isnull()], data_merged[data_merged['is_promoted'].isnull()]
print("Shape of Train Dataset: ",train.shape)
print("Shape of Test Dataset: ",test.shape)

train.drop(columns=['employee_id'],inplace=True)
print("Shape of Train Dataset: ",train.shape)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train.drop(columns=['is_promoted']),train['is_promoted'], test_size=0.3,random_state=seed_value)
print("Shape of X Train Dataset: ",X_train.shape)
print("Shape of Y Train Dataset: ", y_train.shape)
print("Shape of X Valid Dataset: ",X_valid.shape)
print("Shape of Y Valid Dataset: ",y_valid.shape)

from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train,y_train)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)

from sklearn.feature_selection import SelectKBest

select_top40 = SelectKBest(mutual_info_classif,k=40)
select_top40.fit(X_train,y_train)
X_train.columns[select_top40.get_support()]

feat_select = X_train.columns[select_top40.get_support()]
X_train[feat_select].head(5)

#SVM
from sklearn.svm import SVC
svm=SVC(random_state=1)
svm.fit(X_train,y_train)
print("train accuracy:",svm.score(X_train,y_train))
print("test accuracy:",svm.score(X_valid,y_valid))

from sklearn.metrics import classification_report
class_names=np.array(['0','1'])
prediction_SVM_all = svm.predict(X_valid)
print(classification_report(y_valid, prediction_SVM_all, target_names=class_names))

# Function to plot the confusion Matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd' 
    thresh = cm.max() / 2.
    for i in range (cm.shape[0]):
        for j in range (cm.shape[1]):
            
            plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, prediction_SVM_all)
plot_confusion_matrix(cm,class_names)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(max_depth=5,n_estimators=100)
model_rf.fit(X_train,y_train)
print("train accuracy:",model_rf.score(X_train,y_train))
print("test accuracy:",model_rf.score(X_valid,y_valid))

from sklearn.metrics import classification_report
class_names=np.array(['0','1'])
prediction_RandomForest_all = model_rf.predict(X_valid)
print(classification_report(y_valid, prediction_RandomForest_all, target_names=class_names))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, prediction_RandomForest_all)
plot_confusion_matrix(cm,class_names)

from sklearn.ensemble import AdaBoostClassifier

model_adaboost = AdaBoostClassifier(n_estimators=100)
model_adaboost.fit(X_train,y_train)
print("train accuracy:",model_adaboost.score(X_train,y_train))
print("test accuracy:",model_adaboost.score(X_valid,y_valid))

from sklearn.metrics import classification_report
class_names=np.array(['0','1'])
prediction_AdaBoost_all = model_adaboost.predict(X_valid)
print(classification_report(y_valid, prediction_AdaBoost_all, target_names=class_names))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, prediction_AdaBoost_all)
plot_confusion_matrix(cm,class_names)

from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(3)
model_knn.fit(X_train,y_train)
print("train accuracy:",model_knn.score(X_train,y_train))
print("test accuracy:",model_knn.score(X_valid,y_valid))

from sklearn.metrics import classification_report
class_names=np.array(['0','1'])
prediction_KNN_all = model_knn.predict(X_valid)
print(classification_report(y_valid, prediction_KNN_all, target_names=class_names))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, prediction_KNN_all)
plot_confusion_matrix(cm,class_names)

from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(max_depth=5)
model_dt.fit(X_train,y_train)
print("train accuracy:",model_dt.score(X_train,y_train))
print("test accuracy:",model_dt.score(X_valid,y_valid))

from sklearn.metrics import classification_report
class_names=np.array(['0','1'])
prediction_DT_all = model_dt.predict(X_valid)
print(classification_report(y_valid, prediction_DT_all, target_names=class_names))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, prediction_DT_all)
plot_confusion_matrix(cm,class_names)

from sklearn.naive_bayes import GaussianNB
model_nb = GaussianNB()
model_nb.fit(X_train,y_train)
print("train accuracy:",model_nb.score(X_train,y_train))
print("test accuracy:",model_nb.score(X_valid,y_valid))

from sklearn.metrics import classification_report
class_names=np.array(['0','1'])
prediction_NB_all = model_nb.predict(X_valid)
print(classification_report(y_valid, prediction_NB_all, target_names=class_names))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, prediction_NB_all)
plot_confusion_matrix(cm,class_names)

# Import required libraries for ANN
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor


# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)

print("train accuracy:",mlp.score(X_train,y_train))
print("test accuracy:",mlp.score(X_valid,y_valid))

from sklearn.metrics import classification_report
class_names=np.array(['0','1'])
prediction_ANN_all = mlp.predict(X_valid)
print(classification_report(y_valid, prediction_ANN_all, target_names=class_names))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, prediction_ANN_all)
plot_confusion_matrix(cm,class_names)