import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('winequality-red.csv')
isnull = df.columns[df.isnull().any()].tolist()

def setGrade(q):
    if q>6:
        return 'good'
    else:
        return 'bad'

df['grade'] = df['quality'].apply(setGrade)

def showCorrelation():
    cor_mat = df.corr()
    mask = np.array(cor_mat)
    mask[np.tril_indices_from(mask)] = False
    fig = plt.gcf()
    fig.set_size_inches(30, 12)
    sns.heatmap(data=cor_mat, mask=mask, square=True, annot=True, cbar=True)
    plt.show()

showCorrelation()

#columns with maximum correlatiion -> alcohol, volatile acid,

def showBoxPlots(col1,col2):
    df.boxplot(column=col1, by=col2)
    plt.show()

#negative coorelation. indicates that the quality decreases with increased volatile acidit
showBoxPlots('volatile acidity','quality')

# positive coorelation. indicates that quality increases with increase in alcohol content
showBoxPlots('alcohol','quality')

sns.countplot(df['grade'])
plt.show()


#BUILDING MODELS

x = df.drop(['quality','grade'],axis=1)
y = df['grade']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

logreg = LogisticRegression(solver='liblinear')
knn = KNeighborsClassifier()
rf = RandomForestClassifier(n_estimators=10)

models = [logreg, knn , rf]
model_names = ['LogisticRegression','KNearestNeighbors', 'RandomForestClassifier']
accuracy_scores = []
d = {}

for model in range(len(models)):
    mod = models[model]
    mod.fit(x_train, y_train)
    pred = mod.predict(x_test)
    accuracy_scores.append(accuracy_score(y_test, pred)*100 )
    print('connfusion matrix of '+model_names[model])
    print(confusion_matrix(y_test,pred))
    print("classification report of " + model_names[model])
    print(classification_report(y_test,pred))


d = {'Modelling Algo': model_names, 'Accuracy': accuracy_scores}
print(d)

acc_frame=pd.DataFrame(d)
print(acc_frame)
sns.barplot(y='Modelling Algo', x='Accuracy',data=acc_frame)
plt.show()

#----CROSS VALIDATION ON ALL MODELS--------

#knn

kscore = cross_val_score(knn,x,y,cv=8).mean()
print("knn accuracy after cross validation =" + str(kscore*100))

#randomforest

rfscore = cross_val_score(rf,x,y,cv=8).mean()
print("Random Forest accuracy after cross validation =" + str(rfscore*100))

#logistic

logregscore = cross_val_score(logreg,x,y,cv=8).mean()
print("Logistic accuracy after cross validation =" + str(logregscore*100))

#------------HYPERPARAMATER TUNING-----------

#Finding the optimal value for k in knn

k_scores=[]
k_range = range(1,30)
for k in k_range:
    knn= KNeighborsClassifier(n_neighbors=k)
    k_scores.append(cross_val_score(knn,x,y,cv=8).mean())

plt.plot(k_range,k_scores)
plt.xlabel('Value of K')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

param_grid = {'n_neighbors': np.arange(1,30)}
knn_cv= GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
knn_cv.fit(x,y)
print("Best value of K in K neighbors is " + str(knn_cv.best_params_))
print("Best Accuracy of the model when k is " + str(knn_cv.best_params_)+ " is " + str(knn_cv.best_score_))

# Finding the optimal value of C in Logistic using gridsearch

c_space = np.logspace(-5,8,15)
param_grid = {'C': c_space}
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
logreg_cv.fit(x,y)
print('Best value of C in Logistic Regression is ' + str(logreg_cv.best_params_))
print("Best Accuracy of the model when C is " + str(logreg_cv.best_params_)+ " is " + str(logreg_cv.best_score_))
