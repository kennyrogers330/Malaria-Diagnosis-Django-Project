import pandas as pd
df = pd.read_csv('malaria_clinical_data.csv')
x = df.median()
df.fillna(x, inplace = True, axis=0)
df.drop_duplicates(inplace = True)

#########################Part 2##################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm #support vector machine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

input("Press enter to calculate accuracy of models")
datafeatures = ['temperature','parasite_density', 'wbc_count', 
                'hb_level', 'hematocrit', 'mean_cell_volume', 'mean_corp_hb',
                'mean_cell_hb_conc', 'platelet_count', 'platelet_distr_width', 
                'mean_platelet_vl', 'neutrophils_percent', 'lymphocytes_percent',
                'mixed_cells_percent', 'neutrophils_count', 'lymphocytes_count',
                'mixed_cells_count']
X=df[datafeatures]
y=df['Clinical_Diagnosis']

X_train, X_test, y_train, y_test=train_test_split (X,y, test_size=0.3)

#Create a Decision Tree, Logistic Regression, Suport Vector Machine  and Random Forest Classifiers

Decision_tree_model= DecisionTreeClassifier()
Logistic_regression_Model=LogisticRegression(solver='lbfgs',max_iter=10000)
SVM_model=svm.SVC(kernel='linear')
RF_model=RandomForestClassifier(n_estimators=100)

#Train the models using the training sets

Decision_tree_model.fit(X_train, y_train)
Logistic_regression_Model.fit(X_train, y_train)
SVM_model.fit(X_train, y_train)
RF_model.fit(X_train, y_train)

#Predict the response for test dataset

DT_Prediction =Decision_tree_model.predict(X_test)
LR_Prediction =Logistic_regression_Model.predict(X_test)
SVM_Prediction =SVM_model.predict(X_test)
RF_Prediction =RF_model.predict(X_test)

# Calculation of Model Accuracy

DT_score=accuracy_score(y_test, DT_Prediction)
lR_score=accuracy_score(y_test, LR_Prediction)
SVM_score=accuracy_score(y_test, SVM_Prediction)
RF_score=accuracy_score(y_test, RF_Prediction)

# Display Accuracy

print ("Decistion Tree accuracy =", DT_score*100,"%")
print ("Logistic Regression accuracy =", lR_score*100,"%")
print ("Suport Vector Machine accuracy =", SVM_score*100,"%")
print ("Random Forest accuracy =", RF_score*100,"%")

print(X)

