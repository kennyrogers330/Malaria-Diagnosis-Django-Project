import pandas as pd

from sklearn.ensemble import RandomForestClassifier
import joblib
df = pd.read_csv('malaria_clinical_data.csv')
x = df.median()
df.fillna(x, inplace = True, axis=0)
df.drop_duplicates(inplace = True)

datafeatures = ['temperature','parasite_density', 'wbc_count', 
                'hb_level', 'hematocrit', 'mean_cell_volume', 'mean_corp_hb',
                'mean_cell_hb_conc', 'platelet_count', 'platelet_distr_width', 
                'mean_platelet_vl', 'neutrophils_percent', 'lymphocytes_percent',
                'mixed_cells_percent', 'neutrophils_count', 'lymphocytes_count',
                'mixed_cells_count']
X=df[datafeatures]
y=df['Clinical_Diagnosis']


RF_model=RandomForestClassifier(n_estimators=100)
RF_model.fit(X.values, y)

#Create Persisting Model
joblib.dump(RF_model, 'Malaria-diagnosis.joblib')