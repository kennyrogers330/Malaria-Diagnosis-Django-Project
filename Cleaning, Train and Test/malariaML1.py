import pandas as pd
df = pd.read_csv('malaria_clinical_data.csv')
x = df.median()
input("Press Enter to replace empty cells with median at axis=0 ")
df.fillna(x, inplace = True, axis=0)
#print(df.to_string())

input("Press enter to view duplicates")
print(df.duplicated())
input("Press Enter to delete duplicates")
df.drop_duplicates(inplace = True)
#print(df.to_string())

input("Press enter to calculate accuracy of models")
datafeatures = ['temperature','parasite_density', 'wbc_count', 
                'hb_level', 'hematocrit', 'mean_cell_volume', 'mean_corp_hb',
                'mean_cell_hb_conc', 'platelet_count', 'platelet_distr_width', 
                'mean_platelet_vl', 'neutrophils_percent', 'lymphocytes_percent',
                'mixed_cells_percent', 'neutrophils_count', 'lymphocytes_count',
                'mixed_cells_count']
X=df[datafeatures]
y=df['Clinical_Diagnosis']
print(X)
