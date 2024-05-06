import pandas as pd
import sys
sys.path.append('./docs/csv') # not clean but ok for now
from csv_handler import *

def clean_csv():
    df = pd.read_csv('./docs/csv/arg_aug_trad.csv')

    columns = ['PAROLES', 'Dimension Dialogique','Domaine', 'Langue']
    csv = pd.DataFrame(columns=columns)

    sentences = df['PAROLES']
    labels = df['Dimension Dialogique']
    domains = df['Domaine']
    languages = df['Langue']
    for i, row in df.iterrows():
        if i%100 == 0:
            print(i/len(df)*100)
        double = False
        for j in range(i+1, len(df)):
            if row['PAROLES'] == df.loc[j, 'PAROLES']:
                double = True
                break
            if df.loc[j, 'Langue'] == 'zu':
                break
        if not double:
            csv = csv._append(row, ignore_index=True)
    csv.to_csv('./docs/csv/clean_arg_aug_trad.csv')

def change_labels_named_from_id(path):
    df = pd.read_csv(path)
    new_df = pd.DataFrame(columns=['PAROLES', 'Dimension Dialogique', 'Domaine', 'Langue'])

    reverse_domain_dico = {v: k for k, v in domain_dico.items()}
    reverse_arg_dico = {v: k for k, v in arg_dico.items()}

    for i, row in df.iterrows():
        new_df = new_df._append({'PAROLES': row['PAROLES'], 'Dimension Dialogique': reverse_arg_dico[row['Dimension Dialogique']], 'Domaine': reverse_domain_dico[row['Domaine']], 'Langue': row['Langue']}, ignore_index=True)
        
    
    new_df.to_csv('./docs/csv/arg_aug_trad_named.csv')


change_labels_named_from_id('./docs/csv/clean_arg_aug_trad.csv')