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
        for j in range(i+1, i+140): # Less than 140 languages
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

def remove_duplicatas(in_path, out_path):
    """
        Remove duplicates sentences from a csv file
    """

    df = pd.read_csv(in_path)
    new_df = pd.DataFrame(columns=['PAROLES', 'Dimension Dialogique', 'Domaine', 'Langue'])

    i = -1
    while i < len(df) - 1:
        i += 1
        row = df.loc[i]

        print(i/len(df)*100)
        sentences_added = []
        for j in range(i, i+140):
            if df.loc[j, 'Langue'] == 'zu': # Last language
                i = j
                break
            concat = "".join(df.loc[j, 'PAROLES'].lower().split())
            if concat not in sentences_added:

                new_df = new_df._append(df.loc[j], ignore_index=True)
                sentences_added.append(concat)

        new_df.to_csv(out_path)

#remove_duplicatas('./docs/csv/arg/data_aug/arg_aug.csv', './docs/csv/arg/data_aug/arg_aug_cleaned.csv')
remove_duplicatas('./docs/csv/dom/data_aug/dom_aug.csv', './docs/csv/dom/data_aug/dom_aug_cleaned.csv')
#change_labels_named_from_id('./docs/csv/clean_arg_aug_trad.csv')