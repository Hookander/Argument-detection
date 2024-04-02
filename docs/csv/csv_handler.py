import pandas as pd
import torch
import numpy as np


def concat_all_csv(path = './docs/csv/csvsum.csv'):
    """
        The problem is that, for each line that contains an argument (fact or value), 
        the domain of that argument (ecological, economic, etc.) is given in the next line, 
        so we must concatenate the two lines in addition of concatenating all the csv.

        Also, might have multiple domains -> create multiple lines
    """
    eq1 = pd.read_csv('./docs/csv/Eq1.csv', on_bad_lines='skip', sep=';')
    eq2 = pd.read_csv('./docs/csv/Eq2.csv', on_bad_lines='skip', sep=';')
    eq3 = pd.read_csv('./docs/csv/Eq3.csv', on_bad_lines='skip', sep=';')
    eq4 = pd.read_csv('./docs/csv/Eq4.csv', on_bad_lines='skip', sep=';')

    eq_tab = [eq1, eq2, eq3, eq4]

    # We clean the data and create a new dataframe
    columns = ['L', 'PAROLES', 'Dimension Dialogique', 'Dimension Epistémique', 'Domaine']
    sum_csv = pd.DataFrame(columns=columns)
    for eq in eq_tab:
        for index, row in eq.iterrows():
            has_append = False
            if all(pd.notna(row[column]) for column in ['L', 'PAROLES', 'Dimension Dialogique', 'Dimension Epistémique']):
                if row['Dimension Dialogique'].strip()[:-1] == 'Arg':
                    domains = eq.loc[index + 1]['Dimension Epistémique'][1:-1].split(', ') # We remove the brackets
                    for domain in domains:
                        has_append = True
                        sum_csv = sum_csv._append({'L': row['L'], 'PAROLES': row['PAROLES'], 'Dimension Dialogique': row['Dimension Dialogique'], 'Dimension Epistémique': row['Dimension Epistémique'], 'Domaine': domain}, ignore_index=True)
                if not(has_append):
                    sum_csv = sum_csv._append(row, ignore_index=True)

    sum_csv.to_csv(path)

concat_all_csv(path = './docs/csv/csvsum3.csv')

def get_data_with_full_labels(path = './docs/csv/csvsum.csv', clear_labels = True):
    
    # Not ideal for now in regards to Pn but might be improved later
    clear_dict = {"0" : 0,"Préc" : 1, "Arg+":2, "Arg-" : 3, "[+]" : 4, "[-]" : 5, "Q" : 7, "Pn" : 8}
    
    df = pd.read_csv(path)
    sentences = df['PAROLES'].to_list()
    labels = df['Dimension Dialogique'].to_list()
    if clear_labels:
        for i,l in enumerate(labels):
            l = l.strip()
            if not (l in clear_dict or (l[0] == 'P' and l[1:].isdigit())):
                print(i,l)
            if l in clear_dict:
                labels[i] = clear_dict[l]
            elif l[0] == 'P' and l[1:].isdigit(): # Pn
                labels[i] = 8 # ?int(l[1:])
            else:
                print(f"Label {l} not recognized at line {i}")
            

    return sentences, labels

def get_data_with_simp_labels(path = './docs/csv/csvsum.csv', shuffle = False):
    """
        Only cares about if a sentence if an argument or not,
        and whether it's about facts or sentiments
        Matching :
            Nothing : 0
            Arg + facts : 1
            Arg + value : 2
    """
    
    df = pd.read_csv(path)
    sentences = df['PAROLES'].to_list()
    labels = df['Dimension Dialogique'].to_list()
    arg_types = df['Dimension Epistémique'].to_list()

    ret = [0 for _ in labels]

    for i,l in enumerate(labels):
            lab = l.strip()
            typ = arg_types[i].strip()
            match (lab, typ):
                case ('Arg+', 'FAI') :
                    ret[i] = 1
                case ('Arg-', 'FAI'):
                    ret[i] = 1
                case ('Arg+', 'VA'):
                    ret[i] = 2
                case ('Arg-', 'VA'):
                    ret[i] = 2
                case _:
                    ret[i] = 0
    if shuffle:
        perm = np.random.permutation(len(sentences))
        sentences = [sentences[i] for i in perm]
        ret = [ret[i] for i in perm]
    return sentences, ret













