import pandas as pd
import torch

def concat_all_csv(path = './docs/csv/csvsum.csv'):
    eq1 = pd.read_csv('./docs/csv/Eq1.csv', on_bad_lines='skip', sep=';')
    eq2 = pd.read_csv('./docs/csv/Eq2.csv', on_bad_lines='skip', sep=';')
    eq3 = pd.read_csv('./docs/csv/Eq3.csv', on_bad_lines='skip', sep=';')
    eq4 = pd.read_csv('./docs/csv/Eq4.csv', on_bad_lines='skip', sep=';')

    eq_tab = [eq1, eq2, eq3, eq4]

    # We clean the data and create a new dataframe
    columns = ['L', 'PAROLES', 'Dimension Dialogique', 'Dimension Epistémique']
    sum_csv = pd.DataFrame(columns=columns)
    for eq in eq_tab:
        for index, row in eq.iterrows():
            if all(pd.notna(row[column]) for column in ['L', 'PAROLES', 'Dimension Dialogique', 'Dimension Epistémique']):
                sum_csv = sum_csv._append(row, ignore_index=True)

    sum_csv.to_csv(path)


def get_data(path = './docs/csv/csvsum.csv', clear_labels = True):
    
    # Not ideal for now in regards to Pn but might be improved later
    clear_dict = {"0" : 0,"Préc" : 1, "Arg+":2, "Arg-" : 3, "[+]" : 4, "[-]" : 5, "Q" : 7, "Pn" : 8}
    
    df = pd.read_csv(path)
    sentences = df['PAROLES'].to_numpy()
    labels = df['Dimension Dialogique'].to_numpy()
    if clear_labels:
        for i,l in enumerate(labels):
            l = l.strip()
            if not (l in clear_dict or (l[0] == 'P' and l[1:].isdigit())):
                print(i,l)
            if l in clear_dict:
                labels[i] = clear_dict[l]
            elif l[0] == 'P' and l[1:].isdigit():
                labels[i] = 8 # ?int(l[1:])
            else:
                ValueError(f"Label {l} not recognized at line {i}")
            
    print(labels[:10])
    return sentences, labels

get_data()