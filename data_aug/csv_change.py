import pandas as pd

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