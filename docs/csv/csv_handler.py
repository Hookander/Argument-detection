import pandas as pd

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
sum_csv.to_csv('./docs/csv/csvsum.csv')
