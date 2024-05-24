from docx import Document
import csv
import os
import pandas as pd

def extract_tables_from_docx(docx_path):
    # Charger le document Word
    document = Document(docx_path)
    
    # Initialiser une liste pour stocker toutes les données des tableaux
    tables_data = []
    
    # Parcourir tous les tableaux dans le document
    format = False
    for table in document.tables:
        table_data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            if not format and 'PAROLES' in row_data:
                    format = True
            if format and row_data :
                table_data.append(row_data)
        if table_data != []:
            tables_data +=table_data
    if not format:
        raise Exception("le document n'est pas sous le bon format il faut que la table est 'PAROLES' comme nom de colonne")
    return tables_data
      
def write_tables_to_dataframe(tables_data):
    # Initialiser une liste vide pour stocker les DataFrames de chaque tableau
    dataframes = []
    
    # Convertir chaque tableau en DataFrame et l'ajouter à la liste
    
    df = pd.DataFrame(tables_data[1:], columns=tables_data[0])
    dataframes.append(df)
    
    # Concaténer tous les DataFrames en un seul, avec des lignes vides entre les tableaux
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df
    

# Utilisation exemple
docx_path = 'doc_traitement/Eq4.docx'  # Assurez-vous que ce chemin est correct

def doc_to_sentences(docx_path):
    # Vérifier si le fichier Word existe
    if not os.path.isfile(docx_path):
        print(f"File not found: {docx_path}")
    else:
        
        tables_data = extract_tables_from_docx(docx_path)
        df = write_tables_to_dataframe(tables_data)
        return df['PAROLES'].tolist()
