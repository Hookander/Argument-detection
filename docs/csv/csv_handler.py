import pandas as pd
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Import train_test_split function

#! DO NOT CHANGE
domain_dico = {'Nothing/nan' : 0, 'efficacité': 1, 'utilité': 2, 'éthique': 3, 'faisabilité': 4, 'esthétique': 5,
                    'organisation': 6, 'liberté': 7, 'partage': 8, 'engagement': 9, 'équité': 10,
                    'climatique': 11, 'confiance': 12, 'nuisance': 13, 'acceptabilité': 14, 'écologique': 15,
                    'praticité': 16, 'économique': 17, 'agréabilité': 18, 'taille': 19, 'relations sociales': 20}

domain_dico_new = {'Nothing/nan' : 0, 'efficacité': 1, 'utilité': 1, 'éthique': 2, 'faisabilité': 3, 'esthétique': 5,
                    'organisation': 4, 'liberté': 11, 'partage': 12, 'engagement': 11, 'équité': 2,
                    'climatique': 6, 'confiance': 7, 'nuisance': 8, 'acceptabilité': 3, 'écologique': 6,
                    'praticité': 1, 'économique': 9, 'agréabilité': 1, 'taille': 4, 'relations sociales': 10}

arg_dico = {'Nothing/nan' : 0, 'Arg_fact': 1, 'Arg_value': 2}

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
                        sum_csv = sum_csv._append({'L': row['L'], 'PAROLES': row['PAROLES'], 'Dimension Dialogique': row['Dimension Dialogique'], 'Dimension Epistémique': row['Dimension Epistémique'], 'Domaine': domain.lower()}, ignore_index=True)
                if not(has_append):
                    sum_csv = sum_csv._append(row, ignore_index=True)

    sum_csv.to_csv(path)


def get_data_with_full_labels(path = './docs/csv/csvsum.csv', clear_labels = True):
    
    # Not ideal for now in regards to Pn but might be improved later
    clear_dict = {"0" : 0,"Préc" : 1, "Arg+":2, "Arg-" : 3, "[+]" : 4, "[-]" : 5, "Q" : 7, "Pn" : 8}
    
    df = pd.read_csv(path)
    sentences = df['PAROLES'].to_list()
    labels = df['Dimension Dialogique'].to_list()
    domains = df['Domaine'].to_list()
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
            

    return sentences, labels, domains


def get_data_aug(paths = []):
    """
        Get the data from the augmented csv
        (already simplified labels so we can't use get_data_with_simp_labels)
    """
    full_s = []
    full_l = []
    full_d = []
    if paths == []:
        paths = ["./docs/csv/arg_aug.csv", "./docs/csv/arg_aug_trad_named.csv"]
    for path in paths:
        df = pd.read_csv(path)
        sentences = df['PAROLES'].to_list()
        labels = df['Dimension Dialogique'].to_list()
        domains = df['Domaine'].to_list()
        for i, l in enumerate(labels):
            if l == 'Arg_fact':
                label = 1
            elif l == 'Arg_value':
                label = 2
            else:
                label = 0
            full_s.append(sentences[i])
            full_l.append(label)
        for i, l in enumerate(domains):
            if l in domain_dico:
                domain = domain_dico[l]
            else:
                try :
                    if math.isnan(l):
                        domain = 0
                except: f"get_data_aug : Domain {l} not recognized at line {i}"
            full_d.append(domain)
    return full_s, full_l, full_d

def get_data_with_simp_labels(path = './docs/csv/csvsum.csv', shuffle = False):
    """
        Only cares about if a sentence if an argument or not,
        and whether it's about facts or sentiments

        Matching for sentences:
            Nothing : 0
            Arg + facts : 1
            Arg + value : 2
        
        Matching for the argument's domains:
            ['efficacité', 'utilité', 'éthique', 'faisabilité', 'esthétique', 
            'organisation', 'liberté', 'partage', 'engagement', 'équité', 'climatique', 
            'confiance', 'nuisance', 'acceptabilité', 'écologique', 'praticité', 'économique', 
            'agréabilité', 'taille', 'relations sociales']
    """

    # Take into account synonyms
    
    df = pd.read_csv(path)
    sentences = df['PAROLES'].to_list()
    labels = df['Dimension Dialogique'].to_list()
    arg_types = df['Dimension Epistémique'].to_list()
    domains = df['Domaine'].to_list()

    ret = [0 for _ in labels]
    dom = [0 for _ in domains]
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
    for i,l in enumerate(domains):
        if l in domain_dico:
            dom[i] = domain_dico[l]
        else:
            try :
                if math.isnan(l):
                    dom[i] = 0
            except: f"get_data_with_simp_labels : Domain {l} not recognized at line {i}"
            
            
    if shuffle:
        seed = 42

        # We shuffle the data with a seed to keep the same order across all trainings
        perm = np.random.default_rng(seed = seed).permutation(len(sentences))
        
        sentences = [sentences[i] for i in perm]
        ret = [ret[i] for i in perm]
        dom = [dom[i] for i in perm]
    return sentences, ret, dom

def create_train_test_csv(path, typ, ratio = 0.7):
    """
     Creates 2 separated csv files (train/test) to allow us to do some data_augmentation on the train ONLY
     (doing it on the test creates overfitting)

    """

    sentences, labels, domains = get_data_with_simp_labels(path)
    if typ == 'arg':
        sent_train, sent_test, lab_train, lab_test = train_test_split(sentences, labels, test_size=1-ratio, random_state=42, stratify=labels)

        df_train = pd.DataFrame(columns = ['PAROLES', 'Dimension Dialogique'])
        df_test = pd.DataFrame(columns = ['PAROLES', 'Dimension Dialogique'])

        # We'll only do the data_aug on the arguments
        df_train_arg_only = pd.DataFrame(columns = ['PAROLES', 'Dimension Dialogique'])

        for i in range(len(sent_train)):
            df_train = df_train._append({'PAROLES': sent_train[i], 'Dimension Dialogique': lab_train[i]}, ignore_index=True)
            if lab_train[i] != 0:
                df_train_arg_only = df_train_arg_only._append({'PAROLES': sent_train[i], 'Dimension Dialogique': lab_train[i]}, ignore_index=True)
        for i in range(len(sent_test)):
            df_test = df_test._append({'PAROLES': sent_test[i], 'Dimension Dialogique': lab_test[i]}, ignore_index=True)
        df_train.to_csv('./docs/csv/arg/train_arg.csv')
        df_test.to_csv('./docs/csv/arg/test_arg.csv')
        df_train_arg_only.to_csv('./docs/csv/arg/train_arg_only.csv')

    elif typ == 'dom':
        sentences = [sentences[i] for i in range(len(labels)) if labels[i] != 0]
        domains = [domains[i] for i in range(len(labels)) if labels[i] != 0]
        sent_train, sent_test, dom_train, dom_test = train_test_split(sentences, domains, test_size=1-ratio, random_state=42, stratify=domains)

        df_train = pd.DataFrame(columns = ['PAROLES', 'Domaine'])
        df_test = pd.DataFrame(columns = ['PAROLES', 'Domaine'])
        for i in range(len(sent_train)):
            df_train = df_train._append({'PAROLES': sent_train[i], 'Domaine': dom_train[i]}, ignore_index=True)
        for i in range(len(sent_test)):
            df_test = df_test._append({'PAROLES': sent_test[i], 'Domaine': dom_test[i]}, ignore_index=True)
        df_train.to_csv('./docs/csv/dom/train_dom.csv')
        df_test.to_csv('./docs/csv/dom/test_dom.csv')

#create_train_test_csv('./docs/csv/csvsum.csv', 'arg')

def get_train_test(typ, use_data_aug = True):
    if typ == 'arg':
        train = pd.read_csv('./docs/csv/arg/train_arg.csv')
        test = pd.read_csv('./docs/csv/arg/test_arg.csv')

        sentences_train = train['PAROLES'].to_list()
        labels_train = train['Dimension Dialogique'].to_list()
        sentences_test = test['PAROLES'].to_list()
        labels_test = test['Dimension Dialogique'].to_list()

    elif typ == 'dom':
        train = pd.read_csv('./docs/csv/dom/train_dom.csv')
        test = pd.read_csv('./docs/csv/dom/test_dom.csv')

        sentences_train = train['PAROLES'].to_list()
        labels_train = train['Domaine'].to_list()
        sentences_test = test['PAROLES'].to_list()
        labels_test = test['Domaine'].to_list()
    else:
        print("Invalid type")
        return

    if use_data_aug:
        sentences_aug, labels_aug, domains_aug = get_data_aug(typ)
        sentences_train += sentences_aug
        labels_train += labels_aug
    
    return sentences_train, labels_train, sentences_test, labels_test


def plot_data_distribution(typ, remove_nothing = True, data_aug = True):
    """
        typ = 'arg' or 'dom'
    """
    if typ == 'arg':
        i = 1
        data = get_data_with_simp_labels()[1]
    elif typ == 'dom':
        i = 2
        data = get_data_with_simp_labels()[2]
    else:
        print("Invalid type")
        return
    
    if data_aug:
        data += get_data_aug()[i]
    
    if remove_nothing:
        data = [d for d in data if d != 0]

    plt.hist(data, bins=range(0, 22), alpha=0.7, rwidth=0.85)
    plt.savefig(f'./docs/{typ}_distribution.png')
    plt.show()

#plot_data_distribution('dom',remove_nothing = True, data_aug = False)
#get_data_with_simp_labels()[2]

def create_arg_only_file(output_path = './docs/csv/arg_only_csv.csv'):
    """
        Get the csv with only the arguments to do the data augmentation
    """
    sentences, labels, domains = get_data_with_simp_labels()
    inv_arg_dico = {v: k for k, v in arg_dico.items()}
    inv_domain_dico = {v: k for k, v in domain_dico.items()}
    arg_sentences = []
    arg_labels = []
    arg_domains = []
    for i,l in enumerate(labels):
        if l != 0: # Not nothing -> an argument
            arg_sentences.append(sentences[i])
            arg_labels.append(inv_arg_dico[l])
            arg_domains.append(inv_domain_dico[domains[i]])
    arg_df = pd.DataFrame({'PAROLES': arg_sentences, 'Dimension Dialogique': arg_labels, 'Domaine': arg_domains})
    arg_df.to_csv(output_path)
    with open('./docs/arg_only.txt', 'w') as f:
        for i, s in enumerate(arg_sentences):
            f.write(str(i)+'-' + s + '\n')
        f.close()

def create_arg_augmented_csv(in_txt_file, arg_only_path = './docs/csv/arg_only_csv.csv', output_path = './docs/csv/arg_aug.csv'):
    """
        The csv contains just the arguments modified, in the same order as in the 
        arg_only_csv file.
        So we need to get the labels from there
    """
    with open(in_txt_file, 'r') as f:
        lines = f.readlines()
        lines = [line[7:-3] for line in lines]
        f.close()
    arg_df = pd.read_csv(arg_only_path)
    df_augmented = pd.DataFrame(columns=['PAROLES', 'Dimension Dialogique', 'Domaine'])
    for i, line in enumerate(lines):
        arg_type = arg_df.loc[i]['Dimension Dialogique']
        domain = arg_df.loc[i]['Domaine']
        df_augmented = df_augmented._append({'PAROLES': line, 'Dimension Dialogique': arg_type, 'Domaine': domain}, ignore_index=True)
    df_augmented.to_csv(output_path)
#create_arg_augmented_csv('./docs/csv/arg_augmented.txt')
#create_arg_only_file()








