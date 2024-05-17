from googletrans import Translator
import json
import sys
import pandas as pd
sys.path.append('./docs/csv') # not clean but ok for now
from csv_handler import *



def translate(sentence, language = 'en'):
    translator = Translator()
    translation = translator.translate(sentence, src='fr', dest=language)
    back_translation = translator.translate(translation.text, src=language, dest='fr')
    if back_translation.text != sentence: 
         return back_translation.text

languages = {
    'af': 'afrikaans',
    'sq': 'albanian',
    'am': 'amharic',
    'ar': 'arabic',
    'hy': 'armenian',
    'az': 'azerbaijani',
    'eu': 'basque',
    'be': 'belarusian',
    'bn': 'bengali',
    'bs': 'bosnian',
    'bg': 'bulgarian',
    'ca': 'catalan',
    'ceb': 'cebuano',
    'ny': 'chichewa',
    'zh-cn': 'chinese (simplified)',
    'zh-tw': 'chinese (traditional)',
    'co': 'corsican',
    'hr': 'croatian',
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'eo': 'esperanto',
    'et': 'estonian',
    'tl': 'filipino',
    'fi': 'finnish',
    'fy': 'frisian',
    'gl': 'galician',
    'ka': 'georgian',
    'de': 'german',
    'el': 'greek',
    'gu': 'gujarati',
    'ht': 'haitian creole',
    'ha': 'hausa',
    'haw': 'hawaiian',
    'iw': 'hebrew',
    'he': 'hebrew',
    'hi': 'hindi',
    'hmn': 'hmong',
    'hu': 'hungarian',
    'is': 'icelandic',
    'ig': 'igbo',
    'id': 'indonesian',
    'ga': 'irish',
    'it': 'italian',
    'ja': 'japanese',
    'jw': 'javanese',
    'kn': 'kannada',
    'kk': 'kazakh',
    'km': 'khmer',
    'ko': 'korean',
    'ku': 'kurdish (kurmanji)',
    'ky': 'kyrgyz',
    'lo': 'lao',
    'la': 'latin',
    'lv': 'latvian',
    'lt': 'lithuanian',
    'lb': 'luxembourgish',
    'mk': 'macedonian',
    'mg': 'malagasy',
    'ms': 'malay',
    'ml': 'malayalam',
    'mt': 'maltese',
    'mi': 'maori',
    'mr': 'marathi',
    'mn': 'mongolian',
    'my': 'myanmar (burmese)',
    'ne': 'nepali',
    'no': 'norwegian',
    'or': 'odia',
    'ps': 'pashto',
    'fa': 'persian',
    'pl': 'polish',
    'pt': 'portuguese',
    'pa': 'punjabi',
    'ro': 'romanian',
    'ru': 'russian',
    'sm': 'samoan',
    'gd': 'scots gaelic',
    'sr': 'serbian',
    'st': 'sesotho',
    'sn': 'shona',
    'sd': 'sindhi',
    'si': 'sinhala',
    'sk': 'slovak',
    'sl': 'slovenian',
    'so': 'somali',
    'es': 'spanish',
    'su': 'sundanese',
    'sw': 'swahili',
    'sv': 'swedish',
    'tg': 'tajik',
    'ta': 'tamil',
    'te': 'telugu',
    'th': 'thai',
    'tr': 'turkish',
    'uk': 'ukrainian',
    'ur': 'urdu',
    'ug': 'uyghur',
    'uz': 'uzbek',
    'vi': 'vietnamese',
    'cy': 'welsh',
    'xh': 'xhosa',
    'yi': 'yiddish',
    'yo': 'yoruba',
    'zu': 'zulu'}




def do_data_aug(typ):
    #sentences_train, labels_train, sentences_test, labels_test = get_train_test(typ, use_data_aug = False)
    if typ == 'arg':
        df = pd.read_csv('./docs/csv/arg/train_arg_only.csv')
        sentences_train = df['PAROLES'].tolist()
        labels_train = df['Dimension Dialogique'].tolist()
        columns = ['PAROLES', 'Dimension Dialogique', 'Langue']
        print("starting")
        arg_aug_trad_csv = pd.DataFrame(columns=columns)

        for i in range(len(sentences_train)):
            for language in languages.keys():
                while True:
                    try:
                        translated = translate(sentences_train[i], language)
                        break
                    except Exception as e:
                        print(e)
                if translated:
                    arg_aug_trad_csv = arg_aug_trad_csv._append({'PAROLES': translated, 'Dimension Dialogique': labels_train[i], 'Langue': language}, ignore_index=True)
            print(i/len(sentences_train)*100)

            arg_aug_trad_csv.to_csv('./docs/csv/arg/data_aug/arg_aug.csv')

    
    elif typ == 'dom':
        df = pd.read_csv('./docs/csv/dom/train_dom_only.csv')
        sentences_train = df['PAROLES'].tolist()
        labels_train = df['Domaine'].tolist()

        columns = ['PAROLES', 'Domaine', 'Langue']
        print("starting")
        dom_aug_trad_csv = pd.DataFrame(columns=columns)

        for i in range(len(sentences_train)):
            for language in languages.keys():
                while True:
                    try:
                        translated = translate(sentences_train[i], language)
                        break
                    except Exception as e:
                        print(e)
                if translated:
                    dom_aug_trad_csv = dom_aug_trad_csv._append({'PAROLES': translated, 'Domaine': labels_train[i], 'Langue': language}, ignore_index=True)
            print(i/len(sentences_train)*100)

            dom_aug_trad_csv.to_csv('./docs/csv/dom/data_aug/dom_aug.csv')

do_data_aug('arg')
do_data_aug('dom')