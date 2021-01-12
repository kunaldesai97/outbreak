
'''Python file to clean up description and extract number of cases and deaths from news articles using spacy (NLP)'''

import pandas as pd
import re
import spacy
from spacy import displacy
from spacy.pipeline import EntityRuler
from word2number import w2n

# Configuration parameters - Pandas
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

def regex_matcher(str):
    '''Function to get the numerical value of number in words'''
    for word in str.split(' '):
        word = word.replace(',','')
        try:
            value = w2n.word_to_num(word)
            return int(value)
        except ValueError:
            pass

def extract_deaths_and_cases(df,parameter):
    '''Function to extract cases and deaths'''

    doc = nlp(df)
    if parameter == "D":
        deaths = []
        for ent in doc.ents:
            if ent.label_=="DEATHS":
                deaths.append(ent.text)
        if len(deaths) > 0:
            value = list(map(regex_matcher, deaths))
            value = list(filter(None, value))
            if len(value) > 0:
                death_numbers = max(value)
                return death_numbers
    elif parameter == "C":
        cases = []
        for ent in doc.ents:
            if ent.label_=="CASES":
                cases.append(ent.text)
        if len(cases) > 0:
            value = list(map(regex_matcher, cases))
            value = list(filter(None, value))
            if len(value) > 0:
                case_numbers = max(value)
                return case_numbers 

def main():
    df = pd.read_csv('Outbreak.csv')
    df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
    df['Month'] = df['Date'].apply(lambda x: x.month)
    df['Year'] = df['Date'].apply(lambda x: x.year)
    df_1996_2020 = df[df['Year'].between(1996,2020,inclusive=True)]
    df_1996_2020 = df_1996_2020[df_1996_2020['Country'].notnull()]

    #Removing punctuations, special characters
    df_1996_2020['Description'] = df_1996_2020['Description'].apply(lambda x: re.sub('[^a-zA-Z0-9_.,()]',' ',str(x)))
    df_1996_2020['Description'] = df_1996_2020['Description'].apply(lambda x:re.sub(r'[0-9]+[.][0-9]+',' ',str(x)))

    #Outliers
    remove_years = list(range(1996,2021,1))
    remove_years = '|'.join(str(x) for x in remove_years)
    df_1996_2020['Description'] = df_1996_2020['Description'].apply(lambda x: re.sub(r'\d+(?=\s+(January|February|March|April|May|June|July|August|September|October|November|December))',' ',str(x)))
    df_1996_2020['Description'] = df_1996_2020['Description'].apply(lambda x: re.sub(r'(January|February|March|April|May|June|July|August|September|October|November|December)',' ',str(x)))
    df_1996_2020['Description'] = df_1996_2020['Description'].apply(lambda x: re.sub(r'(?<=population of)\s[^a-zA-Z]+',' ',str(x)))
    df_1996_2020['Description'] = df_1996_2020['Description'].apply(lambda x:re.sub(r'\d+(?=\s+(year old))',' ',str(x)))
    df_1996_2020['Description'] = df_1996_2020['Description'].apply(lambda x:re.sub(r'\d+(?=\s+(doses))',' ',str(x)))

    df_1996_2020['Description'] = df_1996_2020['Description'].apply(lambda x:re.sub(r'\b({})'.format(remove_years),'',str(x)))


    df_1996_2020['Deaths'] = df_1996_2020['Description'].map(lambda x: extract_deaths_and_cases(x,"D"))
    df_1996_2020['Deaths'] = df_1996_2020['Deaths'].fillna(0)
    df_1996_2020['Cases'] = df_1996_2020['Description'].map(lambda x: extract_deaths_and_cases(x,"C"))
    df_1996_2020['Cases'] = df_1996_2020['Cases'].fillna(0)
    df_1996_2020[['ID','Date','Country','Disease','Update','Month','Year','Cases','Deaths']].to_csv('Deaths_Cases_populated.csv')

if __name__ == '__main__':

    #Loading NLP package to perform text extraction
    nlp = spacy.load("en_core_web_md")
    ruler = EntityRuler(nlp, overwrite_ents=True)
    ruler.add_patterns([{"label": "DEATHS","pattern": [{"ENT_TYPE": "CARDINAL"}, {"TEXT": {"REGEX": r"[\s*a-zA-Z]"}, "OP": "*"},
                                                        {"TEXT": {"REGEX": r'\b(deaths|fatal|fatal.|death|dead|died|mortal.|deceased|succumbed)\b'}}]}])
    ruler.add_patterns([{"label": "DEATHS","pattern": [{"ENT_TYPE": "DATE"}, {"TEXT": {"REGEX": r"[\s*a-zA-Z]"}, "OP": "*"},
                                                        {"TEXT": {"REGEX": r'\b(deaths|fatal|fatal.|death|dead|died|mortal.|deceased|succumbed)\b'}}]}])
    ruler.add_patterns([{"label": "CASES", "pattern": [{"ENT_TYPE": "CARDINAL"}, {"TEXT": {"REGEX": r"[\s*a-zA-Z]"}, "OP": "*"},
                                                       {"TEXT": {"REGEX": r'\b(case[s]?)\b'}}]}])
    ruler.add_patterns([{"label": "CASES","pattern": [{"TEXT": {"REGEX": r'\b(case[s]?)\b'}},{"TEXT": {"REGEX": r"[\s*a-zA-Z]"}, "OP": "*"},
                                                      {"ENT_TYPE": "CARDINAL"}]}])
    ruler.add_patterns([{"label": "CASES", "pattern": [{"ENT_TYPE": "CARDINAL"},{"TEXT": {"REGEX": r'\b(case[s]?)\b'}},
                                                       {"TEXT": {"REGEX": r"[\s*a-zA-Z]"}, "OP": "*"},
                                                       {"ENT_TYPE": "CARDINAL"}]}])
    ruler.add_patterns([{"label": "CASES", "pattern": [{"ENT_TYPE": "CARDINAL"}, {"TEXT": {"REGEX": r"[\s*a-zA-Z]"}, "OP": "*"},
                                                       {"TEXT": {"REGEX": r'\b(tested positive)\b'}}]}])
    ruler.add_patterns([{"label": "CASES", "pattern": [{"TEXT": {"REGEX": r'\b(affecting)\b'}},
                                                       {"TEXT": {"REGEX": r"[\s*a-zA-Z]"}, "OP": "*"},
                                                       {"ENT_TYPE": "CARDINAL"}]}])
    ruler.add_patterns([{"label": "CASES", "pattern": [{"ENT_TYPE": "CARDINAL"}, {"TEXT": {"REGEX": r"[\s*a-zA-Z]"}, "OP": "*"},
                                                       {"TEXT": {"REGEX": r'\b(affected)\b'}}]}])
    nlp.add_pipe(ruler)
    main()
