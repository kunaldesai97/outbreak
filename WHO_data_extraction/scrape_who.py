'''Python file to create consolidated dataset of outbreaks from 1996 to 2020'''

from bs4 import BeautifulSoup
import os
from requests import get
import pandas as pd
from datetime import datetime
from functools import reduce
import pycountry

def download_yearwise(path):
    '''Function to download HTML files of news articles for different years'''

    years_dir = os.path.join(path,'Years')
    os.mkdir('News_Articles')
    news_dir = os.path.join(path,'News_Articles')
    os.chdir(news_dir)
    count = 0
    for filename in os.listdir(years_dir):
        if filename.endswith(".html"):
            file = open(years_dir+'/'+filename, 'r')
            soup = BeautifulSoup(file, 'html.parser')
            year = filename[:-5]

            os.mkdir(year)
            os.chdir(str(os.getcwd())+'/'+year)
            tag = soup.find("div", {"class": "col_2-1_1"})
            archive = tag.find("ul",{"class":"auto_archive"})
            links = archive.find_all("a")

            for link in links:
                path_link  = 'https://www.who.int'
                path_link = path_link + link['href']
                count+=1
                response = get(path_link)
                str_date = link.contents[0].replace(' ','_')

                with open(str(count)+'_'+str_date+'.html','w') as file:
                    file.write(response.text)
            
            os.chdir(news_dir)

    os.chdir(path)

def download_archive(path):
    '''Function to download the archive of outbreaks for different years'''

    os.mkdir('Years')
    years = [i for i in range(1996,2021)]
    for year in years:
        url = 'https://www.who.int/csr/don/archive/year/'+str(year)+'/en/'
        response = get(url)

        with open(path+'/Years/'+str(year)+'.html','w') as file:
            file.write(response.text)

def extract_data(path):
    '''Function to scrape data from news articles and create a consolidated CSV file'''
    
    country  = []
    disease = []
    id = []
    date = []
    description = []
    update = []
    list_disease = []
    links = []


    list_disease = ['Acute diarrhoeal syndrome', 'Acute febrile syndrome', 'Acute haemorrhagic fever syndrome', 'Acute neurological syndrome', 'Acute respiratory syndrome', 'Acute watery diarrhoeal syndrome', 'Anthrax', 'Avian', 'Botulism', 'Buffalopox', 'Carbapenem-resistant P. aeruginosa', 'Chikungunya', 'Cholera', 'Coccidioidomycosis', 'Creutzfeldt-Jakob disease', 'Crimean-Congo haemorrhagic fever', 'Dengue', 'Dengue fever', 'Dengue haemorrhagic fever', 'Diphtheria', 'Dysentry', 'Ebola', 'Ebola virus disease', 'Encephalitis', 'Elizabethkingia', 'Encephalitis, Saint-Louis', 'Enterohaemorrhagic ', 'Enterohaemorrhagic escherischia coli infection', 'Enterovirus', 'Foodborne disease', 'Gonococcal infection', 'Guillain-BarrÃ© syndrome ', 'Haemorrhagic fever syndrome', 'Haemorrhagic fever with renal syndrome', 'Hantavirus pulmonary syndrome', 'Hepatitis', 'HIV infection','Influenza A(H1N1)', 'Japanese encephalitis', 'Lassa fever', 'Legionellosis', 'Leishmaniasis', 'Leptospirosis', 'Listeriosis', 'Louseborne typhus', 'Malaria', 'Marburg haemorrhagic fever', 'Measles', 'Meningitis', 'Meningococcal disease', 'MERS-CoV', 'Microcephaly', 'Middle East respiratory syndrome coronavirus (MERS-CoV)', 'Monkeypox', 'Myocarditis', 'Nipah virus', 'Novel Coronavirus', "O'Nyong-Nyong fever", 'Oropouche virus disease','Pandemic (H1N1)', 'Pertussis', 'Plague', 'Polio','Poliomyelitis', 'Rabies', 'Relapsing fever', 'Rift Valley fever', 'Salmonellosis', 'Seoul virus', 'SARS', 'Shigellosis', 'Smallpox vaccine - accidental exposure', 'Staphylococcal food intoxication','Swine flu','Swine influenza','Influenza','Tularemia', 'Tularaemia', 'Typhoid fever', 'Viral haemorrhagic fever', 'Vector-borne diseases', 'West Nile', 'Yellow fever', 'Zika virus infection']

    for pth, dir, files in os.walk(os.path.join(path,'News_Articles')):
        for filename in files:
            if type(filename) == str and filename.endswith('.html'):
                path_file = os.path.join(pth, filename)
                file = open(path_file, 'r')
                soup = BeautifulSoup(file, 'html.parser')
                print(filename)
                file_items = filename.split('_')

                # Extract ID of article
                id.append(file_items[0])

                # Extract Date of article
                try:
                    date_items = filename[:-5].split('_')[1:]
                    dateString = ""
                    for item in date_items:
                        dateString = dateString+'-'+item
                    datetime_object = datetime.strptime(dateString[1:], '%d-%B-%Y')
                    date.append(datetime_object)
                except:
                    date.append('NA')

                # Extract country of article
                try:
                    flag = False
                    title = soup.find("h1",{"class":"headline"})
                    headline = title.contents[0]
                    if 'Nigeria' in headline:
                        country.append('Nigeria')
                    else:
                        for c in pycountry.countries:
                            if c.name.split(',')[0] in headline:
                                flag = True
                                country.append(c.name.split(',')[0])
                                break
                        if flag == False:
                            country.append('NA')

                except Exception as e:
                    country.append('NA')

                # Extract Disease of article
                try:
                    flag_disease = False
                    title = soup.find("h1",{"class":"headline"})
                    headline = title.contents[0]
                    for d in list_disease:
                        if d.lower() in headline.lower():
                            flag_disease = True
                            disease.append(d)
                            break

                    if flag_disease == False:
                        disease.append('NA')
                
                except Exception as e:
                    disease.append('NA')

                # Extract update of article
                try:
                    title = soup.find("h1",{"class":"headline"})
                    headline = title.contents[0]
                    if 'update' in headline.lower():
                        update.append('Yes')
                    else:
                        div_update = soup.find('div',{'class':'meta'})
                        sub_head = div_update.find('span',{'class':""})
                        if 'update' in sub_head.contents[0].lower():
                            update.append('Yes')
                        else:
                            update.append('No')

                except:
                    update.append('No')
            
                # Extract Description of article
                try:
                    desc = []
                    desc_tag = soup.find("div",{"id":"primary"})
                    for content in desc_tag.find_all("p"):
                        if content.find('b'):
                            content.b.replaceWith(' ')
                        desc.append(content.text)
                    desc = reduce(lambda x,y: x+y, desc[0:])
                    description.append(desc)

                except Exception as e:
                    description.append('NA')

    df = pd.DataFrame({"ID":id,"Date":date,"Country":country,"Disease":disease,"Description":description, "Update":update})
    df.to_csv(path+'/Outbreak.csv')


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    download_archive(dir_path)
    download_yearwise(dir_path)
    extract_data(dir_path)

if __name__ == '__main__':
    main()
