import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk import FreqDist
from nltk.tokenize import word_tokenize

import requests
import csv
from bs4 import BeautifulSoup
import os

import pandas as pd
import string
import matplotlib.pyplot as plt

import networkx as nx

#---------------Elaborazione delle funzioni---------------------

#Se crea una funzione per avere tutte l'url in un CSV
def web_scraping_url(url, dircartellaecsv):
        
    response = requests.get(url) # Effettuare la richiesta GET alla pagina web
    
    soup = BeautifulSoup(response.text, 'html.parser') # Crea un oggetto BeautifulSoup per analizzare il contenuto HTML

    pagination_items = soup.find_all(attrs={'class': 'pagination-link'}) # Variabile per memorizzare l'URL della pagina

    # Creare un archivio dei dati raccolti
    df_urls = pd.DataFrame()
    pg_texts = set()
    
    # Iterare tutte le volte che ci sono pagine
    while pagination_items is not None and len(pagination_items) > 0:
        tags_h2 = soup.find_all('h2') 
        if tags_h2:
            for tag_h2 in tags_h2:
                links = tag_h2.find_all('a') # Trovare i collegamenti all'interno di ogni elemento h2
                for link in links: #per ogni link nel tag h2, creare una lista di link
                    url=link.get('href')
                    df_urls = pd.concat([df_urls, pd.DataFrame({'URL': [url]})], ignore_index=True) # Aggiungere l'URL al DataFrame
                    df_urls = df_urls[~df_urls['URL'].str.contains('/tag/')]
        else:
            print(f"No se encontró la sección '{tags_h2}' en la página.")

        # Cerca il link per andare alla pagina successiva

        pg_texts.add(pagination_items[0].text)
        pagination_items = soup.find_all(attrs={'class': 'pagination-link'})

        # Rimuove dagli elementi di impaginazione l'elemento in cui il testo si trova in pg_texts
        pagination_items = [x for x in pagination_items if x.text not in pg_texts]
        if(len(pagination_items) > 0):
            new_link = pagination_items[0].get('href')
            
        else:
            pagination_items = None # Nessun link trovato alla pagina successiva, uscire dal ciclo
            
    print('URL salvate nel csv')
    return df_urls.to_csv(dircartellaecsv, index=False)

def article(url_article): #ottenere il testo di un articolo
        
    # Effettuare la richiesta GET alla pagina web
    response = requests.get(url_article)
    # Crea un oggetto BeautifulSoup per analizzare il contenuto HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Creare un data frame vuoto per salvare i dati dell'articolo
    df_art=pd.DataFrame()
    
    # Creare un data frame con i dati degli articoli di giornale
    title = soup.find_all('h1')
    date = soup.find_all('time')
    paragraph= [p.text.replace('\n',' ') for p in soup.find_all('p') if p.text]
    df_art=paragraph
    print(f'Testo dal articolo salvato nel data frame')
    return df_art

def clean_text(paragraphs): # Pulire ogni testo dal data frame
    print('pulendo...')    
    global stopwords
    global punctuation 
    global stop_words_adicionales
    
    clean_paragraphs = []
    
    for paragraph in paragraphs:
        tokens = word_tokenize(paragraph.lower()) #per dividire il testo in una lista ordinata
        clean_words = [token for token in tokens if token.isalpha() and token not in stopwords and token not in punctuation and token not in stop_words_adicionales]
        clean_paragraphs.append(clean_words)
    print('testo pulito')
    return clean_paragraphs
    
def count_words(df, column):
    conteo_palabras = []

    for fila in df[column]:
        palabras = str(fila).split(',')
        fila_conteo = {}
        
        for palabra in palabras:
            palabra = palabra.strip()  # Rimuovi spazi bianchi intorno a ogni parola
            
            if palabra in fila_conteo:
                fila_conteo[palabra] += 1
            else:
                fila_conteo[palabra] = 1
        
        conteo_palabras.append(fila_conteo)
        df['conteo_palabras'] = pd.Series(conteo_palabras) #Per essere allineato con ogni riga
    print('conteggio effettuato')
    return df

def sum_values(df, dictcolumn):
    suma_valores = {} # Inizializzare un dizionario per memorizzare la somma dei valori

    for fila in df[dictcolumn]: # Scorrere ogni riga nella colonna specificata
        
        for palabra, frecuencia in fila.items():
            
            if palabra in suma_valores: # Sommare i valori perogni parola
                suma_valores[palabra] += frecuencia
            else:
                suma_valores[palabra] = frecuencia
    
    # Crea un dataframe nuovo in base al dizionario 
    df_freq_gral = pd.DataFrame(list(suma_valores.items()), columns=['palabras', 'frecuencia'])
    print('Nuovo dataframe creato con il conteggio di parole')
    return df_freq_gral

def generate_ngrams_from_dataframe(df, column, n):
    ngrams = [] # Lista per salvare i dati generati
    for row in df[column]: # Iterare in ogni riga dal dataframe
        words = ' '.join(row) # Unire  le parole dalla riga in una sola catena di testo
        for i in range(len(words.split()) - n + 1): 
            ngram = ' '.join(words.split()[i:i+n]) # Avere l' n-grama actuale
            ngrams.append(ngram) # Aggiungere l'ngrama alle liste di ngrama
    return ngrams

def create_bigram_network(df):
    G = nx.Graph()
    for index, row in df.iterrows():
        bigram = row['bigrama']
        frequency = row['frecuencias']
        G.add_edge(bigram[0], bigram[1], weight=frequency)
    return G



#-------------------------1ra Parte: Ottenere i dati-----------------------------------------
# 1. Dati una URL di ricerca, una cartela dove salvare il CSV, si chiama la funzione web scraping 
url_p = 'https://www.liberoquotidiano.it/search/?keyword=Silvio%20Berlusconi&sortField=pubdate'  
dircartellaecsv= r'C:\Users\JuliaElenaSilloCondo\OneDrive - ITS Angelo Rizzoli\Documenti\Phyton_Corsaro_Luigi\webscraping-20230505T101355Z-001\enlaces4.csv'
web_scraping_url(url_p,dircartellaecsv)
    # Il risultato è un CSV con tutti il URL dalla ricerca

# 2. Si costruisce un data frame con i testi degli articoli
mycsv=dircartellaecsv 

# 3. Per ogni link nel csv si entra nell'articolo e si estrae il contenuto
with open(mycsv, "r") as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        cabecera = next(lector_csv)  # Leggere la prima riga come head
        df_article=pd.DataFrame()
        for fila in lector_csv:     # Iterare ogni riga dal CSV
            url = fila[0]# Ottenere la URL 
            text_article = article(url) # Se obtiene el data frame con la URL y el texto del cuerpo de los articulos (sin limpiar)
            # Se crea un data frame 
            df_article = pd.concat([df_article, (pd.DataFrame({'url': [url], 'paragraphs': [text_article]}))], ignore_index=True)
            



#-----------------------------------2da Parte: Preparacion de datos --------------------------------------

# 1. Definire le variabili globali per pulire i dati
stopwords = set(stopwords.words('italian'))
punctuation = string.punctuation
stop_words_adicionales = ['\xa0', ',', 'La', 'Amici]', 'Ex', 'decisione,', 'Previsioni', 'Attacco', 'che']

# 2. Si pulisce il articolo e si genera una lista con le parole pulite        
df_article['pragraphs_clean'] = df_article['paragraphs'].apply(clean_text)

#Se realiza el conteo de palabras
column='pragraphs_clean' # Se establece la columna de la cual se quiere contar las palabras
        
df_frequency = count_words(df_article,column) #se cuenta cada palabra y se crea una nueva columna con un diccionario
#print(df_frequency)
#print(type(df_frequency))
columna_diccionario='conteo_palabras'

df_freq_gral = sum_values(df_frequency, columna_diccionario)
# print(df_freq_gral)


#-----------------------3ra Parte: Analisis de datos---------------------------------------------
#Estadisticas generales con Pandas
suma_total = df_freq_gral['frecuencia'].sum()
num_palabras_unicas = df_freq_gral.shape[0]
max_frecuencia = df_freq_gral['frecuencia'].max()
min_frecuencia = df_freq_gral['frecuencia'].min()
promedio_frecuencia = df_freq_gral['frecuencia'].mean()
mediana_frecuencia = df_freq_gral['frecuencia'].median()
desviacion_estandar = df_freq_gral['frecuencia'].std()

# Palabra mas frecuente
df_freq_gral_sorted = df_freq_gral.sort_values(by='frecuencia', ascending=False)

# Mostrare le statistiche generali
print(f'Totale di parole:{suma_total}')
print (f'Totale di parole unici:{num_palabras_unicas}')
print(f'Numero massimo di volte in cui una parola viene ripetuta:{max_frecuencia}')
print(f'Promedio delle parole:{promedio_frecuencia}')
print(f'Mediana delle frequenze:{mediana_frecuencia}')
print(f'Deviazione standard delle frequenze:{desviacion_estandar}')
print(f"La parola piu frecuente è: {df_freq_gral_sorted.iloc[0]['palabras']}") #Qui si ha presso il secondo valore perche manca pulire i dati esistono valori vuoti

# Grafici
#Elaborazione d'una lista di Ngramas
ngrams_result = generate_ngrams_from_dataframe(df_article, 'paragraphs', 2)
#print(ngrams_result)

ngram_counts = pd.Series(ngrams_result).value_counts()  # Se hace el conteo de los valores únicos NGRAMS
ngram_counts = ngram_counts.reset_index()  # Se resetea el índice de la serie
ngram_counts.columns = ['bigrama', 'frecuencias']  # Se renombran las columnas del DataFrame

#  10 vslori piu alti in "frecuencias"
n_bigram=20
top_frequencies = ngram_counts.nlargest(n_bigram, 'frecuencias')

# Estrare i dati dei valori in bigramas e frecuencias
bigram_values = top_frequencies['bigrama'].tolist()
frequency_values = top_frequencies['frecuencias'].tolist()

# Creare un grafici
plt.bar(bigram_values, frequency_values)

# Configurazione dei grafici
plt.xlabel('Bigrama')
plt.ylabel('Frecuency')
plt.title('Top 20 insieme di parole piu usate')

# Ajustare per vedere X completo
plt.xticks(rotation=90)

# Mostrar el gráfico
plt.show()

print('gracias!!')





