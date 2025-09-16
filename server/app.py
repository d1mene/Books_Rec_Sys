import warnings
warnings.filterwarnings("ignore")


import streamlit as st
import numpy as np
import pandas as pd
import lightfm as lf
import nmslib
import pickle
import scipy.sparse as sparse
import plotly.express as px


@st.cache_resource
def read_files(folder_name="data"):
    """
    Функция для чтения данных.
    Возвращает два DataFrame c ретйингами и характеристиками книг.
    """
    ratings = pd.read_csv(folder_name + '/ratings.csv')
    books = pd.read_csv(folder_name + '/books.csv')
    return ratings, books

def make_mappers(books):
    """
    Функция, сопоставляющая теги книг с их названиями и авторами
    """
    name_mapper = dict(zip(books.book_id, books.title))
    author_mapper = dict(zip(books.book_id, books.authors))
    
    return name_mapper, author_mapper

def load_embeddings(file_name='data/item_embeddings.pkl'):
    """
    Функция чтения эмбеддингов
    """
    with open(file_name, 'rb') as f:
        item_embeddings = pickle.load(f)
        
    nms_index = nmslib.init(method='hnsw', space='cosinesimil')
    nms_index.addDataPointBatch(item_embeddings)
    nms_index.createIndex(print_progress=True)
    
    return item_embeddings, nms_index

def nearest_books_nms(book_id, index, n=10):
    """
    Функция для поиска ближайших соседей.
    """
    nn = index.knnQuery(item_embeddings[book_id], k=n)
    
    return nn

def get_recommendation_df(ids, distances, name_mapper, author_mapper):
    """
    Функция, создающая конечный запрос в виде DataFrame:
    * book_name - название книги;
    * book_author - имя автора;
    * distance - значение метрики расстояния до книги;
    """
    names = []
    authors = []
    
    for idx in ids:
        names.append(name_mapper[idx])
        authors.append(author_mapper[idx])
        
    rec_df = pd.DataFrame({'book_name': names, 'book_author': authors, 'distance': distances})
    
    return rec_df

# Загружаем данные
ratings, books = read_files(folder_name='data')
# Создаём словари для сопоставления id книг и их названий/авторов
name_mapper, author_mapper = make_mappers(books)
# Загружаем эмбеддинги и создаём индекс для поиска
item_embeddings, nms_idx = load_embeddings()


#'''
#<><<<><><><><><<><><><Интерфейс><><><><><><><><<><><>
#'''

st.title('Recommendation System for Books')
st.markdown("""### Welcome to the Book's Recommendation App!
This application is a prototype of a recommendation system based on a machine learning model.

To use the application, you need:
1. Enter the approximate name of the book you like
2. Select its exact name in the pop-up list of books
3. Specify the number of books you need to recommend

After that, the application will give you a list of books most similar to the book you specified""")

title = st.text_input('Please enter book name', '')
title = title.lower()

output = books[books['title'].apply(lambda x: x.lower().find(title)) >= 0]

option = st.selectbox('Select the book you need', output['title'].values)

if option:
    st.markdown('You\'ve selected: "{}"'.format(option))
    print(books[books['title'] == option]['image_url'])
    img = books[books['title'] == option]['image_url'].iloc[0]
    
    if img:
        st.image(books[books['title'] == option]['image_url'].iloc[0])
    
    count_recommendation = st.number_input(
        label="Specify number of recommendations you need", value=10, min_value=1)
    

    val_index = output[output['title'] == option]['book_id'].values
    ids, distances = nearest_books_nms(val_index, nms_idx, count_recommendation+1)
    ids, distances = ids[1:], distances[1:]

    st.markdown('Most similar books:\n')
    df = get_recommendation_df(ids, distances, name_mapper, author_mapper)
    st.dataframe(df[['book_name', 'book_author']])
    
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    
    fig = px.bar(
        data_frame=df,
        x='book_name',
        y='distance',
        color='distance',
        hover_data='book_author',
        title='Cosine distances to the nearest books',
        color_continuous_scale=['#87CEFA', '#8A2BE2'],
    )
    st.plotly_chart(fig)
else:
    st.markdown('Please, provide more info')


