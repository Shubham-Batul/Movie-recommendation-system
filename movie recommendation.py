#!/usr/bin/env python
# coding: utf-8

# # Importing dataset

# In[1]:


import pandas as pd


movies = pd.read_csv("movies.csv")


# In[43]:


movies


# # Cleaning title

# In[56]:


import re

def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title


# In[57]:


movies["clean_title"] =  movies["title"].apply(clean_title)


# In[58]:


movies


# In[ ]:





# In[59]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf = vectorizer.fit_transform(movies["clean_title"])


# In[60]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    
    return results


# # Creating search widget 

# In[61]:


# pip install ipywidgets


# In[62]:


import ipywidgets as widgets
from IPython.display import display

movie_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
movie_list = widgets.Output()

def on_type(data):
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            display(search(title))

movie_input.observe(on_type, names='value')


display(movie_input, movie_list)


# # Creating recommendation system 
# 
# Works based on ratings given to a movie by other users

# In[63]:


movie_id = 89745


movie = movies[movies["movieId"] == movie_id]


# In[64]:


ratings = pd.read_csv("ratings.csv")


# In[65]:


ratings


# In[74]:


ratings.dtypes


# In[75]:


similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()


# In[76]:


similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]


# In[77]:


similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

similar_user_recs = similar_user_recs[similar_user_recs > .10]


# In[78]:


all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]


# In[79]:


all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())


# In[80]:


rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
rec_percentages.columns = ["similar", "all"]


# In[82]:


rec_percentages


# In[83]:


rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]


# In[84]:


rec_percentages = rec_percentages.sort_values("score", ascending=False)


# In[85]:


rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")


# In[86]:


def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]


# In[87]:


import ipywidgets as widgets
from IPython.display import display

movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

movie_name_input.observe(on_type, names='value')

display(movie_name_input, recommendation_list)


# In[ ]:




