# -*- coding: utf-8 -*-
"""anime_fix.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WI1lcG2YL2OSVxzU5oDQG-Tq5d1tXldH

# **Prepare Dataset**
"""

!pip install opendatasets

# Commented out IPython magic to ensure Python compatibility.
import opendatasets as od
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as pylab
# %matplotlib inline
pd.set_option('display.max_columns', 500)
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

import warnings
warnings.filterwarnings('ignore')

od.download('https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database')

"""# **Data Understanding**"""

anime_data=pd.read_csv('/content/anime-recommendations-database/anime.csv')
rating_data=pd.read_csv('/content/anime-recommendations-database/rating.csv')

print ("The shape of the  data is (row, column):"+ str(anime_data.shape))
print (anime_data.info())

print ("The shape of the  data is (row, column):"+ str(rating_data.shape))
print (rating_data.info())

anime_fulldata=pd.merge(anime_data,rating_data,on='anime_id',suffixes= ['', '_user'])
anime_fulldata = anime_fulldata.rename(columns={'name': 'anime_title', 'rating_user': 'user_rating'})

anime_fulldata.head()

anime_fulldata.shape

anime_fulldata = anime_fulldata.loc[anime_fulldata['user_id'] <= 10000, ['anime_id','anime_title','genre','type','episodes','rating','members','user_id','user_rating']]

anime_fulldata.shape

"""# **Data Exploratory**"""

combine_anime_rating = anime_fulldata.dropna(axis = 0, subset = ['anime_title'])
anime_ratingCount = (combine_anime_rating.
     groupby(by = ['anime_title'])['user_rating'].
     count().
     reset_index().rename(columns = {'rating': 'totalRatingCount'})
    [['anime_title', 'user_rating']]
    )

top10_animerating=anime_ratingCount[['anime_title', 'user_rating']].sort_values(by = 'user_rating',ascending = False).head(10)
ax=sns.barplot(x="anime_title", y="user_rating", data=top10_animerating, palette="Dark2")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40, ha="right")
ax.set_title('Top 10 Anime based on rating counts',fontsize = 22)
ax.set_xlabel('Anime',fontsize = 20)
ax.set_ylabel('User Rating count', fontsize = 20)

duplicate_anime=anime_fulldata.copy()
duplicate_anime.drop_duplicates(subset ="anime_title",
                     keep = 'first', inplace = True)

top10_animemembers=duplicate_anime[['anime_title', 'members']].sort_values(by = 'members',ascending = False).head(10)
ax=sns.barplot(x="anime_title", y="members", data=top10_animemembers, palette="gnuplot2")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40, ha="right")
ax.set_title('Top 10 Anime based on members',fontsize = 22)
ax.set_xlabel('Anime',fontsize = 20)
ax.set_ylabel('Community Size', fontsize = 20)

plt.figure(figsize = (15, 7))
plt.subplot(1,2,1)
anime_fulldata['rating'].hist(bins=70)
plt.title("Rating of websites")
plt.subplot(1,2,2)
anime_fulldata['user_rating'].hist(bins=70)
plt.title("Rating of users")

nonull_anime=anime_fulldata.copy()
nonull_anime.dropna(inplace=True)
from collections import defaultdict

all_genres = defaultdict(int)

for genres in nonull_anime['genre']:
    for genre in genres.split(','):
        all_genres[genre.strip()] += 1

from wordcloud import WordCloud

genres_cloud = WordCloud(width=800, height=400, background_color='white', colormap='gnuplot').generate_from_frequencies(all_genres)
plt.imshow(genres_cloud, interpolation='bilinear')
plt.axis('off')

"""# **Data Preparation**

**Handling NaN values**
"""

anime_feature=anime_fulldata.copy()
anime_feature["user_rating"].replace({-1: np.nan}, inplace=True)
anime_feature.head()

anime_feature = anime_feature.dropna(axis = 0, how ='any')
anime_feature.isnull().sum()

"""**Filtering user_id**"""

anime_feature['user_id'].value_counts()

counts = anime_feature['user_id'].value_counts()
anime_feature = anime_feature[anime_feature['user_id'].isin(counts[counts >= 200].index)]

"""**Pivot**"""

anime_pivot=anime_feature.pivot_table(index='anime_title',columns='user_id',values='user_rating').fillna(0)
anime_pivot.head()

"""# **Content Based Filtering**

**Cleaning Anime title**
"""

import re
def text_cleaning(text):
    """
    Function to clean text by removing specific patterns.

    Parameters:
    text : str
        The text to be cleaned.

    Returns:
    str
        The cleaned text.
    """
    text = re.sub(r'&quot;', '', text)
    text = re.sub(r'.hack//', '', text)
    text = re.sub(r'&#039;', '', text)
    text = re.sub(r'A&#039;s', '', text)
    text = re.sub(r'I&#039;', 'I\'', text)
    text = re.sub(r'&amp;', 'and', text)

    return text

anime_data['name'] = anime_data['name'].apply(text_cleaning)

"""**Term Frequency (TF) and Inverse Document Frequency (IDF)**"""

from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

anime_data['genre'] = anime_data['genre'].fillna('')
genres_str = anime_data['genre'].str.split(',').astype(str)
tfv_matrix = tfv.fit_transform(genres_str)

tfv_matrix.shape

from sklearn.metrics.pairwise import sigmoid_kernel

sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

indices = pd.Series(anime_data.index, index=anime_data['name']).drop_duplicates()

"""**Recommendation function**"""

def give_rec(title, sig=sig):
    """
    Function to provide recommendations based on cosine similarity.

    Parameters:
    title : str
        The title of the anime for which recommendations are sought.
    sig : numpy.ndarray, optional
        The cosine similarity matrix. Default is 'sig'.

    Returns:
    pandas.DataFrame
        DataFrame containing the top 10 most similar anime along with their ratings.
    """
    idx = indices[title]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:11]
    anime_indices = [i[0] for i in sig_scores]
    return pd.DataFrame({'Anime name': anime_data['name'].iloc[anime_indices].values,
                                 'Rating': anime_data['rating'].iloc[anime_indices].values})

give_rec('Dragon Ball Kai')

give_rec("Death Note")

"""# **Collaborative Filtering**

**Use column user_id,anime_id, and user_rating**

**only 1000 user**
"""

ratings = anime_fulldata.loc[(anime_fulldata['user_id'] <= 1000) & (anime_fulldata['anime_id'] <= 1500), ['user_id', 'anime_id', 'user_rating']]

ratings.head()

ratings.shape

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

"""**Data Preparation**"""

user_ids = ratings['user_id'].unique().tolist()
print('list userID: ', user_ids)

user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded userID : ', user_to_user_encoded)

user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke userID: ', user_encoded_to_user)

anime_ids = ratings['anime_id'].unique().tolist()
anime_to_anime_encoded = {x: i for i, x in enumerate(anime_ids)}
anime_encoded_to_anime = {i: x for i, x in enumerate(anime_ids)}

ratings['user'] = ratings['user_id'].map(user_to_user_encoded)
ratings['anime'] = ratings['anime_id'].map(anime_to_anime_encoded)

num_users = len(user_to_user_encoded)
print(num_users)
num_anime = len(anime_encoded_to_anime)
print(num_anime)
ratings['rating'] = ratings['user_rating'].values.astype(np.float32)
min_rating = min(ratings['rating'])
max_rating = max(ratings['rating'])

print('Number of User: {}, Number of Anime: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_anime, min_rating, max_rating
))

df = ratings.sample(frac=1, random_state=42)
df

x = df[['user', 'anime']].values
y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x, y)

"""**Training**"""

class RecommenderNet(tf.keras.Model):
  """
    RecommenderNet class for building a recommendation system model.

    Args:
        num_users (int): Number of users in the system.
        num_anime (int): Number of anime items in the system.
        embedding_size (int): Size of the embedding vectors for users and anime.
        **kwargs: Additional keyword arguments.

    Attributes:
        num_users (int): Number of users in the system.
        num_anime (int): Number of anime items in the system.
        embedding_size (int): Size of the embedding vectors for users and anime.
        user_embedding (tf.keras.layers.Embedding): Embedding layer for users.
        user_bias (tf.keras.layers.Embedding): Embedding layer for user biases.
        resto_embedding (tf.keras.layers.Embedding): Embedding layer for anime.
        resto_bias (tf.keras.layers.Embedding): Embedding layer for anime biases.
    """
  def __init__(self, num_users, num_anime, embedding_size, **kwargs):
    """
        Initialize the RecommenderNet model.

        Args:
            num_users (int): Number of users in the system.
            num_anime (int): Number of anime items in the system.
            embedding_size (int): Size of the embedding vectors for users and anime.
            **kwargs: Additional keyword arguments.
    """
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_anime = num_anime
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding(
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1)
    self.resto_embedding = layers.Embedding(
        num_anime,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.resto_bias = layers.Embedding(num_anime, 1)

  def call(self, inputs):
    """
        Forward pass of the RecommenderNet model.

        Args:
            inputs (tf.Tensor): Input tensor containing user and anime indices.

        Returns:
            tf.Tensor: Output tensor with sigmoid activation applied.
    """
    user_vector = self.user_embedding(inputs[:,0])
    user_bias = self.user_bias(inputs[:, 0])
    resto_vector = self.resto_embedding(inputs[:, 1])
    resto_bias = self.resto_bias(inputs[:, 1])

    dot_user_resto = tf.tensordot(user_vector, resto_vector, 2)

    x = dot_user_resto + user_bias + resto_bias

    return tf.nn.sigmoid(x)

model = RecommenderNet(num_users, num_anime, 50)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

"""**Training Model**"""

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 8,
    epochs = 30,
    validation_data = (x_val, y_val)
)

"""**Metrik**"""

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""**Evaluation**"""

fig, ax = plt.subplots(2, figsize=(16, 8))

mt = history.history['root_mean_squared_error']
mv = history.history['val_root_mean_squared_error']

ax[0].plot(mt)
ax[0].plot(mv)

for plot in ax.flat:
    plot.set(xlabel='rmse', ylabel='val-rmse')

plt.show()

anime_new = anime_fulldata.loc[anime_fulldata['user_id'] <= 1000, ['anime_id', 'anime_title', 'genre']]

anime_df = anime_new
df = ratings

user_id = df.user_id.sample(1).iloc[0]
anime_visited_by_user = df[df.user_id == user_id]

anime_not_visited = anime_df[~anime_df['anime_id'].isin(anime_visited_by_user.anime_id.values)]['anime_id']
anime_not_visited = list(
    set(anime_not_visited)
    .intersection(set(anime_to_anime_encoded.keys()))
)

resto_not_visited = [[anime_to_anime_encoded.get(x)] for x in anime_not_visited]
user_encoder = user_to_user_encoded.get(user_id)
user_resto_array = np.hstack(
    ([[user_encoder]] * len(resto_not_visited), resto_not_visited)
)

"""**Get Recommendation**"""

ratings = model.predict(user_resto_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_anime_ids = [
    anime_encoded_to_anime.get(resto_not_visited[x][0]) for x in top_ratings_indices
]

print('Showing recommendations for user: {}'.format(user_id))
print('=' * 9)
print('Anime with high ratings from user')
print('----' * 8)

top_anime_user = (
    anime_visited_by_user.sort_values(
        by='rating',
        ascending=False
    )
    .head(5)
    .anime_id.values
)

anime_df_rows = anime_df[anime_df['anime_id'].isin(top_anime_user)]
for row in anime_df_rows.itertuples():
    print(row.anime_title, ':', row.genre)

print('Top 10 Anime recommendations')
print('----' * 8)

for anime_id in recommended_anime_ids[:10]:
    recommended_anime_info = anime_df[anime_df['anime_id'] == anime_id].iloc[0]
    print(recommended_anime_info.anime_title, ':', recommended_anime_info.genre)
