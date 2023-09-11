# model-related
import numpy as np
import tensorflow as tf
# frontend
from dash import Dash, html, dcc, dash_table, Patch
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
# vk api as a must
import vk_api
# preprocessing
import pymorphy2
import re
from unicodedata import normalize
# misc
import json
import requests as req
import sqlite3
from urllib.parse import urlparse

STOPLIST_URL = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.json'

morph = pymorphy2.MorphAnalyzer(lang='ru')
stoplist = set(json.loads(req.get(STOPLIST_URL).text))

def purify_text(text):
    removed_punct = re.sub("[.;,!?\-'\"]+", ' ', text)
    only_rus = re.sub('[^а-яА-Я\s]+', '', removed_punct)
    return re.sub('\s+', ' ', only_rus)

def remove_stopwords_and_nf(text):
    words = text.split()
    return ' '.join(morph.parse(x)[0].normal_form for x in filter(lambda xx: xx not in stoplist, words))

# prepare text
def prepare_text(text):
    pure_text = purify_text(text)
    clean = remove_stopwords_and_nf(pure_text)
    return normalize('NFKD', clean.lower().strip())


model = tf.keras.models.load_model('model/model.keras')
model.summary()

# get vector of predictions
def get_label_num(text):
    clean_text = prepare_text(text)
    pred = model.predict([clean_text])
    return np.argmax(pred[0,:]) + 1

topics = {}

with sqlite3.connect('datasets/posts.db') as conn:
    cur = conn.execute('select id, name from labels')
    for row in cur:
        topics[int(row[0])] = row[1]

text = 'привет абитуриент олимпиада пройдет в школе пенал не понадобится'
topic_num = get_label_num(text)
print(f'topic("{text}") = "{topics[topic_num]}"')

# vk api stuff
# could've stored in the envvar but...
token = open('token.txt', 'r').read()
version = 5.131

vk_session = vk_api.VkApi(token=token)
vk = vk_session.get_api()

# https://vk.com/jumoreski?w=wall-92876084_465495
def get_post_id(post_url):
    res = urlparse(post_url)
    if res.scheme == 'https' and res.hostname == 'vk.com' and res.query != '':
        query = res.query
        m = re.search('w=wall(-?[0-9]+_[0-9]+)', query)
        if m:
            return m.group(1)

    raise RuntimeError('Invalid post URL!')

def get_post_text(post_url):
    post_id = get_post_id(post_url)
    resp = vk.wall.getById(posts=post_id, v=version)
    return resp[0]['text']

#print(get_post_text('https://vk.com/jumoreski?w=wall-92876084_465495'))

# frontend time!
