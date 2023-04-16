import json
import string
import streamlit as st
from streamlit_lottie import st_lottie
import tensorflow as tf
import numpy as np
import pickle
from nltk.corpus import stopwords 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.stem.porter import PorterStemmer
import nltk
pr = PorterStemmer()
nltk.download('stopwords')

st.set_page_config(
  page_title="SMS Spam Classifier",
  layout="wide"
)

@st.cache_data
def load_image_json(path):
    """ Load animation and images from json """
    with open(path, 'r') as j:
        animation = json.loads(j.read())
        return animation

back = load_image_json('Assets/ai.json')

res = None

max_len = 100
# voca_size = 10000
# oov_tok = "<OOV>"

model = load_model('ML/BiLSTM.h5')
# Load the tokenizer from the file
with open('ML/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def inputer(text):
    nopun = [chir for chir in text if chir not in string.punctuation]
    nopun = ''.join(nopun)
    # find the root word  ad removing Stop words
    with_stem_words = [pr.stem(word) for word in nopun.split() if word.lower() not in stopwords.words('english')]
    text_seq = tokenizer.texts_to_sequences([with_stem_words])
    text_pad_seq =  pad_sequences(text_seq, maxlen=max_len, padding='post', truncating='post')
    predict(text_pad_seq)
    


def predict(pad_seq):
    global res
    prediction = model.predict([pad_seq])
    print(prediction[0])
    print(np.round(prediction))
    # print("asdadada-------"+str(output))
    # Return the predicted class label
    if np.round(prediction).any() == 1:
        res = 1
    else:
        res = 0

def reset():
   global res
   print(res)
   res = None

def input_container():
    st.write("# SMS Spam Classifier")
    text = st.text_area('', placeholder="Enter the sms to classify",on_change=reset())
    st.button('Submit',type='primary',on_click=inputer(text))

c1,c2 = st.columns(2)

with st.container():
    
    with c1:
        input_container()
    
        if res == None:
            st.empty()
        if res == 1:
            st.error("This text is a spam message!",icon="🚨")
        if res == 0:
            st.success("This text is not a spam message",icon='✅')

        with st.expander("See spam sms example"):
            st.write("""
                   *  Congrats! You are the lucky winner of a $1000 gift card from Amazon. Click here to claim your prize now: bit.ly/xxxxxx
                   *  You have been pre-approved for a low-interest loan of up to $10,000. No credit check required. Apply now: loan.com.spam/xxxxxx
                   *  Your FedEx package with tracking code GB-6412-GH83 is waiting for you to set delivery preferences: b3c7f.info/xxxxxx
                   *  Hi, this is John from Apple Support. We have detected some unusual activity on your iCloud account. Please log in here to confirm your details: apple.com.fake/xxxxxx
                   *  You have been selected for a special offer from Netflix. Get 3 months of free streaming with this exclusive link: netflix.com.spam/xxxxxx
                """)
    with c2:
        st_lottie(back, speed=1, loop=True, quality="high")
