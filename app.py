import streamlit as st
import pandas as pd
import tensorflow as tf
import utils
import pickle

model = tf.keras.models.load_model('model.keras')

with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)

st.write("# Sentiment Analysis Project")
user_input = st.text_area("Write a sentece about your feelings right now")

col1, col2 = st.columns([0.87, 0.13])
with col2:
        analyze_button = st.button("Analyze") 
user_input = utils.text_preprocessing(user_input)

if analyze_button and user_input:
        data = pd.Series(user_input)
        data = utils.data_preparation(data, tokenizer)
        print(data)

        prediction=model.predict(data)
        result = prediction.max()
        if (result == prediction[0][0]):
                st.write('You are feeling sad today! \U0001F641')
        elif (result == prediction[0][1]):
                st.write('You are feeling joy today! \U0001F602')
        elif (result == prediction[0][2]):
                st.write('You are feeling loved today! \U00002764')
        elif (result == prediction[0][3]):
                st.write('You are feeling anger today! \U0001F620')
        elif (result == prediction[0][4]):
                st.write('You are feeling fear today! \U0001F628 ')
        else:
                st.write('You are surprised! \U0001F62E')
else:
        pass

