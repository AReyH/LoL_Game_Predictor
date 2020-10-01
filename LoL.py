from tensorflow import keras
import streamlit as st
import numpy as np
import joblib

st.title("FF at 15?")
st.markdown("### By Arturo Rey")

scaler = joblib.load('scaler1.save') 

time = st.number_input('Input the number of minutes it has been since the game started:')

firstBlood = st.selectbox('Who got First Blood?', ['Red Team', 'Blue Team'])

blueKills = st.number_input('How many kills does Blue Team have?')

blueAssists = st.number_input('How many assists does Blue Team have?')

blueDragons = st.number_input('How many Dragons does Blue Team have?')

blueCS_min = 1

blueCS_min = st.number_input('How much CS does Blue Team have?')

redKills = st.number_input('How many kills does Red Team have?')

redAssists = st.number_input('How many assists does Red Team have?')

redDragons = st.number_input('How many Dragons does Red Team have?')

redCS_min = 1

redCS_min = st.number_input('How much CS does Red Team have?')

if firstBlood == "Blue Team":
    firstBlood = 1
else:
    firstBlood = 0

if blueCS_min > 1:  # This is to avoid the model running and crashing before inputing values.
    cs_min_blue = blueCS_min/time
    cs_min_red = redCS_min/time

    result = np.array([firstBlood, int(blueKills), int(blueAssists), int(blueDragons), float(cs_min_blue), int(redKills), int(redAssists), int(redDragons), float(cs_min_red)])
    result_ = result.reshape(1,-1)
    result__ = scaler.transform(result_)
    #st.write(result)
    n_model = keras.models.load_model('LoLSt.h5',custom_objects=None,compile=False)

    prediction = n_model.predict(result__)

    if prediction > 0.5:
        st.title("Blue Team wins.")
    else:
        st.title('Red Team wins.')