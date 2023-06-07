import streamlit as st
import pickle
import numpy as np

model = pickle.load(open(r"naval.pkl", 'rb'))

st.title('Epm Value Predictor')

cols = []


# GT Compressor decay state coefficient
drop_kMc = st.selectbox("GT Compressor Decay State Coefficient", [1.0, 0.95])

# GT Turbine decay state coefficient
drop_kMt = st.selectbox("GT Turbine Decay State Coefficient", [1.0, 0.975])

for i in range(16):
    feature = st.number_input(f"Enter feature {i+1}", value=0.0, step=0.1)
    cols.append(feature)

if st.button('Predict values'):
    features = np.array(cols).reshape(1, -1)
    prediction = model.predict(features)

    st.write('Predicted values:')
    st.write(prediction)

# Embedding YouTube video
# Replace with your YouTube video URL or ID
video_url = "https://screenrec.com/share/XB9xad7gJH"
st.video(video_url)
