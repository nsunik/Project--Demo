import streamlit as st
import classify
import time
from PIL import Image
import numpy as np

st.set_page_config(layout="wide", page_title="Satellite Image Classification", page_icon="")

st.title("Satellite Image Classification[Tensorflow]")
c1, c2 = st.columns(2)
with c1:
    uploaded_file = st.file_uploader("Please Upload Images ", accept_multiple_files=False)
    class_names = ['Severe Tropical Storm_SS', 'Tropical Depression_D', 'Tropical Strom_S', 'Typhoon_T']
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if st.button('   Submit  '):
            results = classify.predict(image)
            with st.spinner('Loading Result...'):
                time.sleep(2)
                st.markdown("This Image most likely belongs to")
                st.subheader(
                    " {} with a {:.2f} percent confidence."
                        .format(class_names[np.argmax(results)], 100 * np.max(results)))
                st.write(results)
        

if uploaded_file is None:
    st.header("Please Upload Images")
with c2:
    if uploaded_file is not None:
        st.image(uploaded_file)
    else:
        st.subheader("")


# Conclusion
