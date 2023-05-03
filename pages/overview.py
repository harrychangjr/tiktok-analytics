import streamlit as st
import pandas as pd
from st_aggrid import AgGrid
import io
from st_pages import Page, show_pages, add_page_title

st.header("Overview")
st.markdown("Upload your files here to load your data!")

uploaded_files = st.file_uploader(
    "Choose CSV or Excel files to upload",
    accept_multiple_files=True,
    type=['csv', 'xlsx'])

if uploaded_files:
    data_list = []
    for uploaded_file in uploaded_files:
        # read the file
        st.write("▾ Filename:", uploaded_file.name)
        bytes_data = uploaded_file.read()
        data = None
        if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            data = pd.read_excel(io.BytesIO(bytes_data))
            AgGrid(data)
        else:
            data = pd.read_csv(io.StringIO(bytes_data.decode('utf-8')))
            AgGrid(data)

        # preview the data
        #st.write('Preview of', uploaded_file.name)
        # st.write(data)

        data_list.append(data)