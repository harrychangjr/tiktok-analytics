import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from millify import millify
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from st_aggrid import AgGrid
import io
from st_pages import Page, show_pages, add_page_title

# Set page title
st.set_page_config(page_title="Content - Tiktok Analytics Dashboard", page_icon = "📊", layout = "centered", initial_sidebar_state = "auto")

st.header("Content")
st.markdown("""Upload your files here to load your data!

*'Trending videos', 'Video Posts' (xlsx or csv format)*
""")

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