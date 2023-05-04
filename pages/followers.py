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
st.set_page_config(page_title="Followers - Tiktok Analytics Dashboard", page_icon = "📊", layout = "centered", initial_sidebar_state = "auto")

st.header("Followers")
st.markdown("""Upload your files here to load your data! 

*'Follower activity', 'Top territories', 'Gender', 'Total followers' (xlsx or csv format)*
""")

uploaded_files = st.file_uploader(
    "Choose CSV or Excel files to upload",
    accept_multiple_files=True,
    type=['csv', 'xlsx'])

if uploaded_files:
    data_list = []   
    # read the file
    with st.expander("View uploaded data"):
        for uploaded_file in uploaded_files:
            st.write("▾ Filename:", uploaded_file.name)
            bytes_data = uploaded_file.read()
            data = None
            if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                data = pd.read_excel(io.BytesIO(bytes_data))
                AgGrid(data)
            else:
                data = pd.read_csv(io.StringIO(bytes_data.decode('utf-8')))
                AgGrid(data)

            data_list.append(data)

    tab1, tab2, tab3, tab4 = st.tabs(["Follower Activity", "Gender", "Top Territories", "Followers"])
    for data in data_list:
        #st.write(data.columns)
        #st.write(data)
        with tab1:
            if 'Active followers' in data.columns:
                # create a list of all unique dates in the data
                unique_dates = data['Date'].unique()

                # Add a date filter widget
                filter_type = st.sidebar.radio("Select filter type for Follower Activity", ["Individual date", "Total date range"])
                if filter_type == "Individual date":
                    selected_date = st.sidebar.selectbox("Select a date", unique_dates)

                    # Filter the data to only include the selected date
                    filtered_data = data[data['Date'] == selected_date]

                    # Calculate the average number of followers for the selected date
                    avg_followers = filtered_data['Active followers'].mean()


                elif filter_type == "Total date range":
                    # Filter the data to include the entire date range
                    filtered_data = data
                    # Calculate the average number of followers for the entire date range
                    avg_followers = filtered_data['Active followers'].mean()
                
                #else:
                    # Get the start and end dates using st.date_input
                    #start_date = st.sidebar.date_input("Select start date", data['Date'].min(), data['Date'].max())
                    #end_date = st.sidebar.date_input("Select end date", data['Date'].min(), data['Date'].max())

                    # Filter the data based on the start and end dates
                    #filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

                # create a datetime column from the date and hour columns
                filtered_data['Datetime'] = pd.to_datetime(filtered_data['Date'] + ' ' + (filtered_data['Hour'] - 1).astype(str) + ':00:00')

                # group the data by the datetime column and calculate the sum of active followers
                grouped_data = filtered_data.groupby("Datetime")["Active followers"].sum().reset_index()

                # create a line chart using Plotly Express
                fig = px.line(filtered_data, x="Datetime", y="Active followers", title="Follower Activity")

                # Add average line
                fig.add_shape(type='line', x0=filtered_data['Datetime'].min(), y0=avg_followers, x1=filtered_data['Datetime'].max(), y1=avg_followers, line=dict(color='red', width=3, dash='dash'))

                # Annotate average value onto average line
                fig.add_annotation(x=filtered_data['Datetime'].max(), y=avg_followers, text=f"Average: {avg_followers:.0f}", showarrow=True, arrowhead=2)
                
                st.plotly_chart(fig)
        with tab2:
            if 'Gender' in data.columns:
            #st.write("Pie chart for 'Gender'")
                gender_data = data.groupby('Gender')['Distribution'].apply(lambda x: pd.to_numeric(x.str.replace('%', ''), errors='coerce').dropna().mean()).reset_index()
                fig = px.pie(gender_data, values='Distribution', names='Gender', title='Gender Distribution (%)',
                        color_discrete_sequence=['#1f77b4', '#ff7f0e'])
                st.plotly_chart(fig)
        with tab3:
            if 'Top territories' in data.columns:
                territories_data = data.groupby('Top territories')['Distribution'].apply(lambda x: pd.to_numeric(x.str.replace('%', ''), errors='coerce').dropna().mean()).reset_index()
                territories_data = territories_data.sort_values(by='Distribution', ascending=True)
            #st.write("Top 5 territories by distribution")
                fig = px.bar(territories_data, x='Distribution', y='Top territories', title='Distribution (%) of Top 5 Countries',
                            color_discrete_sequence=px.colors.qualitative.Dark2)
                st.plotly_chart(fig)
        with tab4:
            if 'Followers' in data.columns:
                fig = px.line(data, x="Date", y="Followers", title="Total Followers", markers=True,
                hover_data={'Followers': ':.2f'})
                st.plotly_chart(fig)


        