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
import re
import emoji
from collections import Counter
from st_pages import Page, show_pages, add_page_title

# Set page title
st.set_page_config(page_title="Content - Tiktok Analytics Dashboard", page_icon = "📊", layout = "centered", initial_sidebar_state = "auto")

st.header("Content")
st.markdown("""Upload your files here to load your data!

*'Trending videos' (xlsx or csv format)*
""")

uploaded_files = st.file_uploader(
    "Choose CSV or Excel files to upload",
    accept_multiple_files=True,
    type=['csv', 'xlsx'])

if uploaded_files:
    data_list = []
    
        # read the file
    with st.expander("View uploaded data - data covered over 7-day period only"):
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
        #st.write(data_list)

    #tab1, tab2 = st.tabs(["Trending Videos", "Video Posts"])
    for data in data_list:
        #st.write(data.columns)
        #st.write(data)
        #with tab1:
            if 'video_view_within_days' in data.columns: #Trending Videos
                # Extract hashtags using a regex pattern
                def extract_hashtags(title):
                    return re.findall(r'#\w+', title)

                # Extract emojis using the emoji library
                #def extract_emojis(title):
                    #return [c for c in title if c in emoji.emojize(c)]
                
                # Remove emojis and hashtags from the title
                def clean_title(title):
                    #title_without_emojis = emoji.get_emoji_regexp().sub(u'', title)
                    title_without_hashtags = re.sub(r'#\w+', '', title)
                    return title_without_hashtags.strip()

                data['Hashtags'] = data['Video title'].apply(extract_hashtags)
                #data['Emojis'] = data['Video title'].apply(extract_emojis)
                data['Cleaned_title'] = data['Video title'].apply(clean_title)
                # Add a new column to store the hashtag count
                data['Hashtag_count'] = data['Hashtags'].apply(len)
                st.write(data)
                options = ["Total views", "Total shares", "Total likes", "Total comments", "Hashtags"]
                selected_feature = st.selectbox(label="Select feature", options=options, index=0)
                if selected_feature == "Total views":
                    data = data.sort_values(by='Total views', ascending=True)
                    fig = px.bar(data, x='Total views', y='Cleaned_title', title='Views of trending videos for the week',
                            color_discrete_sequence=px.colors.qualitative.Alphabet, hover_data={'Total views': ':.2f'})
                    st.plotly_chart(fig)
                elif selected_feature == "Total shares":
                    data = data.sort_values(by='Total shares\xa0', ascending=True)
                    fig = px.bar(data, x='Total shares\xa0', y='Cleaned_title', title='Shares of trending videos for the week',
                            color_discrete_sequence=px.colors.qualitative.Set1, hover_data={'Total shares\xa0': ':.2f'})
                    st.plotly_chart(fig)
                elif selected_feature == "Total likes":
                    data = data.sort_values(by='Total likes', ascending=True)
                    fig = px.bar(data, x='Total likes', y='Cleaned_title', title='Likes of trending videos for the week',
                            color_discrete_sequence=px.colors.qualitative.Antique, hover_data={'Total likes': ':.2f'})
                    st.plotly_chart(fig)
                elif selected_feature == "Total comments":
                    data = data.sort_values(by='Total comments', ascending=True)
                    fig = px.bar(data, x='Total comments', y='Cleaned_title', title='Comments of trending videos for the week',
                            color_discrete_sequence=px.colors.qualitative.Vivid, hover_data={'Total comments': ':.2f'})
                    st.plotly_chart(fig)
                elif selected_feature == "Hashtags":
                    # Count the occurrences of each hashtag
                    hashtag_counts = Counter(hashtag for hashtags in data['Hashtags'] for hashtag in hashtags)

                    # Get the top N most common hashtags
                    N = 10
                    top_hashtags = hashtag_counts.most_common(N)

                    # Display the top hashtags
                    print(f"Top {N} hashtags:")
                    for hashtag, count in top_hashtags:
                        print(f"{hashtag}: {count}")

                    # Visualize the results with a Plotly bar chart
                    fig = go.Figure(go.Bar(
                        x=[t[0] for t in top_hashtags],
                        y=[t[1] for t in top_hashtags],
                        text=[t[1] for t in top_hashtags],
                        textposition='auto',
                        marker_color='rgba(58, 71, 80, 0.6)',
                        opacity=0.8
                    ))

                    fig.update_layout(
                        title=f'Top {N} Hashtags',
                        xaxis_title='Hashtags',
                        yaxis_title='Count',
                        xaxis_tickangle=-45
                    )

                    st.plotly_chart(fig)

                    tab1, tab2, tab3, tab4 = st.tabs(["vs Views", "vs Shares", "vs Likes", "vs Comments"])
                    with tab1:
                        fig = px.scatter(data, x='Hashtag_count', y='Total views', hover_data=['Cleaned_title'])
                        fig.update_layout(
                            title='Hashtag Count vs. Views',
                            xaxis_title='Hashtag Count',
                            yaxis_title='Views'
                        )
                        st.plotly_chart(fig)
                    with tab2:
                        fig = px.scatter(data, x='Hashtag_count', y='Total shares\xa0', hover_data=['Cleaned_title'])
                        fig.update_layout(
                            title='Hashtag Count vs. Shares',
                            xaxis_title='Hashtag Count',
                            yaxis_title='Shares'
                        )
                        st.plotly_chart(fig)
                    with tab3:
                        fig = px.scatter(data, x='Hashtag_count', y='Total likes', hover_data=['Cleaned_title'])
                        fig.update_layout(
                            title='Hashtag Count vs. Likes',
                            xaxis_title='Hashtag Count',
                            yaxis_title='Likes'
                        )
                        st.plotly_chart(fig)
                    with tab4:
                        fig = px.scatter(data, x='Hashtag_count', y='Total comments', hover_data=['Cleaned_title'])
                        fig.update_layout(
                            title='Hashtag Count vs. Comments',
                            xaxis_title='Hashtag Count',
                            yaxis_title='Comments'
                        )
                        st.plotly_chart(fig)


        #with tab2:
            #if 'video_view_within_days' not in data.columns: #Video Posts
                #pass
        