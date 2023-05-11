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
import openpyxl
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#from st_pages import Page, show_pages, add_page_title

# Set page title
st.set_page_config(page_title="Content - Tiktok Analytics Dashboard", page_icon = "📊", layout = "centered", initial_sidebar_state = "auto")

st.header("Content")
st.markdown("""Upload your files here to load your data!

*'Trending videos' (xlsx or csv format)*
""")

def plot_chart(data, chart_type, x_var, y_var, z_var=None, show_regression_line=False, show_r_squared=False):
    scatter_marker_color = 'green'
    regression_line_color = 'red'
    if chart_type == "line":
        fig = px.line(data, x=x_var, y=y_var)

    elif chart_type == "bar":
        fig = px.bar(data, x=x_var, y=y_var)

    elif chart_type == "scatter":
        fig = px.scatter(data, x=x_var, y=y_var, color_discrete_sequence=[scatter_marker_color])

        if show_regression_line:
            X = data[x_var].values.reshape(-1, 1)
            y = data[y_var].values.reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            r_squared = r2_score(y, y_pred)  # Calculate R-squared value

            fig.add_trace(
                go.Scatter(x=data[x_var], y=y_pred[:, 0], mode='lines', name='Regression Line', line=dict(color=regression_line_color))
            )

            # Add R-squared value as a text annotation
            fig.add_annotation(
                x=data[x_var].max(),
                y=y_pred[-1, 0],
                text=f"R-squared: {r_squared:.4f}",
                showarrow=False,
                font=dict(size=14),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1,
                borderpad=4
            )

    elif chart_type == "heatmap":
        fig = px.imshow(data, color_continuous_scale='Inferno')

    elif chart_type == "scatter_3d":
        if z_var is not None:
            fig = px.scatter_3d(data, x=x_var, y=y_var, z=z_var, color=data.columns[0])
        else:
            st.warning("Please select Z variable for 3D line plot.")
            return

    elif chart_type == "line_3d":
        if z_var is not None:
            fig = go.Figure(data=[go.Scatter3d(x=data[x_var], y=data[y_var], z=data[z_var], mode='lines')])
            fig.update_layout(scene=dict(xaxis_title=x_var, yaxis_title=y_var, zaxis_title=z_var))  # Set the axis name
        else:
            st.warning("Please select Z variable for 3D line plot.")
            return

    elif chart_type == "surface_3d":
        if z_var is not None:
            fig = go.Figure(data=[go.Surface(z=data.values)])
            fig.update_layout(scene=dict(xaxis_title=x_var, yaxis_title=y_var, zaxis_title=z_var))  # Set the axis name
        else:
            st.warning("Please select Z variable for 3D line plot.")
            return

    elif chart_type == "radar":
        fig = go.Figure()
        for col in data.columns[1:]:
            fig.add_trace(go.Scatterpolar(r=data[col], theta=data[x_var], mode='lines', name=col))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[data[data.columns[1:]].min().min(), data[data.columns[1:]].max().max()])))

    st.plotly_chart(fig)

def plot_radar_chart(data, columns):
    df = data[columns]
    fig = go.Figure()

    for i in range(len(df)):
        date_label = data.loc[i, 'Date']
        fig.add_trace(go.Scatterpolar(
            r=df.loc[i].values,
            theta=df.columns,
            fill='toself',
            name=date_label
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, df.max().max()]
            )
        ),
        showlegend=True
    )

    st.plotly_chart(fig)

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
                options = ["Total views", "Total shares", "Total likes", "Total comments", "Number of Hashtags", "Hashtag Performance"]
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
                elif selected_feature == "Number of Hashtags":
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
                elif selected_feature == "Hashtag Performance":
                    # Tokenize hashtags and create a list of unique hashtags
                    tokenized_hashtags = data["Hashtags"].tolist()
                    unique_hashtags = list(set([tag for tags in tokenized_hashtags for tag in tags]))
                    # Train a word2vec model
                    model = Word2Vec(tokenized_hashtags, vector_size=50, window=5, min_count=1, workers=4)

                    # Create a hashtag vector dictionary
                    hashtag_vectors = {tag: model.wv[tag] for tag in unique_hashtags}
                    st.subheader("Explaining the concept of hashtag performance scores and cosine similarity scores")
                    st.markdown(
                        """
                    So how are the **performance scores** calculated for each feature? 

                    In each line, the code goes through each unique hashtag and selects all videos that use that hashtag. It then calculates the mean of the respective performance metric (views, shares, likes, or comments) for those videos.

                    This gives an average performance score for each hashtag, which can be used as an indication of how well videos with that hashtag tend to perform on average. However, this is a simplistic metric and there may be other factors influencing the performance of a video. It's also worth noting that the mean is sensitive to extreme values, so a few very popular or unpopular videos could skew the average performance for a given hashtag.
                    """)

                    st.markdown("""
                    How about **cosine similarity**?

                    Cosine similarity is a metric used to measure how similar two vectors are, irrespective of their size. In the context of Natural Language Processing (NLP), and in this case, it's used to measure the semantic similarity between two hashtags based on their embeddings (vectors) generated by the Word2Vec model.

                    In simple terms, it measures the cosine of the angle between two vectors. If the vectors are identical, the angle is 0, so the cosine is 1, indicating perfect similarity. If the vectors are orthogonal (i.e., the angle between them is 90 degrees), they're considered not similar, and the cosine similarity is 0. If the vectors point in opposite directions (i.e., the angle is 180 degrees), the cosine similarity is -1, indicating that they're diametrically dissimilar.
                    """)
                    # Calculate the average performance of each hashtag - views
                    hashtag_performance_views = {tag: data[data["Hashtags"].apply(lambda x: tag in x)]["Total views"].mean() for tag in unique_hashtags}

                    # Calculate the average performance of each hashtag - shares
                    hashtag_performance_shares = {tag: data[data["Hashtags"].apply(lambda x: tag in x)]["Total shares\xa0"].mean() for tag in unique_hashtags}

                    # Calculate the average performance of each hashtag - likes
                    hashtag_performance_likes = {tag: data[data["Hashtags"].apply(lambda x: tag in x)]["Total likes"].mean() for tag in unique_hashtags}

                    # Calculate the average performance of each hashtag - comments
                    hashtag_performance_comments = {tag: data[data["Hashtags"].apply(lambda x: tag in x)]["Total comments"].mean() for tag in unique_hashtags}

                    # Calculate the similarity between hashtags
                    similarity_matrix = cosine_similarity(list(hashtag_vectors.values()))

                    # Convert the similarity matrix into a DataFrame
                    similarity_df = pd.DataFrame(similarity_matrix, index=unique_hashtags, columns=unique_hashtags)

                    # Convert the performance dictionaries into DataFrames
                    perf_views_df = pd.DataFrame(list(hashtag_performance_views.items()), columns=["hashtag", "views"])
                    perf_shares_df = pd.DataFrame(list(hashtag_performance_shares.items()), columns=["hashtag", "shares"])
                    perf_likes_df = pd.DataFrame(list(hashtag_performance_likes.items()), columns=["hashtag", "likes"])
                    perf_comments_df = pd.DataFrame(list(hashtag_performance_comments.items()), columns=["hashtag", "comments"])

                    # Merge the performance DataFrames into a single DataFrame
                    perf_df = pd.merge(perf_views_df, perf_shares_df, on="hashtag")
                    perf_df = pd.merge(perf_df, perf_likes_df, on="hashtag")
                    perf_df = pd.merge(perf_df, perf_comments_df, on="hashtag")

                    # Convert the similarity matrix into a 1D series
                    similarity_series = similarity_df.unstack()

                    # Rename the series index
                    similarity_series.index.rename(["hashtag1", "hashtag2"], inplace=True)

                    # Convert the series into a DataFrame
                    similarity_df = similarity_series.to_frame("similarity").reset_index()

                    # Merge the similarity DataFrame with the performance DataFrame
                    merged_df = pd.merge(similarity_df, perf_df, left_on="hashtag1", right_on="hashtag")

                    # Calculate the correlation between hashtag similarity and performance
                    correlation = merged_df[["similarity", "views", "shares", "likes", "comments"]].corr()
                    st.subheader("Correlation matrix between hashtag cosine similarity values and performance values")
                    #st.write(correlation)
                    # Create a heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

                    # Show the plot in Streamlit
                    st.pyplot(plt)
                    # Rename the 'value' columns to make them unique
                    df1 = pd.DataFrame(list(hashtag_performance_views.items()), columns=["hashtag", "value"])
                    df2 = pd.DataFrame(list(hashtag_performance_shares.items()), columns=["hashtag", "value"])
                    df3 = pd.DataFrame(list(hashtag_performance_likes.items()), columns=["hashtag", "value"])
                    df4 = pd.DataFrame(list(hashtag_performance_comments.items()), columns=["hashtag", "value"])

                    df1.rename(columns={'value': 'value_views'}, inplace=True)
                    df2.rename(columns={'value': 'value_shares'}, inplace=True)
                    df3.rename(columns={'value': 'value_likes'}, inplace=True)
                    df4.rename(columns={'value': 'value_comments'}, inplace=True)

                    # Merge the DataFrames on the 'hashtag' column
                    merged_df = df1.merge(df2, on='hashtag').merge(df3, on='hashtag').merge(df4, on='hashtag')

                    st.subheader("Hashtag Performance Scores based on Views, Shares, Likes and Comments - Calculated using average of all videos' metrics containing a particular hashtag")
                    st.write(merged_df)

                    # Create a pair plot with regression lines
                    st.write("**Pair Plots**")
                    sns.pairplot(merged_df, kind="reg", diag_kind="kde")
                    st.pyplot(plt)

                    # Split the data into train and test sets
                    X = merged_df[["value_shares", "value_likes", "value_comments"]]
                    y = merged_df["value_views"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    #st.write("**3D Scatterplot of Predictors with Regression Plane**")
                    # Create a 3D scatterplot with a regression plane
                    #fig = plt.figure()
                    #ax = fig.add_subplot(111, projection="3d")

                    #ax.scatter(merged_df["value_shares"], merged_df["value_likes"], merged_df["value_comments"], c="blue", marker="o")

                    # Create a meshgrid for the regression plane
                    #x_range = np.linspace(merged_df["value_shares"].min(), merged_df["value_shares"].max(), num=10)
                    #y_range = np.linspace(merged_df["value_likes"].min(), merged_df["value_likes"].max(), num=10)
                    
                    #x_grid, y_grid = np.meshgrid(x_range, y_range)

                    # Calculate the regression plane
                    #X_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))
                    #plane = np.zeros(X_grid.shape[0])

                    #for i, (x, y) in enumerate(X_grid):
                        #plane[i] = model.predict(np.column_stack((x, y, merged_df["value_comments"].mean())).reshape(1, -1))

                    #z_grid = plane.reshape(x_grid.shape)

                    #ax.set_xlabel("value_shares")
                    #ax.set_ylabel("value_likes")
                    #ax.set_zlabel("value_comments")

                    #plt.show()
                    #st.pyplot(plt)

                    # Fit the models on the training data
                    lr_model = LinearRegression()
                    lr_model.fit(X_train, y_train)

                    rf_model = RandomForestRegressor()
                    rf_model.fit(X_train, y_train)

                    xgb_model = XGBRegressor()
                    xgb_model.fit(X_train, y_train)

                    # Make predictions on the testing data using the trained models
                    lr_pred = lr_model.predict(X_test)
                    rf_pred = rf_model.predict(X_test)
                    xgb_pred = xgb_model.predict(X_test)

                    # Calculate the evaluation metrics for each model
                    st.write("**Training regression models to predict value_views using value_shares, value_likes and value_comments**")
                    models = ["Linear Regression", "Random Forest", "XGBoost"]
                    predictions = [lr_pred, rf_pred, xgb_pred]

                    # Initialize a list to hold the model metrics
                    model_metrics = []

                    for model, pred in zip(models, predictions):
                        mse = mean_squared_error(y_test, pred)
                        mae = mean_absolute_error(y_test, pred)
                        r2 = r2_score(y_test, pred)
    
                        # Append a dictionary of the metrics to the list
                        model_metrics.append({"Model": model, "Mean Squared Error": mse, "Mean Absolute Error": mae, "R^2 Score": r2})
                    
                    # Convert the list of dictionaries into a DataFrame
                    metrics_df = pd.DataFrame(model_metrics)

                    # Display the DataFrame in Streamlit
                    st.write(metrics_df)


                    x_var = st.sidebar.selectbox("Select X variable", merged_df.columns)
                    y_var = st.sidebar.selectbox("Select Y variable", merged_df.columns)
                    show_regression_line = False
    
                    z_var_options = ["None"] + list(merged_df.columns)
                    z_var = st.sidebar.selectbox("Select Z variable for 3D charts (if applicable)", z_var_options)
            
            # Allow user to select time frequency for resampling
            #time_frequency = st.sidebar.selectbox("Select time frequency", ["Day", "Week", "Month"])

            #if time_frequency == "Week":
                #data_resampled = data.resample('W').sum()
            #elif time_frequency == "Month":
                #data_resampled = data.resample('M').sum()
            #else:
                #data_resampled = data
                    st.subheader("Various plots to represent performance scores for views, shares, likes and comments")
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Line", "Bar", "Scatterplot", "Heatmap", 
                                                            "3D Scatterplot", "3D Lineplot", "3D Surfaceplot"])
                    with tab1:
                        st.write("Lineplot")
                        plot_chart(merged_df, "line", x_var, y_var)

                    with tab2:
                        st.write("Barplot")
                        plot_chart(merged_df, "bar", x_var, y_var)

                    with tab3:
                        st.write("Scatterplot")
                        show_regression_line = st.checkbox("Show regression line")
                        plot_chart(merged_df, "scatter", x_var, y_var, show_regression_line=show_regression_line)

                    with tab4:
                        st.write("Heatmap")
                        plot_chart(merged_df, "heatmap", x_var, y_var) 

                    with tab5:
                        st.write("3D Scatterplot")
                        if z_var != "None":
                            plot_chart(merged_df, "scatter_3d", x_var, y_var, z_var)    

                    with tab6:
                        st.write("3D Lineplot")
                        if z_var != "None":
                            plot_chart(merged_df, "line_3d", x_var, y_var, z_var)

                    with tab7:
                        st.write("3D Surfaceplot")
                        if z_var != "None":
                            plot_chart(merged_df, "surface_3d", x_var, y_var, z_var)

                    #with tab8:
                        #st.write("Radar chart for 'Last 60 days'")
                        #radar_columns = ['value_views', 'value_shares', 'value_likes', 'value_comments']
                        #plot_radar_chart(data, radar_columns)

                    tab1, tab2, tab3, tab4 = st.tabs(["vs Views", "vs Shares", "vs Likes", "vs Comments"])
                    with tab1:
                        st.subheader("Hashtag Performance - Views:")
                        #for tag, perf in hashtag_performance.items():
                            #st.write(f"{tag}: {perf}")
                        #AgGrid(hashtag_performance_views)
                        #df1 = pd.DataFrame(list(hashtag_performance_views.items()), columns=["hashtag", "value"])
                        # Sort the DataFrame by the 'value' column in descending order
                        sorted_df1 = df1.sort_values(by="value_views", ascending=False)
                        st.write(sorted_df1)
                        # Highlight specific bars (use 'rgba' values for transparency)
                        highlighted_bars = ['#fyp', '#tiktok', '#foryou', '#trending', '#viral']
                        sorted_df1['color'] = sorted_df1['hashtag'].apply(lambda x: 'black' if x in highlighted_bars else 'red')
                        fig = px.bar(sorted_df1, x='hashtag', y='value_views', title='Hashtag performance for the week',
                        color='color', color_discrete_map='identity', hover_data={'value_views': ':.2f'})

                        fig.update_layout(title='Hashtag performance for the week - Views', xaxis_title='Hashtag', yaxis_title='Value')
                        st.plotly_chart(fig)
                        
                    with tab2:
                        st.subheader("Hashtag Performance - Shares:")
                        #for tag, perf in hashtag_performance.items():
                            #st.write(f"{tag}: {perf}")
                        #AgGrid(hashtag_performance_views)
                        #df2 = pd.DataFrame(list(hashtag_performance_shares.items()), columns=["hashtag", "value"])
                        # Sort the DataFrame by the 'value' column in descending order
                        sorted_df2 = df2.sort_values(by="value_shares", ascending=False)
                        st.write(sorted_df2)
                        # Highlight specific bars (use 'rgba' values for transparency)
                        highlighted_bars = ['#fyp', '#tiktok', '#foryou', '#trending', '#viral']
                        sorted_df2['color'] = sorted_df2['hashtag'].apply(lambda x: 'black' if x in highlighted_bars else 'blue')
                        fig = px.bar(sorted_df2, x='hashtag', y='value_shares', title='Hashtag performance for the week',
                        color='color', color_discrete_map='identity', hover_data={'value_shares': ':.2f'})

                        fig.update_layout(title='Hashtag performance for the week - Shares', xaxis_title='Hashtag', yaxis_title='Value')
                        st.plotly_chart(fig)
                    with tab3:
                        st.subheader("Hashtag Performance - Likes:")
                        #for tag, perf in hashtag_performance.items():
                            #st.write(f"{tag}: {perf}")
                        #AgGrid(hashtag_performance_views)
                        #df3 = pd.DataFrame(list(hashtag_performance_likes.items()), columns=["hashtag", "value"])
                        # Sort the DataFrame by the 'value' column in descending order
                        sorted_df3 = df3.sort_values(by="value_likes", ascending=False)
                        st.write(sorted_df3)
                        # Highlight specific bars (use 'rgba' values for transparency)
                        highlighted_bars = ['#fyp', '#tiktok', '#foryou', '#trending', '#viral']
                        sorted_df3['color'] = sorted_df3['hashtag'].apply(lambda x: 'black' if x in highlighted_bars else 'green')
                        fig = px.bar(sorted_df3, x='hashtag', y='value_likes', title='Hashtag performance for the week',
                        color='color', color_discrete_map='identity', hover_data={'value_likes': ':.2f'})

                        fig.update_layout(title='Hashtag performance for the week - Likes', xaxis_title='Hashtag', yaxis_title='Value')
                        st.plotly_chart(fig)
                    with tab4:
                        st.subheader("Hashtag Performance - Comments:")
                        #for tag, perf in hashtag_performance.items():
                            #st.write(f"{tag}: {perf}")
                        #AgGrid(hashtag_performance_views)
                        #df4 = pd.DataFrame(list(hashtag_performance_views.items()), columns=["hashtag", "value"])
                        # Sort the DataFrame by the 'value' column in descending order
                        sorted_df4 = df4.sort_values(by="value_comments", ascending=False)
                        st.write(sorted_df4)
                        # Highlight specific bars (use 'rgba' values for transparency)
                        highlighted_bars = ['#fyp', '#tiktok', '#foryou', '#trending', '#viral']
                        sorted_df4['color'] = sorted_df4['hashtag'].apply(lambda x: 'black' if x in highlighted_bars else 'orange')
                        fig = px.bar(sorted_df4, x='hashtag', y='value_comments', title='Hashtag performance for the week',
                        color='color', color_discrete_map='identity', hover_data={'value_comments': ':.2f'})

                        fig.update_layout(title='Hashtag performance for the week - Comments', xaxis_title='Hashtag', yaxis_title='Value')
                        st.plotly_chart(fig)

                   
                    # Calculate the similarity between hashtags
                    #similarity_matrix = cosine_similarity(list(hashtag_vectors.values()))
                    #st.write(similarity_matrix)
                    # Convert the similarity matrix into a DataFrame
                    #similarity_df = pd.DataFrame(similarity_matrix, index=unique_hashtags, columns=unique_hashtags)
                    #st.write(similarity_df)
                    # Calculate the correlation between hashtag similarity and performance
                    #correlations = np.corrcoef(similarity_df, data["Total views"], rowvar=False)
                    #st.write(correlations)

        #with tab2:
            #if 'video_view_within_days' not in data.columns: #Video Posts
                #pass
        