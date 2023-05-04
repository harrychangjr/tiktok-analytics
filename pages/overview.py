import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from millify import millify
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from st_aggrid import AgGrid
import io
from st_pages import Page, show_pages, add_page_title
from streamlit_extras.metric_cards import style_metric_cards

# Set page title
st.set_page_config(page_title="Overview - Tiktok Analytics Dashboard", page_icon = "📊", layout = "centered", initial_sidebar_state = "auto")


st.header("Overview")
st.markdown("""Upload your files here to load your data! 

*'Last 60 days' (xlsx or csv format)*
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

        if show_regression_line and x_var != 'Date':
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
    for uploaded_file in uploaded_files:
        # read the file
        with st.expander("View uploaded data"):
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

        # convert "Date" column to datetime object and set as index
        #data['Date'] = pd.to_datetime(data['Date'])
        #data.set_index('Date', inplace=True)

        data_list.append(data)


        # Replace "data" with your actual dataframe
        sums = data.sum()
        #st.write(sums) # To check table values for indexing
        col1, col2, col3, col4, col5 = st.columns((5))
        with col1:
            st.metric(label="Video views", value=sums[1])
        with col2:
            st.metric(label="Profile views", value=sums[2])
        with col3:
            st.metric(label="Likes", value=sums[3])
        with col4:
            st.metric(label="Comments", value=sums[4])
        with col5:
            st.metric(label="Shares", value=sums[5])
        #style_metric_cards()

        # Generate specific charts based on the file name
        if uploaded_file.name == "Last 60 days.xlsx" or uploaded_file.name == "Last 60 days.csv":
    
            x_var = st.sidebar.selectbox("Select X variable for Last 60 days", data.columns)
            y_var = st.sidebar.selectbox("Select Y variable for Last 60 days", data.columns)
            show_regression_line = False
    
            z_var_options = ["None"] + list(data.columns)
            z_var = st.sidebar.selectbox("Select Z variable for 3D charts (if applicable)", z_var_options)
            
            # Allow user to select time frequency for resampling
            #time_frequency = st.sidebar.selectbox("Select time frequency", ["Day", "Week", "Month"])

            #if time_frequency == "Week":
                #data_resampled = data.resample('W').sum()
            #elif time_frequency == "Month":
                #data_resampled = data.resample('M').sum()
            #else:
                #data_resampled = data

            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Line", "Bar", "Scatterplot", "Heatmap", 
                                                            "3D Scatterplot", "3D Lineplot", "3D Surfaceplot", "Radar chart"])
            with tab1:
                st.write("Lineplot for 'Last 60 days'")
                plot_chart(data, "line", x_var, y_var)

            with tab2:
                st.write("Barplot for 'Last 60 days'")
                plot_chart(data, "bar", x_var, y_var)

            with tab3:
                st.write("Scatterplot for 'Last 60 days'")
                show_regression_line = st.checkbox("Show regression line for Last 60 days scatterplot (does not apply when X = Date)")
                plot_chart(data, "scatter", x_var, y_var, show_regression_line=show_regression_line)

            with tab4:
                st.write("Heatmap for 'Last 60 days'")
                plot_chart(data, "heatmap", x_var, y_var) 

            with tab5:
                st.write("3D Scatterplot for 'Last 60 days'")
                if z_var != "None":
                    plot_chart(data, "scatter_3d", x_var, y_var, z_var)    

            with tab6:
                st.write("3D Lineplot for 'Last 60 days'")
                if z_var != "None":
                    plot_chart(data, "line_3d", x_var, y_var, z_var)

            with tab7:
                st.write("3D Surfaceplot for 'Last 60 days'")
                if z_var != "None":
                    plot_chart(data, "surface_3d", x_var, y_var, z_var)

            with tab8:
                st.write("Radar chart for 'Last 60 days'")
                radar_columns = ['Video views', 'Profile views', 'Likes', 'Comments', 'Shares']
                plot_radar_chart(data, radar_columns)                       
                # Add more conditions for other specific file names if needed