import streamlit as st
from st_pages import Page, show_pages, add_page_title

### Structure ###
# Streamlit file uploader to upload following files:
# Engagement - "Last 60 days.csv" 
# Content - "Trending videos.csv", "Video Posts.csv"
# Followers - "Follower activity.csv", "Top territories.csv", "Gender.csv", "Total followers.csv"

# sidebar with 3 above-mentioned sections + overview + how-to guide

# for each section, try to replicate the respective dashboards + see if can add additional filters and sliders

# Optional -- adds the title and icon to the current page
# add_page_title()

# Set page title
st.set_page_config(page_title="Tiktok Analytics Dashboard", page_icon = "📊", layout = "centered", initial_sidebar_state = "auto")

st.header("Tiktok Analytics Dashboard")

# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be
show_pages(
    [
        Page("app.py", "Home", "🏠"),
        Page("pages/overview.py", "Overview", "🌐"),
        Page("pages/content.py", "Content", "📖"),
        Page("pages/followers.py", "Followers", "👥"),

    ]
)

st.subheader("So how do I use this?")