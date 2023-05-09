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

def social_icons(width=24, height=24, **kwargs):
        icon_template = '''
        <a href="{url}" target="_blank" style="margin-right: 20px;">
            <img src="{icon_src}" alt="{alt_text}" width="{width}" height="{height}">
        </a>
        '''

        icons_html = ""
        for name, url in kwargs.items():
            icon_src = {
                "youtube": "https://img.icons8.com/ios-filled/100/null/youtube-play.png",
                "linkedin": "https://img.icons8.com/ios-filled/100/null/linkedin.png",
                "github": "https://img.icons8.com/ios-filled/100/null/github--v2.png",
                "wordpress": "https://img.icons8.com/ios-filled/100/null/wordpress--v1.png",
                "email": "https://img.icons8.com/ios-filled/100/null/filled-message.png",
                "website": "https://img.icons8.com/ios/100/null/domain--v1.png"
            }.get(name.lower())

            if icon_src:
                icons_html += icon_template.format(url=url, icon_src=icon_src, alt_text=name.capitalize(), width=width, height=height)

        return icons_html
st.subheader("So how do I use this?")
st.markdown("""
Designed to provide extra value to the existing dashboard available on TikTok Analytics, individual users or businesses can download the relevant csv/xlsx files before loading them
to this custom dashboard.

|Section|Description|Files required|
|--------|-----------|--------------|
|Overview| Number cards for video views, profile views, likes, comments and shares over past 60 days, with various plots for up to 3 variables at once | `Last 60 days` |
|Content| Analysis of trending videos over past 7 days, including investigating relationships between number of hashtags used and video views| `Trending videos` |
|Followers| Broad insights of follower activity by hour, as well as breakdown of demographics by country and gender| `Follower activity` `Top territories` `Gender` `Total followers` |


Dashboard will be updated as and whenever necessary. Enjoy!
""")
st.markdown("""""")
linkedin_url = "https://www.linkedin.com/in/harrychangjr/"
github_url = "https://github.com/harrychangjr"
email_url = "mailto:harrychang.work@gmail.com"
website_url = "https://harrychangjr.streamlit.app"
st.markdown(
    social_icons(32, 32, LinkedIn=linkedin_url, GitHub=github_url, Email=email_url, Website=website_url),
    unsafe_allow_html=True)
st.markdown("")
st.markdown("*Copyright © 2023 Harry Chang*")
