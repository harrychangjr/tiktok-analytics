# Import base streamlit dependency
import streamlit as st
# Import pandas to load the analytics data
import pandas as pd
# Import subprocess to run tiktok script from command line
from subprocess import call
# Import plotly for viz
import plotly.express as px

### Structure ###
# Streamlit file uploader to upload following files:
# Engagement - "Last 60 days.csv" 
# Content - "Trending videos.csv", "Video Posts.csv"
# Followers - "Follower activity.csv", "Top territories.csv", "Gender.csv", "Total followers.csv"

# sidebar with 3 above-mentioned sections + overview + how-to guide

# for each section, try to replicate the respective dashboards + see if can add additional filters and sliders