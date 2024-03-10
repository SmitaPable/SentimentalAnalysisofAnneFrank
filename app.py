import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

# Register converters for matplotlib
register_matplotlib_converters()

# Replace 'your_input.csv' with the actual path to your CSV file
csv_file_path = r'C:\Users\DELL\Desktop\Project\Untitled Folder\emotions_between_dates_with_sentiment.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Convert 'Start Date' to datetime format
df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')

# Filter out rows with NaN dates
df = df.dropna(subset=['Start Date'])

# Set up Streamlit app
st.title('Emotion Counts over Time')

# Select emotions for plotting
selected_emotions = st.multiselect('Select Emotions', df.columns[3:])

# Plot a bar chart with x-axis ticks for selected dates and dark colors for each emotion
fig, ax = plt.subplots(figsize=(16, 10))  # Increased figure size

# Define darker colors
colors = ['#1a75ff', '#00802b', '#b30000', '#cc6600', '#4d0099', '#66b2ff', '#ff9900', '#993366']

bar_width = 0.8  # Adjust the width of the bars
bar_space = 1.0  # Adjust the space between the bars

positions = range(len(df['Start Date']))  # Define positions before the loop

for i, (emotion, color) in enumerate(zip(df.columns[3:], colors)):
    if emotion in selected_emotions:
        ax.bar(positions, df[emotion], label=emotion, color=color, width=bar_width, alpha=0.7)

plt.xlabel('Start Date')
plt.ylabel('Emotion Count')
plt.title('Emotion Counts over Time')
plt.xticks(positions[::8], df['Start Date'][::8].dt.strftime('%d-%m-%Y'), rotation=45, ha='right')  # Set x-axis ticks every 8 positions
plt.legend()
plt.tight_layout()

# Display the plot using Streamlit
st.pyplot(fig)
