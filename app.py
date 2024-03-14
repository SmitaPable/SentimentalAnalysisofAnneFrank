import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from wordcloud import WordCloud
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.dates import DateFormatter
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as pl
from PIL import Image


# Additional imports for the 9th graph
import numpy as np
import base64

# Additional imports for the Word Cloud
from io import StringIO

@st.cache_data()
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

#img = get_img_as_base64("Anne-Frank.jpeg")
img= get_img_as_base64("Anne-Frank.jpg")
page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{img}");
        background-size: cover;
        background-position: center;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}


    </style>
    """

st.markdown(page_bg_img, unsafe_allow_html=True)
   


# Define emotions and corresponding colors
emotions = ['Joy', 'Sadness', 'Anger', 'Fear', 'Trust', 'Disgust', 'Surprise', 'Anticipation']
colors = ['#32CD32', '#4B0082', '#8B0000', '#1A5E63', '#87CEEB', '#556B2F', '#FFA500', '#FF69B4']

# Define the common figure size
common_figsize = (10, 6)

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
st.write(f"<h1 style='color:Black;font-size: 50px; text-shadow: -4px -4px 0 white;'>Sentimental Analysis of Anne Frank Diary</h1>", unsafe_allow_html=True)

# Sidebar title
st.sidebar.title('Graph 1: Emotion Counts over Time')
# Adjust the title font size using Markdown and HTML syntax
st.markdown("<h3 style='text-align: left; color: black;'>Graph 1: Emotion Counts over Time</h1>", unsafe_allow_html=True)


# Allow the user to select specific emotion counts with a sidebar widget
selected_emotions = st.sidebar.multiselect('Choose emotions', df.columns[df.columns.str.contains('Count')])

# Define colors for each emotion
emotion_colors = {'Joy Count': '#32CD32', 'Sadness Count': '#4B0082', 'Anger Count': '#8B0000',
                  'Fear Count': '#1A5E63', 'Trust Count': '#87CEEB', 'Disgust Count': '#556B2F',
                  'Surprise Count': '#FFA500', 'Anticipation Count': '#FF69B4'}

# Plot selected emotion counts using Matplotlib and display it using st.pyplot()
if selected_emotions:
    fig, ax = plt.subplots(figsize=common_figsize)

    # Ensure consistent colors for selected emotions
    selected_colors = [emotion_colors[emotion] for emotion in selected_emotions]

    bar_width = 0.8
    bar_space = 1.0

    positions = range(len(df['Start Date']))

    for i, (emotion, color) in enumerate(zip(selected_emotions, selected_colors)):
        ax.bar(positions, df[emotion], label=emotion, color=color, width=bar_width, alpha=0.7)

    plt.xlabel('Start Date')
    plt.ylabel('Emotion Count')
    plt.title('Selected Emotion Counts over Time')
    plt.xticks(positions[::8], df['Start Date'][::8].dt.strftime('%d-%m-%Y'), rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    # Display the selected plot using Streamlit
    st.pyplot(fig)
else:
    # If no emotions are selected, display a message
    st.sidebar.info('Please select emotions from the sidebar to visualize.')

# Second Graph - Emotion Distribution Pie Chart
st.sidebar.title('Graph 2: Emotion Distribution Pie Chart')
# Adjust the title font size using Markdown and HTML syntax
st.markdown("<h3 style='text-align: left; color: black;'>Graph 2: Emotion Distribution Pie Chart</h1>", unsafe_allow_html=True)

# Calculate total counts for each emotion
total_counts = df.filter(like='Count').sum()

# Plotting the pie chart
fig_pie, ax_pie = plt.subplots(figsize=(10, 8))

# Define colors for each emotion
colors_pie = ['#32CD32', '#4B0082', '#8B0000', '#1A5E63', '#87CEEB', '#556B2F', '#FFA500', '#FF69B4']

ax_pie.pie(total_counts, labels=total_counts.index, autopct='%1.1f%%', colors=colors_pie, startangle=90)

plt.title('Emotion Distribution Pie Chart')

# Display the pie chart using Streamlit
st.pyplot(fig_pie)

# Third Graph - Sentiment Score Distribution Histogram
st.sidebar.title('Graph 3: Sentiment Score Distribution Histogram')
# Adjust the title font size using Markdown and HTML syntax
st.markdown("<h3 style='text-align: left; color: black;'>Graph 3: Sentiment Score Distribution Histogram</h1>", unsafe_allow_html=True)

# Define the number of bins for the histogram
num_bins = 20

# Calculate histogram data
hist_data, bin_edges = np.histogram(df['Sentiment Score'], bins=num_bins)

# Define custom colormap from dark red to red to green to dark green
colors = ['#8B0000', '#FF0000', '#FFD700', '#32CD32']
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)

# Normalize histogram data
norm = mcolors.Normalize(vmin=bin_edges.min(), vmax=bin_edges.max())

# Plotting the histogram for Sentiment Scores with the custom colormap
fig_hist, ax_hist = plt.subplots(figsize=common_figsize)

# Plot histogram bars with custom colors
for count, edge in zip(hist_data, bin_edges[:-1]):
    color = cmap(norm(edge))
    ax_hist.bar(edge, count, color=color, width=(bin_edges[1] - bin_edges[0]))

plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Sentiment Score Distribution Histogram')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the histogram using Streamlit
st.pyplot(fig_hist)

# Fourth Graph - Create a heatmap
st.sidebar.title('Graph 4: Correlation Heatmap')
# Adjust the title font size using Markdown and HTML syntax
st.markdown("<h3 style='text-align: left; color: black;'>Graph 4: Correlation Heatmap</h1>", unsafe_allow_html=True)

selected_columns_corr = df.filter(like='Count')

# Calculate the correlation matrix
correlation_matrix = selected_columns_corr.corr()

# Define colorscale for the heatmap
colorscale = 'RdBu'

# Plot the heatmap using Plotly
fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values,
                                x=correlation_matrix.columns.tolist(),
                                y=correlation_matrix.index.tolist(),
                                colorscale=colorscale,
                                showscale=True))

# Add title
fig.update_layout(title='Correlation Heatmap')

# Hide annotation (numbers) on the heatmap
fig.update_traces(showscale=True)

# Display the heatmap using Streamlit
st.plotly_chart(fig)

# Additional information
st.sidebar.info("This heatmap shows the correlation between different emotion counts.")

# Fifth Graph - Word Cloud for Emotions
st.sidebar.title('Graph 5: Word Clouds for Emotions')
# Adjust the title font size using Markdown and HTML syntax
st.markdown("<h3 style='text-align: left; color: black; font-size: 24px;'>Graph 5: Word Clouds for Emotions</h1>", unsafe_allow_html=True)

# Allow the user to select the emotion for the word cloud
selected_wordcloud_emotion = st.sidebar.selectbox('Select emotion for Word Cloud', df.columns[df.columns.str.contains('Words')])

# Define colors for each emotion
emotion_colors = {'Joy Words': '#32CD32', 'Sadness Words': '#4B0082', 'Anger Words': '#8B0000',
                  'Fear Words': '#1A5E63', 'Trust Words': '#87CEEB', 'Disgust Words': '#556B2F',
                  'Surprise Words': '#FFA500', 'Anticipation Words': '#FF69B4'}

# Generate word cloud for the selected emotion
if selected_wordcloud_emotion:
    # Concatenate words for the specified emotion
    words = ' '.join(df[selected_wordcloud_emotion].dropna())

    # Define a custom color function to apply the specified color
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return emotion_colors[selected_wordcloud_emotion]

    # Generate the word cloud with the custom color function
    wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=color_func).generate(words)

    # Plot the word cloud
    st.image(wordcloud.to_array(), caption=f'Word Cloud for {selected_wordcloud_emotion[:-6]}')

# Display additional information or text below the word cloud
st.sidebar.info("This word cloud shows the most frequent words associated with the selected emotion.")

# Sixth Graph - Distribution of Sentiment Labels with Extreme Scores
st.sidebar.title('Graph 6: Distribution of Sentiment Labels with Extreme Scores')
# Adjust the title font size using Markdown and HTML syntax
st.markdown("<h3 style='text-align: left; color: black; font-size: 24px;'>Graph 6: Distribution of Sentiment Labels with Extreme Scores</h1>", unsafe_allow_html=True)

# Define sentiment categories
sentiment_categories = ['Highly Positive', 'Positive', 'Negative', 'Highly Negative', 'Neutral']

# Prepare data for plotting
plot_data = {'Sentiment Label': [], 'Count': [], 'Extreme Score': []}

for label in sentiment_categories:
    subset = df[df['Sentiment Label'] == label]
    
    if not subset.empty:
        count = subset['Sentiment Label'].count()
        if label in ['Highly Positive', 'Positive']:
            extreme_score = subset['Sentiment Score'].max()
        else:
            extreme_score = subset['Sentiment Score'].min()
        
        plot_data['Sentiment Label'].append(label)
        plot_data['Count'].append(count)
        plot_data['Extreme Score'].append(extreme_score)

# Define colors for each sentiment category
colors = ['green', 'lightgreen', 'darkred', 'red', 'lightgrey']

# Plot a bar chart for Sentiment Labels with count and extreme score
fig_sentiment, ax_sentiment = plt.subplots(figsize=common_figsize)

bars_sentiment = ax_sentiment.bar(plot_data['Sentiment Label'], plot_data['Count'], color=colors, alpha=0.7)

# Annotate bars with extreme scores
for bar, score in zip(bars_sentiment, plot_data['Extreme Score']):
    height = bar.get_height()
    ax_sentiment.text(bar.get_x() + bar.get_width() / 2, height, f'Extreme: {score}', ha='center', va='bottom')

plt.xlabel('Sentiment Label')
plt.ylabel('Count')
plt.title('Distribution of Sentiment Labels')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Display the sentiment distribution plot using Streamlit
st.pyplot(fig_sentiment)


# Seventh Graph - Emotion Transition Diagram
st.sidebar.title('Graph 7: Emotion Transition Diagram')
# Adjust the title font size using Markdown and HTML syntax
st.markdown("<h3 style='text-align: left; color: black; font-size: 24px;'>Graph 7: Emotion Transition Diagram</h1>", unsafe_allow_html=True)

# Create a directed graph
G = nx.DiGraph()

# Iterate through rows and add edges to the graph
for i in range(len(df) - 1):
    current_emotion = df.at[i, 'Sentiment Label']
    next_emotion = df.at[i + 1, 'Sentiment Label']
    G.add_edge(current_emotion, next_emotion)

# Plot the Emotion Transition Diagram
pos = nx.spring_layout(G)

# Create a new figure and axis
fig_transition, ax_transition = plt.subplots(figsize=common_figsize)

# Draw the graph
nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='lightblue', font_color='black', node_size=1000, ax=ax_transition)

# Set the title
ax_transition.set_title('Emotion Transition Diagram')

# Display the Emotion Transition Diagram using Streamlit
st.pyplot(fig_transition)


# Eighth Graph - Animated Time Series: Evolution of Emotion Counts over Time
st.sidebar.title('Graph 8: Animated Time Series')
# Adjust the title font size using Markdown and HTML syntax
st.markdown("<h3 style='text-align: left; color: black; font-size: 24px;'>Graph 8: Animated Time Series</h1>", unsafe_allow_html=True)

emotions = ['Joy', 'Sadness', 'Anger', 'Fear', 'Trust', 'Disgust', 'Surprise', 'Anticipation']
colors = ['#32CD32', '#4B0082', '#8B0000', '#1A5E63', '#87CEEB', '#556B2F', '#FFA500', '#FF69B4']


# Ensure emotions and colors have the same length
if len(emotions) == len(colors):
    # Create a subplot with a line chart for the evolution of emotion counts over time
    fig_animated = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles=['Animated Time Series: Evolution of Emotion Counts over Time'])

    # Add traces for each emotion
    for i, emotion in enumerate(emotions):
        trace = go.Scatter(x=df['Start Date'], y=df[f'{emotion} Count'], name=emotion, line=dict(color=colors[i]))
        fig_animated.add_trace(trace)

    # Define events
    events = [
        {'event': 'Persecution and Restrictions', 'date': '1942-06-15'},
        {'event': 'First entry after Hiding', 'date': '1942-07-08'},
        {'event': 'First entry after D-Day (Allied Invasion of Normandy)', 'date': '1944-06-09'},
        {'event': 'Warsaw Ghetto Uprising', 'date': '1943-04-19'},
        
    ]

    # Add vertical lines for events with text and emotion counts
    for event in events:
        event_date = pd.to_datetime(event['date'], errors='coerce')

    # Find the closest date in the DataFrame
        closest_date = df['Start Date'].iloc[(df['Start Date'] - event_date).abs().idxmin()]

    # Get the sentiment label on that date
        sentiment = df.loc[df['Start Date'] == closest_date, 'Sentiment Label'].values[0]

    # Add vertical line
        line_length_factor = 1.5
        fig_animated.add_trace(go.Scatter(
           x=[closest_date, closest_date],
           y=[0, fig_animated.data[0].y.max()* line_length_factor],
           mode='lines',
           line=dict(color='darkred',dash ='dash', width=2),
           hoverinfo='text',
           text=f"{event['event']} - {event['date']} - {sentiment}",
           showlegend=False
        ))
            
    else:
            print(f"Event date {event['date']} not found in the DataFrame.")

    # Update layout
    fig_animated.update_layout(
        height=600,
        width=1000,
        title_text='Animated Time Series: Evolution of Emotion Counts over Time',
        xaxis_title='Start Date',
        yaxis_title='Emotion Count',
        showlegend=True,
        xaxis=dict(tickangle=-45, dtick='M1'),  # Rotate date labels and set tick frequency to 1 month
        legend=dict(x=1.05, y=1)  # Adjust legend position
    )

    # Add interactive legend
    fig_animated.add_trace(go.Scatter(visible=False, line=dict(color='white'), name='Hide/Show Legend'))  # Dummy trace for interactive legend

    frames_animated = [go.Frame(data=[go.Scatter(x=df['Start Date'][:frame], y=df[f'{emotion} Count'][:frame], name=emotion, line=dict(color=colors[i])) for i, emotion in enumerate(emotions)] + 
                       [go.Scatter(visible=False, line=dict(color='white'), name='Hide/Show Legend')]) for frame in range(1, len(df)+1)]
    fig_animated.frames = frames_animated

# Update animation settings
    animation_settings_animated = dict(frame=dict(duration=100, redraw=True), fromcurrent=True)
    fig_animated.update_layout(updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play',
                                            method='animate', args=[None, animation_settings_animated])])])

    # Display the animated time series
    st.plotly_chart(fig_animated)
else:
    st.error("The number of emotions and colors must be the same.")


#Graph 9: Highest Count of Each Emotion with Date
st.sidebar.title('Graph 9: Highest Count of Each Emotion with Date')
# Adjust the title font size using Markdown and HTML syntax
st.markdown("<h3 style='text-align: left; color: black; font-size: 24px;'>Graph 9: Highest Count of Each Emotion with Date</h1>", unsafe_allow_html=True)

# Get emotions columns
emotions_cols = ['Joy Count', 'Sadness Count', 'Anger Count', 'Fear Count', 'Trust Count', 'Disgust Count', 'Surprise Count', 'Anticipation Count']

# Define colors for each emotion
emotion_colors = {'Joy Count': '#32CD32', 'Sadness Count': '#4B0082', 'Anger Count': '#8B0000',
                  'Fear Count': '#1A5E63', 'Trust Count': '#87CEEB', 'Disgust Count': '#556B2F',
                  'Surprise Count': '#FFA500', 'Anticipation Count': '#FF69B4'}

# Initialize lists to store maximum counts and corresponding dates for each emotion
max_counts = []
max_dates = []

# Find maximum count and corresponding date for each emotion
for emotion in emotions_cols:
    max_count = df[emotion].max()
    max_date = df.loc[df[emotion] == max_count, 'Start Date'].iloc[0]
    max_date = pd.to_datetime(max_date).strftime('%Y-%m-%d')
    max_counts.append(max_count)
    max_dates.append(max_date)

# Create a bar chart for maximum counts of each emotion
fig = go.Figure(data=[go.Bar(x=emotions_cols, y=max_counts, text=max_dates, marker_color=[emotion_colors[emotion] for emotion in emotions_cols])])

# Update layout
fig.update_layout(title='Max Counts of Emotions with Dates',
                  xaxis_title='Emotion',
                  yaxis_title='Max Count')

# Display the chart
st.plotly_chart(fig)

#Graph 10: "Emotion Timeline Analysis"
st.sidebar.title('Graph 10: Emotion Timeline Analysis')
# Adjust the title font size and align text to the left using Markdown and HTML syntax
st.markdown("<h3 style='text-align: left; color: black; font-size: 24px;'>Graph 10: Emotion Timeline Analysis</h1>", unsafe_allow_html=True)

# Define emotions and corresponding colors
emotions = ['Joy', 'Sadness', 'Anger', 'Fear', 'Trust', 'Disgust', 'Surprise', 'Anticipation']
colors = ['#32CD32', '#4B0082', '#8B0000', '#1A5E63', '#87CEEB', '#556B2F', '#FFA500', '#FF69B4']

# Ensure emotions and colors have the same length
if len(emotions) == len(colors):
    # Create a sidebar dropdown for selecting emotion
    selected_emotion = st.sidebar.selectbox('Select Emotion:', emotions)

    # Create a subplot with a line chart for the selected emotion
    fig_animated = make_subplots(rows=1, cols=1, shared_xaxes=True,
                                 subplot_titles=[f"{selected_emotion} - Evolution of Emotion Counts over Time"])

    # Add trace for the selected emotion
    selected_index = emotions.index(selected_emotion)
    fig_animated.add_trace(go.Scatter(x=df['Start Date'], y=df[f'{selected_emotion} Count'],
                                      name=selected_emotion, line=dict(color=colors[selected_index])))

    # Update layout
    fig_animated.update_layout(height=600, width=1000, title_text=f'Evolution of {selected_emotion} Counts over Time',
                               xaxis_title='Start Date', yaxis_title=f'{selected_emotion} Count')

    # Add animation settings
    frames = [go.Frame(data=[go.Scatter(x=df['Start Date'][:frame], y=df[f'{selected_emotion} Count'][:frame],
                                        name=selected_emotion, line=dict(color=colors[selected_index]))]) for frame in range(1, len(df)+1)]
    fig_animated.frames = frames

    # Update animation settings
    animation_settings_animated = dict(frame=dict(duration=100, redraw=True), fromcurrent=True)
    fig_animated.update_layout(updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play',
                                            method='animate', args=[None, animation_settings_animated])])])

    # Display the animated time series
    st.plotly_chart(fig_animated)

else:
    st.error("The number of emotions and colors must be the same.")


# Graph 11: Interactive Dashboard
st.sidebar.title("Graph 11: Interactive Dashboard")
st.markdown("<h3 style='text-align: left; color: black; font-size: 24px;'>Graph 11: Customize Dashboard</h1>", unsafe_allow_html=True)

# Load data (optional if you want to use this data for visualization)
@st.cache
def load_data():
    return df

# Add a sidebar
st.sidebar.title("")


# Convert 'Start Date' to date format
df['Start Date'] = pd.to_datetime(df['Start Date'])

# Define emotion colors
emotion_colors = {'Joy Count': '#32CD32', 'Sadness Count': '#4B0082', 'Anger Count': '#8B0000',
                  'Fear Count': '#1A5E63', 'Trust Count': '#87CEEB', 'Disgust Count': '#556B2F',
                  'Surprise Count': '#FFA500', 'Anticipation Count': '#FF69B4'}

# Add widgets to sidebar for customization
selected_chart_type = st.sidebar.selectbox("Select chart type", ["Line Chart", "Bar Chart", "Scatter Plot"])

# Filter column names to remove those containing the word "word"
filtered_columns = [col for col in df.columns if "word" not in col.lower()]

selected_x_axis = st.sidebar.selectbox("Select X-axis", filtered_columns)
selected_y_axis = st.sidebar.selectbox("Select Y-axis", filtered_columns)

# Create visualizations based on user selection
if selected_chart_type == "Line Chart":
    st.subheader(f"Line Chart of {selected_y_axis} against {selected_x_axis}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Start Date'], y=df[selected_y_axis], mode='lines', name=selected_y_axis, line=dict(color=emotion_colors.get(selected_y_axis, None))))
    fig.update_layout(xaxis=dict(range=[df['Start Date'].min(), df['Start Date'].max()]))  # Update x-axis range
    st.plotly_chart(fig)
elif selected_chart_type == "Bar Chart":
    st.subheader(f"Bar Chart of {selected_y_axis} against {selected_x_axis}")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Start Date'], y=df[selected_y_axis], name=selected_y_axis, marker=dict(color=emotion_colors.get(selected_y_axis, None))))
    fig.update_layout(xaxis=dict(range=[df['Start Date'].min(), df['Start Date'].max()]))  # Update x-axis range
    st.plotly_chart(fig)
elif selected_chart_type == "Scatter Plot":
    st.subheader(f"Scatter Plot of {selected_y_axis} against {selected_x_axis}")
    fig = px.scatter(df, x='Start Date', y=selected_y_axis, color=selected_y_axis)
    fig.update_traces(marker=dict(color=emotion_colors.get(selected_y_axis, None)))
    fig.update_layout(xaxis=dict(range=[df['Start Date'].min(), df['Start Date'].max()]))  # Update x-axis range
    st.plotly_chart(fig)




st.sidebar.title("Graph 12: Sentiment distribution over time")
st.markdown("<h3 style='text-align: left; color: black; font-size: 24px;'>Graph 12: sentiment distribution over time</h1>", unsafe_allow_html=True)
    # Plot sentiment distribution over time
fig = px.scatter(df, x='Start Date', y='Sentiment Label', color='Sentiment Label', hover_data={'Sentiment Score': True})
fig.update_traces(marker=dict(size=12))
fig.update_layout(title='Sentiment Distribution Over Time', xaxis_title='Date', yaxis_title='Sentiment')
st.plotly_chart(fig)


