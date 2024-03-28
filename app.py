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
import re 


# Additional imports for the 9th graph
import numpy as np
import base64

# Additional imports for the Word Cloud
from io import StringIO

# Set Streamlit page configuration
st.set_page_config(layout="centered")


@st.cache_data()
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Cache the image data
img = get_img_as_base64("Anne-Frank.jpg")
#img1 = get_img_as_base64("Anne-Frank1.jpg")
img2 = get_img_as_base64("Anne-Frank2.jpg")

# Add CSS to style the header image container and title
page_bg_img = f"""
    <style>
    .header-container {{
        position: relative;
        width: 100%;
        
    }}
    .header-image {{
        
        width: 150%;
        
    }}
    .header-title {{
        position: absolute;
        top: 50%;
        left: 40%;
        transform: translate(-50%, -50%);
        color: white;
        font-size: 36px;
        text-align: center;
        z-index: 1;
        width: 80%;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}

    [data-testid="stSidebar"] {{
        background-image: url("data:image/png;base64,{img2}");
        background-size: cover;
        background-position: center;
    }}
     [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{img2}");
        background-size: cover;
        background-position: center;
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

# The raw URL of the CSV file on GitHub
csv_file_url = r'https://raw.githubusercontent.com/SmitaPable/SentimentalAnalysisofAnneFrank/main/emotions_between_dates_with_sentiment.csv?token=GHSAT0AAAAAACPWME3IWSXQMKTPVOGFG3LMZQDDQRQ'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_url)

# Convert 'Start Date' to datetime format
df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')

# Filter out rows with NaN dates
df = df.dropna(subset=['Start Date'])

st.sidebar.markdown("""
    <h3 style='color:Black;font-size: 30px;text-shadow: -4px -4px 0 white;'>Sentimental Trends of Anne Franks Diary</h3>
    <hr style='border: 1px solid black;'>
""", unsafe_allow_html=True)
# Create a container for the header image
st.markdown(
    f"""
    <div class="header-container">
        <img src="data:image/png;base64,{img}" class="header-image">
        <h1 class="header-title" style="color: black; font-size: 50px; text-shadow: -4px -4px 0 white;">Sentimental Analysis of Anne Frank Diary</h1>
    </div>
    """,
    unsafe_allow_html=True
)
# Set up Streamlit app
#st.write(f"<h1 style='color:Black;font-size: 50px; text-shadow: -4px -4px 0 white;'>Sentimental Analysis of Anne Frank Diary</h1>", unsafe_allow_html=True)

# Sidebar title
st.sidebar.title('Graph 1: Emotion Counts over Time')
# Adjust the title font size using Markdown and HTML syntax
st.markdown("<h3 style='text-align: left; color: black;'>Graph 1: Emotion Counts over Time</h1>", unsafe_allow_html=True)


# Allow the user to select specific emotion counts with a sidebar widget
selected_emotions = st.multiselect('Choose emotions', df.columns[df.columns.str.contains('Count')])

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
    st.info('Please select emotions from the sidebar to visualize.')

st.header("Graph Summary:")
st.markdown("""
<span style="font-size: 20px; color: #000;">


The bar chart above shows the counts of selected emotions over time. Each bar represents the frequency of a specific emotion on different dates.

#### Insights:
- Observe trends and patterns in emotion counts over time.Anne Frank's diary's emotional content changed over time. The diary notes might have first described positive events or feelings. But as time passed, there seemed to be a change in the diary toward more negative incidents or displays of rage. 

Feel free to interact with the plot by selecting emotions from the sidebar.
</span>
""", unsafe_allow_html=True)

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

st.header("Graph Summary:")
st.markdown("""
<span style="font-size: 20px; color: #000;">
            


The pie chart above illustrates the distribution of emotions based on their total counts. Each slice represents the proportion of a specific emotion within the dataset.

#### Insights:
The emotional journey of Anne Frank shows the complex range of feelings she experienced while hiding. The frequency of fear (13.3%) and sadness (12.8%) reveals the significant obstacles and uncertainties she faced, yet moments of joy (14.1%) and anticipation (15.1%) reflect optimism and hopefulness. Furthermore, despite the danger and isolation, the high level of trust (17.7%) indicates a certain dependence on others for solace and support. The 
fact that she expressed anger (10.7%), disgust (8.2%), and surprise (8%) highlights the complexity of her experiences, which included disgust, astonishment, and irritation in the 
midst of the turbulent conditions of hiding during the Holocaust. 

</span>
""", unsafe_allow_html=True)


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

# Display the "Graph Summary" header
st.header("Graph Summary:")

#  Display the text with HTML formatting
st.markdown("""
<span style="font-size: 20px; color: #000;">
This histogram visualizes the distribution of sentiment scores. Each bar represents the frequency of sentiment scores within a certain range.

#### Insights:
The bars are colored based on a custom colormap ranging from dark red to dark green, indicating the sentiment intensity.
According to this histogram study, which has a peak in the negative region, Anne Frank's diary has a significant quantity of negative sentiment. Furthermore, neutral feeling is widely prevalent, whereas extreme positive sentiment is somewhat uncommon. Understanding the general sentiment trends depicted in Anne Frank's story can benefit greatly from these observations.
Feel free to explore the sentiment score distribution and analyze any patterns or trends.
</span>
""", unsafe_allow_html=True)


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

st.header("Graph Summary:")
# Display the graph summary using st.markdown()
st.markdown("""
<span style="font-size: 20px; color: #000;">
The correlation heatmap above illustrates the relationships between different emotion counts. Each cell represents the correlation coefficient between two emotion counts, indicating the strength and direction of their relationship.

#### Insights:
- Colours: The heatmap's colours span from red to blue, and the scale is shown by a colour bar on the right. Higher positive correlations are shown by red, higher negative correlations by blue, and values ranging from positive to negative are represented by the hues in between.
- Interpretation: For instance, the cell with a correlation value of -0.611338 is shaded red and located at the intersection of the X- and Y-axes labelled "Disgust Count" and "Surprise Count." This shows that the counts of the emotions "Disgust" and "Surprise" have a somewhat negative association.
</span>
""", unsafe_allow_html=True)

# Fifth Graph - Word Cloud for Emotions
st.sidebar.title('Graph 5: Word Clouds for Emotions')
# Adjust the title font size using Markdown and HTML syntax
st.markdown("<h3 style='text-align: left; color: black; font-size: 24px;'>Graph 5: Word Clouds for Emotions</h1>", unsafe_allow_html=True)

# Allow the user to select the emotion for the word cloud
selected_wordcloud_emotion = st.selectbox('Select emotion for Word Cloud', df.columns[df.columns.str.contains('Words')])

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

st.header("Graph Summary:")
# Display the graph summary using st.markdown()
st.markdown("""
<span style="font-size: 20px; color: #000;">
The word cloud visualization above displays the most frequent words associated with a selected emotion. Each word's size represents its frequency, and its color corresponds to the selected emotion.

#### Insights:
- Interpretation: For Instance, Word Clouds for Emotions and has the theme "Anger Words," is a portion of a bigger set or presentation. 
The word cloud highlights words related to rage in red, highlighting the emotional context. The larger the word, the more likely it is to be connected to the feeling of anger.

</span>
""", unsafe_allow_html=True)

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

st.header("Graph Summary:")
# Display the graph summary using st.markdown()
st.markdown("""
<span style="font-size: 20px; color: #000;">
The bar chart above represents the distribution of sentiment labels with extreme scores. Each bar corresponds to a sentiment label category, showing the count of occurrences and the extreme score associated with each category.

#### Insights:
- On the based on the graph, we may deduce that most sentiments were categorized as "Highly Positive" or "Highly Negative" as a result of the sentiment analysis, with "Highly Positive" sentiments being somewhat more common. Few "Positive" 
and "Negative" sentiments were found, and no instances of "Neutral" sentiment were noted. The extreme scores most likely reflect the sentiment's intensity or strength.


</span>
""", unsafe_allow_html=True)

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

st.header("Graph Summary:")
# Display the graph summary using st.markdown()
st.markdown("""
<span style="font-size: 20px; color: #000;">
The Emotion Transition Diagram above illustrates the transitions between different emotions over time. Each node represents an emotion, and directed edges indicate transitions from one emotion to another.

#### Insights:
Emotion Transition Diagram shows how each emotional state changes into another. However, It is possible for "Neutral" to change into any of the four emotional states or to stay in "Neutral."

</span>
""", unsafe_allow_html=True)

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

st.header("Graph Summary:")
# Display the graph summary using st.markdown()
st.markdown("""
<span style="font-size: 20px; color: #000;">
The animated time series illustrates the evolution of emotion counts over time, with each line representing a different emotion. Anne's emotional reactions to different events are depicted in an animated time series graph that shows how her emotions have changed over this period.

#### Correlation with Anne Frank's Sentiments:
- **First Entry after Hiding (Negative):** Anne's apprehension and uncertainty upon emerging from hiding likely influenced her negative sentiment, reflecting challenges of reintegrating into a dangerous world.
- **First Entry after D-Day (Neutral):** Anne's cautious optimism or reserved reaction to the Allied invasion may have resulted in a neutral sentiment, balancing hope for liberation with awareness of uncertainties.
- **Warsaw Ghetto Uprising (Highly Negative):** Anne's deep empathy for the suffering in the Warsaw Ghetto likely led to a highly negative sentiment, mirroring the tragic events and profound loss experienced by the Jewish community.

#### Event Markers:
Vertical lines mark significant historical events, providing context for interpreting emotion trends.
</span>
""", unsafe_allow_html=True)

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

st.header("Graph Summary:")
st.markdown("""
<span style="font-size: 20px; color: #000;">
The bar chart above displays the highest count of each emotion along with the corresponding date. Each bar represents a different emotion, and the height of the bar indicates the maximum count recorded for that emotion. The text on each bar shows the date when the maximum count occurred.

#### Insights:
- The dates on the graph match significant events that Anne Frank experienced while she was hiding, offering insight into her emotional journey

#### Emotion Analysis:
In the provided excerpt, Anne Frank undergoes a rollercoaster of emotions, some of them are explained below.

- **Happiness:** Anne experiences a moment of joy on March 6, 1944, amidst the challenges of hiding. This happiness likely stems from companionship and laughter shared with Peter, as well as the enjoyment of simple pleasures like listening to music.

- **Sadness:** On July 8, 1944, Anne grapples with profound sadness as she reflects on the dangers and uncertainties of their situation. The fear of discovery and the potential loss of her cherished diary weigh heavily on her, contributing to her sense of melancholy.

- **Anger:** Anne's privacy violation sparks anger on April 11, 1944, particularly at the suggestion of destroying her diary to conceal their hiding place. This intrusion on her personal space and the threat to her cherished writings elicit a strong sense of injustice and indignation.

- **Fear:** On April 11, 1944, fear grips Anne during a break-in scare, as the threat of discovery looms large. The uncertainty surrounding the intruders' intentions intensifies her apprehension, leaving her feeling vulnerable and on edge as she awaits the arrival of the police.


""", unsafe_allow_html=True)

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
    selected_emotion = st.selectbox('Select Emotion:', emotions)

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


st.header("Graph Summary:")
st.markdown("""
<span style="font-size: 20px; color: #000;">
This interactive visualization allows users to explore the evolution of different emotions over time. Users can select an emotion from the dropdown menu on the sidebar to view its corresponding timeline.

#### Insights:
- This examination illuminates the psychological distress that Anne Frank experienced throughout her incarceration, providing an insight into the spectrum of emotions she faced on a daily basis

This interactive visualization aids in understanding the temporal dynamics of emotions, offering insights into how emotional states evolve over time. Users can explore trends, identify patterns, and gain deeper insights into the emotional landscape across various time periods.
</span>
""", unsafe_allow_html=True)

# Graph 11: Interactive Dashboard
st.sidebar.title("Graph 11: Interactive Dashboard")
st.markdown("<h3 style='text-align: left; color: black; font-size: 24px;'>Graph 11: Customize Dashboard</h1>", unsafe_allow_html=True)

# Load data 
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
selected_chart_type = st.selectbox("Select chart type", ["Line Chart", "Bar Chart", "Scatter Plot"])

# Filter column names to remove those containing the word "word"
filtered_columns = [col for col in df.columns if "word" not in col.lower()]

selected_x_axis = st.selectbox("Select X-axis", filtered_columns)
selected_y_axis = st.selectbox("Select Y-axis", filtered_columns)

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


st.header("Graph Summary:")
st.markdown("""
<span style="font-size: 20px; color: #000;">
This interactive dashboard empowers users to customize their visualizations based on selected chart types and axes variables.
            
#### Insights:
- **Chart Type Selection:** Users can choose from three chart types including Line Chart, Bar Chart, and Scatter Plot using the dropdown menu on the sidebar.
- **Customization Options:** The sidebar also provides options to select the X-axis and Y-axis variables, allowing users to tailor their visualizations according to their preferences.
- **Dynamic Visualization:** Based on the user's selections, the dashboard generates interactive visualizations that dynamically update to reflect changes in the chosen chart type and axes variables.
  
This interactive dashboard enhances user experience by offering flexibility and control over the visualization process. Users can explore different chart types, compare variables, and gain deeper insights into the data through interactive and customizable visualizations.
</span>
            """, unsafe_allow_html=True)


st.sidebar.title("Graph 12: Sentiment distribution over time")
st.markdown("<h3 style='text-align: left; color: black; font-size: 24px;'>Graph 12: sentiment distribution over time</h1>", unsafe_allow_html=True)
    # Define colors for each sentiment category
colors = {'Highly positive': 'green', 'Positive': 'darkgreen', 'Highly Negative': 'darkred', 'Negative': 'red', 'Neutral': 'lightgrey'}

# Plot sentiment distribution over time
fig = px.scatter(df, x='Start Date', y='Sentiment Label', color='Sentiment Label', color_discrete_map=colors, hover_data={'Sentiment Score': True})
fig.update_traces(marker=dict(size=12))
fig.update_layout(title='Sentiment Distribution Over Time', xaxis_title='Date', yaxis_title='Sentiment')
st.plotly_chart(fig)


st.header("Graph Summary:")
st.markdown("""
<span style="font-size: 20px; color: #000;">
This visualization provides insights into the distribution of sentiment over time. Users can explore how sentiment labels vary across different dates.

#### Insights:
- The frequency of Highly Negative thoughts increases noticeably toward the end of the timeframe, indicating a period of increased emotional suffering or despair. This increase in intensely negative feelings may be related to important occasions or difficulties that Anne and her family encountered, which would deepen our comprehension of the psychological effects of being confined during a war.
</span>
""", unsafe_allow_html=True)


st.sidebar.title("Graph 13: Sentiment Labels Distribution")
st.markdown("<h3 style='text-align: left; color: black; font-size: 24px;'>Graph 13: Sentiment Labels Distribution</h1>", unsafe_allow_html=True)
 # Count the occurrences of each sentiment label
sentiment_counts = df['Sentiment Label'].value_counts()

    # Create pie chart
labels = sentiment_counts.index.tolist()
sizes = sentiment_counts.values.tolist()

 # Define custom colors
colors = {'highly positive': 'green', 
              'positive': 'lightgreen', 
              'highly negative': 'darkred', 
              'negative': 'red', 
              'neutral': 'lightgrey'}

    # Create pie chart with custom colors
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=[colors[label.lower()] for label in labels])
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)
plt.title('Sentiment Labels Distribution Pie Chart')
st.header("Graph Summary:")

st.markdown("""
<span style="font-size: 20px; color: #000;">
The pie chart shows how the sentiment labels are distributed in the dataset. The sentiment labels consist of categories ranging from highly positive to highly negative.

#### Insights: 
In general, the chart presents a sentiment analysis that shows a mostly positive sentiment (54.8% when combining Positive and Highly Positive), with negative sentiments (including Negative and Highly Negative) making up 42.8% of the data. The Neutral category remains relatively small compared to the others, highlighting the emotional intensity that permeates Anne Frank's diary entries.
</span>
""", unsafe_allow_html=True)

#Table1: Density of Emotion Words in Diary of Anne Frank
st.sidebar.title("Table1: Density of Emotion Words in Diary of Anne Frank")
st.markdown("<h3 style='text-align: left; color: black; font-size: 24px;'>Table1: Density of Emotion Words in Diary of Anne Frank</h3>", unsafe_allow_html=True)

# Define the emotion word count columns
emotion_word_count_columns = ['Joy Count', 'Sadness Count', 'Anger Count', 'Fear Count', 
                              'Trust Count', 'Disgust Count', 'Surprise Count', 'Anticipation Count']

# Calculate the mean and standard deviation for each emotion
mean_std_per_emotion = {}
for emotion_column in emotion_word_count_columns:
    emotion = emotion_column.split()[0]  # Extract emotion name from column name
    mean_std_per_emotion[emotion] = {
        'Mean': df[emotion_column].mean(),
        'Standard Deviation σ': df[emotion_column].std()
    }


df_stats = pd.DataFrame(mean_std_per_emotion).T.round(2)
# Create a DataFrame for better formatting
df_stats = df_stats.T
# Print the formatted output
st.write(df_stats)
st.write("Table 1: Density of emotion words in Diary of Anne Frank")

st.markdown("""
<div style="font-size: 16px; color: #000;">
    <b>Sentiment Labels:</b> Explore the density of different emotions throughout Anne Frank's diary. Dive into the prevalence of joy, sadness, anger, fear, trust, disgust, surprise, and anticipation words, measured as the number of emotion words per 10,000 words of text.
</div>
""", unsafe_allow_html=True)

#Table2: Density of Polarity Words in Diary of Anne Frank
st.sidebar.title("Table2: Mean and Standard Deviation of Polarity Words Density in Diary of Anne Frank")
st.markdown("<h3 style='text-align: left; color: black; font-size: 24px;'>Table2: Mean and Standard Deviation of Polarity Words Density in Diary of Anne Frank</h3>", unsafe_allow_html=True)

# Define the polarity word count columns
positive_columns = ['Joy Count', 'Trust Count','Anticipation Count']
neutral_columns = ['Surprise Count']
negative_columns = ['Sadness Count', 'Anger Count', 'Fear Count', 'Surprise Count']

# Concatenate all polarity word count columns
polarity_columns = positive_columns + neutral_columns + negative_columns

# Calculate the mean and standard deviation for each polarity category
mean_std_polarity = {}
for polarity_category, columns in {'Positive': positive_columns, 'Neutral': neutral_columns, 'Negative': negative_columns}.items():
    mean_std_polarity[polarity_category] = {
        'Mean': df[columns].sum(axis=1).mean(),
        'Standard Deviation σ': df[columns].sum(axis=1).std()
    }

# Create a DataFrame for better formatting
df_stats = pd.DataFrame(mean_std_polarity).T.round(2)

# Print the results
st.write(df_stats)
st.write("Table 2: Mean and Standard Deviation of polarity words density in Diary of Anne Frank.")
st.markdown("""
<div style="font-size: 16px; color: #000;">
    <b>Polarity Density Analysis:</b> Discover the density of polarity words in Anne Frank's diary, categorized as positive, neutral, and negative sentiments. Understand the distribution of words associated with joy, trust, surprise, anticipation, sadness, anger, and fear, providing insights into the emotional landscape of Anne Frank's narrative.
</div>
""", unsafe_allow_html=True)


#Overview: Emotion and Sentiment Analysis Dashboard
st.sidebar.title("Overview: Emotion and Sentiment Analysis Dashboard:")
st.header("Overview: Emotion and Sentiment Analysis Dashboard:")
st.markdown("""
<div style="font-size: 20px; color: #000;">
The Sentiment Analysis Dashboard provides a comprehensive exploration of Anne Frank's emotional journey as depicted in her diary. Through interactive graphs and charts, users can gain insights into the evolving emotional landscape of the diary, from moments of joy and sadness to instances of anger and fear. Additionally, users can analyze the density of emotional words and polarity words, measuring the prevalence of specific emotions and sentiments throughout the text. The dashboard offers a chronological perspective on Anne Frank's experiences, allowing users to visualize emotion counts over time, explore sentiment distributions, and uncover correlations between different emotions. Word clouds and sentiment histograms provide qualitative and quantitative analyses of the diary's emotional content, enriching the understanding of Anne Frank's narrative. With features such as animated time series and emotion timeline analysis, users can delve deeper into Anne Frank's diary, gaining profound insights into her innermost thoughts and feelings.
            </div>
<div style="font-size: 10px; color: #000;">
Img Source by: https://www.stmartin.ie/anne-frank-a-legacy/
</div>
""", unsafe_allow_html=True)


