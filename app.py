import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf
import ta
from ta.trend import adx
import base64
import datetime
from datetime import date

# Get today's date
today = datetime.date.today()

# Get user input
start = st.sidebar.date_input("Start date", datetime.date(2012, 1, 1), max_value=today)
end = st.sidebar.date_input("End date", datetime.date(2022, 12, 31), max_value=today)

# Validate input
if start > today or end > today:
    st.error("Error: Date cannot be in the future. Please enter a valid date.")
elif start >= end:
    st.error("Error: Start date must be before end date. Please enter valid dates.")
else:
    st.success('Start date: `%s`\n\nEnd date:`%s`' % (start, end))

st.title('Stock Price Prediction, Comparision, Technical Indicators & News with Sentiment Analysis')
user_input = st.text_input('Enter stock ticker','IBM')
df = yf.download(user_input, start=start, end=end)

st.subheader(f'Data description from {start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")} of {user_input} Stock..!')
st.write(df.describe())

import plotly.graph_objs as go

# Create trace for closing price vs time
trace1 = go.Scatter(x=df.index, y=df.Close, mode='lines', name='Close')

# Create layout for the chart
layout = go.Layout(title='Closing Price vs Time',
                   xaxis=dict(title='Time'),
                   yaxis=dict(title='Closing Price'))

# Create Plotly figure and display it
fig = go.Figure(data=[trace1], layout=layout)
st.plotly_chart(fig)

# Create trace for candlestick chart
trace1 = go.Candlestick(x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Price')

# Create layout for the chart
layout = go.Layout(title='Candlestick Chart',
                   xaxis=dict(title='Time'),
                   yaxis=dict(title='Price'))

# Create Plotly figure and display it
fig = go.Figure(data=[trace1], layout=layout)
st.plotly_chart(fig)


ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

# Create trace for 100MA and closing price vs time
trace1 = go.Scatter(x=df.index, y=ma100, mode='lines', name='100MA')
trace2 = go.Scatter(x=df.index, y=df.Close, mode='lines', name='Close')

# Create layout for the chart
layout = go.Layout(title='Closing Price vs Time with 100MA',
                   xaxis=dict(title='Time'),
                   yaxis=dict(title='Closing Price'))

# Create Plotly figure and display it
fig = go.Figure(data=[trace1, trace2], layout=layout)
st.plotly_chart(fig)


# Create trace for 100MA, 200MA, and closing price vs time
trace1 = go.Scatter(x=df.index, y=ma100, mode='lines', name='100MA', line=dict(color='red', dash='dash'))
trace2 = go.Scatter(x=df.index, y=ma200, mode='lines', name='200MA', line=dict(color='green', dash='dash'))
trace3 = go.Scatter(x=df.index, y=df.Close, mode='lines', name='Close', line=dict(color='blue'))

# Create layout for the chart
layout = go.Layout(title='Closing Price vs Time with Moving Averages',
                   xaxis=dict(title='Time'),
                   yaxis=dict(title='Closing Price'))

# Create Plotly figure and display it
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
st.plotly_chart(fig)


data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

model = load_model('keras_model1.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test),np.array(y_test)  
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor 

# Create trace for original and predicted prices
trace1 = go.Scatter(x=df.index[-len(y_test):], y=y_test, mode='lines', name='Original Price')
trace2 = go.Scatter(x=df.index[-len(y_test):len(df)-100], y=y_predicted.flatten(), mode='lines', name='Predicted Price')

# Create layout for the chart
layout = go.Layout(title='Predicted vs Original',
                   xaxis=dict(title='Time'),
                   yaxis=dict(title='Price'))

# Create Plotly figure and display it
fig = go.Figure(data=[trace1, trace2], layout=layout)
st.plotly_chart(fig)

# Determine whether to buy or not to buy the stock based on predicted values

predicted_trend = np.where(y_predicted.flatten() > y_test, "Buy", "Don't Buy")
predicted_trend_desc = "You can consider buying this stock." if "Buy" in predicted_trend else "It might not be the best time to buy this stock."

# Determine the text color based on the prediction
text_color = "#008000" if "Buy" in predicted_trend else "#FF0000"

# Add styling to the last line
predicted_trend_desc_styled = "<p style='font-size: 24px; color: {}; font-weight: bold; text-align: center; padding: 10px; border: 2px solid {}; border-radius: 10px;'>{}</p>".format(text_color, text_color, predicted_trend_desc)

# Display the styled line
st.markdown(predicted_trend_desc_styled, unsafe_allow_html=True)

import streamlit as st
from textblob import TextBlob
from newsapi import NewsApiClient
from datetime import datetime, timedelta

# Function to get emoji based on sentiment score
def get_sentiment_emoji(sentiment):
    if sentiment > 0:
        return "😃 (Positive Sentiment)"  # Positive emoji
    elif sentiment < 0:
        return "😔 (Negative Sentiment)"  # Negative emoji
    else:
        return "😐 (Neutral Sentiment)"  # Neutral emoji

if user_input:
    # get news articles related to the stock symbol using News API
    st.subheader(f"News articles related to {user_input}")
    # initialize NewsApiClient with your API key
    newsapi = NewsApiClient(api_key='517da00e19094775ae25b3cbf6dfaa80')

    # set the date range
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    # get the top headlines for the stock symbol from the past 30 days
    top_headlines = newsapi.get_everything(q=user_input,
                                           language='en',
                                           from_param=from_date,
                                           to=to_date,
                                           sort_by='relevancy',
                                           page_size=100)

import math

if top_headlines['totalResults'] == 0:
    st.write('No news articles found')
else:
    total_articles = top_headlines["totalResults"]
    st.write(f'Total {total_articles} articles found')
    articles_per_page = 5
    num_of_pages = math.ceil(total_articles/articles_per_page)
    page_number = st.number_input('Select Page Number', min_value=1, max_value=num_of_pages, value=1, step=1)
    start_index = (page_number - 1) * articles_per_page
    end_index = start_index + articles_per_page
    articles = top_headlines['articles'][start_index:end_index]
    for i, article in enumerate(articles):
        st.write('---')
        st.write(f"**Title:** [{article['title']}]({article['url']})")
        st.write(f"**Description:** {article['description']}")
        st.write(f"**Source:** {article['source']['name']}")

        # perform sentiment analysis on the news article
        analysis = TextBlob(article['description'])
        sentiment = analysis.sentiment.polarity
        sentiment_emoji = get_sentiment_emoji(sentiment)
        st.write(f"**Sentiment:** {sentiment:.2f} {sentiment_emoji}")

    st.write(f"Showing articles {start_index+1} - {end_index} out of {total_articles}")



# Define function to retrieve stock data
def get_stock_data(ticker):
    data = yf.download(ticker)
    return data

st.subheader('Stock Comparision')

# Define function to add arrow marks for rapid stock changes
def add_arrow_marks(fig, data, percent_change, threshold):
    # Add up arrows for rapid stock increases
    up_indices = percent_change[percent_change > threshold].index
    up_prices = data[percent_change > threshold]
    fig.add_trace(go.Scatter(x=up_indices, y=up_prices, mode='markers', name='Up', marker=dict(symbol='triangle-up', color='green', size=10)))

    # Add down arrows for rapid stock decreases
    down_indices = percent_change[percent_change < -threshold].index
    down_prices = data[percent_change < -threshold]
    fig.add_trace(go.Scatter(x=down_indices, y=down_prices, mode='markers', name='Down', marker=dict(symbol='triangle-down', color='red', size=10)))

# Define function to compare multiple stocks
def compare_stocks(tickers, threshold):
    data = pd.DataFrame()
    for ticker in tickers:
        stock_data = get_stock_data(ticker)
        data[ticker] = stock_data['Close']

    # Calculate percentage change in closing prices for the entire data
    percent_change = data.pct_change() * 100

    fig = go.Figure()

    # Add line traces for each stock
    for ticker in tickers:
        fig.add_trace(go.Scatter(x=data.index, y=data[ticker], mode='lines', name=ticker))

        # Add arrow marks for rapid stock changes for each stock
        add_arrow_marks(fig, data[ticker], percent_change[ticker], threshold)

    # Customize plot layout
    fig.update_layout(title='Stock Comparison', xaxis_title='Date', yaxis_title='Price')
    fig.update_xaxes(tickangle=45)
    fig.update_layout(legend=dict(orientation='h', yanchor='top', y=-0.2))

    # Display plot
    st.plotly_chart(fig)

# Define list of stocks to choose from
tickers = yf.Tickers("AAPL MSFT GOOGL AMZN TSLA JPM JNJ NVDA V NFLX AMD PYPL BAC GC XOM BP CVX T VZ TMUS DIS CRM ORCL IBM GE BA LUV DAL AAL UBER LYFT SBUX MCD KO PEP WMT TGT AMT PLD PSA SPG INTC CSCO QCOM TXN BABA PDD BIDU NTES CRM SAP ASML TM HMC NSANY TSM MU LRCX KLAC AMAT XLNX MCHP CDNS MRVL SPOT PINS TWTR SNAP SHOP WIX ZM DOCU CRM ZS OKTA NET ESTC SPLK NOW TEAM WORK")

# Add multi-select input widget to choose stocks
selected_stocks = st.multiselect('Select stocks to compare', tickers.tickers)

# Add slider widget to adjust the threshold for rapid stock changes
threshold = st.slider('Threshold for Rapid Stock Changes (%)', min_value=1, max_value=20, value=10)

# If multiple stocks are selected, compare them
if len(selected_stocks) > 1:
    compare_stocks(selected_stocks, threshold)
# If no or only one stock is selected, display a message
else:
    st.write('Please select at least two stocks to compare.')

st.subheader('Technical Indicators')

# Define function to retrieve stock data
def get_stock_data(ticker):
    data = yf.download(ticker)
    return data

# Define function to add technical indicators
def add_technical_indicators(data):
    # Add 20-day simple moving average (SMA) to the data
    data['SMA20'] = ta.trend.sma_indicator(data['Close'], window=20)
    
    # Add 50-day simple moving average (SMA) to the data
    data['SMA50'] = ta.trend.sma_indicator(data['Close'], window=50)

    # Add Standard Deviation (SD) of closing prices to the data
    data['SD'] = ta.volatility.bollinger_mavg(data['Close'], window=20, fillna=True) - ta.volatility.bollinger_lband(data['Close'], window=20, window_dev=2, fillna=True)
    
    # Add Relative Strength Index (RSI) to the data
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    
    # Add Moving Average Convergence Divergence (MACD) to the data
    data['MACD'] = ta.trend.macd(data['Close'])
    
    # Add Average Directional Index (ADX) to the data
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=14)

    # Add Stochastic Oscillator to the data
    data['%K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
    data['%D'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)

    # Add Bollinger Bands to the data
    indicator_bb = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['bb_bbm'] = indicator_bb.bollinger_mavg()
    data['bb_bbh'] = indicator_bb.bollinger_hband()
    data['bb_bbl'] = indicator_bb.bollinger_lband()

    # Add Money Flow Index (MFI) to the data
    data['MFI'] = ta.volume.money_flow_index(data['High'], data['Low'], data['Close'], data['Volume'], window=14)
    
    return data

# Define function to display stock chart with technical indicators
def display_stock_chart(ticker, data):
    st.subheader(f"Close values for {ticker}")
    fig = go.Figure(data=go.Scatter(x=data.index, y=data['Close']))
    st.plotly_chart(fig)

    st.subheader(f"Moving average of 20 and 50 for {ticker}")
    fig = go.Figure(data=[
        go.Scatter(x=data.index, y=data['SMA20'], name='SMA20'),
        go.Scatter(x=data.index, y=data['SMA50'], name='SMA50')
    ])
    st.plotly_chart(fig)

    st.subheader(f"Standard deviation for {ticker}")
    fig = go.Figure(data=go.Scatter(x=data.index, y=data['SD']))
    st.plotly_chart(fig)

    st.subheader(f"Relative Strength Index (RSI) for {ticker}")
    fig = go.Figure(data=go.Scatter(x=data.index, y=data['RSI']))
    st.plotly_chart(fig)

    st.subheader(f"Moving average convergence divergence (MACD) for {ticker}")
    fig = go.Figure(data=go.Scatter(x=data.index, y=data['MACD']))
    st.plotly_chart(fig)

    st.subheader(f"Average Directional Index (ADX) for {ticker}")
    fig = go.Figure(data=go.Scatter(x=data.index, y=data['ADX']))
    st.plotly_chart(fig)

    st.subheader(f"Stochastic Oscillator for {ticker}")
    fig = go.Figure(data=[
        go.Scatter(x=data.index, y=data['%K'], name='%K'),
        go.Scatter(x=data.index, y=data['%D'], name='%D')
    ])
    st.plotly_chart(fig)

    st.subheader(f"Bollinger Bands for {ticker}")
    fig = go.Figure(data=[
        go.Scatter(x=data.index, y=data['Close'], name='Close'),
        go.Scatter(x=data.index, y=data['bb_bbm'], name='bb_bbm'),
        go.Scatter(x=data.index, y=data['bb_bbh'], name='bb_bbh'),
        go.Scatter(x=data.index, y=data['bb_bbl'], name='bb_bbl')
    ])
    st.plotly_chart(fig)

    st.subheader(f"Money Flow Index (MFI) for {ticker}")
    fig = go.Figure(data=go.Scatter(x=data.index, y=data['MFI']))
    st.plotly_chart(fig)

# Define function to export data as CSV file
def export_to_csv(data, ticker):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{ticker}.csv"><button style="background-color: #FF4B4B; color: white; padding: 0.5em 1em; border: none; border-radius: 4px; cursor: pointer;">Download {ticker} file</button></a>'
    return href

# Define list of stocks to choose from
stock_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "JNJ", "NVDA", "V", "NFLX", "AMD", "PYPL", "BAC", "GC", "XOM", "BP", "CVX", "T", "VZ", "TMUS", "DIS", "CRM", "ORCL", "IBM", "GE", "BA", "LUV", "DAL", "AAL", "UBER", "LYFT", "SBUX", "MCD", "KO", "PEP", "WMT", "TGT", "AMT", "PLD", "PSA", "SPG", "INTC", "CSCO", "QCOM", "TXN", "BABA", "PDD", "BIDU", "NTES", "CRM", "SAP", "ASML", "TM", "HMC", "NSANY", "TSM", "MU", "LRCX", "KLAC", "AMAT", "XLNX", "MCHP", "CDNS", "MRVL", "SPOT", "PINS", "TWTR", "SNAP", "SHOP", "WIX", "ZM", "DOCU", "CRM", "ZS", "OKTA", "NET", "ESTC", "SPLK", "NOW", "TEAM", "WORK"]

# Add multi-select input widget to choose stocks
selected_stocks = st.multiselect('Select stocks to analyze', stock_list)

# If stocks are selected, analyze them
if selected_stocks:
    for ticker in selected_stocks:
        # Retrieve stock data
        data = get_stock_data(ticker)
        # Add technical indicators to the data
        data = add_technical_indicators(data)
        # Display stock chart with technical indicators
        st.write(f"## {ticker} Stock Price with Technical Indicators")

        # Export data as CSV file and display download button
        csv_link = export_to_csv(data, ticker)
        st.markdown(csv_link, unsafe_allow_html=True)

        display_stock_chart(ticker, data)
else:
    st.write('Please select at least one stock to analyze.')