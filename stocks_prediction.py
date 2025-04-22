import streamlit as st
from ta import trend as ta
from datetime import datetime
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import plotly.subplots as ms
import numpy as np
from polygon import RESTClient

# Streamlit App Title
st.title('AI Stocks Prediction')

# Sidebar Configuration
st.sidebar.header("Parameters Dashboard:")

# Timeframe Mapping to Polygon API intervals
timeframe_map = {
    "1day": "day",        # 1 day
    "1week": "week",      # 1 week
    "1month": "month",    # 1 month
    "1year": "year",      # 1 year
    "3years": "year",     # 3 years
    "5years": "year",     # 5 years
    "10years": "year"     # 10 years
}

select_time_frame = st.sidebar.selectbox(
    "Select the timeframe", list(timeframe_map.keys()), index=None
)

# Load list of stocks from CSV
stocks = pd.read_csv("stocks_list.csv")  # e.g., GOOG, AAPL, MSFT, etc.
selected_stock = st.sidebar.selectbox(
    'Select the stock you want to predict', stocks, index=None
)

chart_types = ("Candlesticks", "Linechart", "Volume", "Rawdata")

# Polygon API setup
API_KEY = "pVFTgAbpHXYori5rXNJ0apdFCcWF0u68"  # Replace with your Polygon API key
client = RESTClient(API_KEY)

# Data loading using Polygon API with get_aggs
def load_data(ticker, interval='day'):
    try:
        # Define the end date (today)
        end_date = datetime.now()
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Define the start date (1 year ago)
        start_date = end_date.replace(year=end_date.year - 1)
        start_date_str = start_date.strftime("%Y-%m-%d")

        # Fetch historical data from Polygon using get_aggs
        aggs = client.get_aggs(
            ticker, 
            1,  # Aggregation multiplier (1 means data is aggregated by the selected interval)
            interval,  # Interval type (e.g., "day", "week")
            start_date_str,  # Start date
            end_date_str  # End date
        )
        print(f"Received data: {aggs}")
        

        # Check if data is empty
        if not aggs:
            print(f"No results found for ticker {ticker} in the specified date range.")
            raise ValueError(f"No data returned for ticker '{ticker}'.")

        # Transform the data into a Pandas DataFrame
        # df = pd.DataFrame(aggs)
       

        # df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        # df = df.drop(columns=['timestamp'])
        
        df = pd.DataFrame(aggs, columns=['open', 'high', 'low', 'close', 'volume', 'vwap', 'timestamp', 'transactions', 'otc'])
        print(f"Raw DataFrame: {df.head()}")
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"Raw DataFrame2: {df.head()}")
        # Check the first few rows of the 'date' column
        print("DATE ROW: ", df['date'].head())

        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'.")

        return df
    except Exception as e:
        # Log the error to understand what failed
        st.error(f"Error downloading data for ticker '{ticker}': {e}")
        return None


# Technical indicators
def add_technical_indicators(data):
    data['SMA_20'] = ta.sma_indicator(data['close'], window=20)
    data['EMA_20'] = ta.ema_indicator(data['close'], window=20)
    return data

# Charting functions
def plot_candlesticks(data):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
        open=data['open'], high=data['high'], low=data['low'], close=data['close']))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], name='EMA 20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20'))
    st.plotly_chart(fig)

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data.index, y=data['close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plot_volume_chart(data):
    fig = ms.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=data.index, y=data['open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data.index, y=data['close'], name="stock_close"))
    index_arr = np.arange(len(data['volume']))
    data.insert(1, "Index", index_arr, True)
    fig.add_trace(go.Bar(x=data['Index'], y=data['volume']), row=2, col=1)
    st.plotly_chart(fig)

# Main app flow
if selected_stock and select_time_frame:
    selected_chart_type = st.sidebar.selectbox(
        'Select the type of the chart you prefer', chart_types, index=None)

    if selected_chart_type:
        with st.spinner('Wait, The data is loading......'):
            # New ticker validation using Polygon (not .info)
            interval_str = timeframe_map.get(select_time_frame, "day")  # Default to "day" if no match
            print(f"Selected interval: {interval_str}")  # Debugging log
            data = load_data(selected_stock, interval=interval_str)

            if data is None:
                st.stop()

            data = add_technical_indicators(data)

            if selected_chart_type == "Candlesticks":
                plot_candlesticks(data)
            elif selected_chart_type == "Linechart":
                plot_raw_data(data)
            elif selected_chart_type == "Rawdata":
                st.subheader('Raw data')
                st.write(data.tail())
            elif selected_chart_type == "Volume":
                plot_volume_chart(data)

            st.success("Data visualization completed.")

         # Prophet Forecast Section
    if selected_chart_type and st.sidebar.button('Generate Prediction'):
        with st.spinner('Generating the prediction...'):
            data_load_state = st.text('Building prediction model...')
            df_train = data[['date', 'close']].rename(columns={"date": "ds", "close": "y"})
            df_train = df_train.dropna()

            if df_train.shape[0] < 2:
                st.error("Not enough valid data to train a prediction model.")
                st.stop()

            n_years = st.sidebar.slider('Years of prediction:', 1, 10)
            period = n_years * 365

            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)

            st.write(f'Forecast plot for {n_years} years')
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)

            st.subheader('Forecast data')
            st.write(forecast.tail())

            st.write("Forecast components")
            fig2 = m.plot_components(forecast)
            st.write(fig2)

            data_load_state.text('Done, prediction successfully generated!')
            st.success("Done!")

            
