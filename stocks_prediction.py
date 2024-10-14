import streamlit as st
from ta import trend as ta
from datetime import date, datetime as dt, timedelta as td
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.subplots as ms
import pandas as pd
import numpy as np
import plotly.express as px
#import time


#git branch --set-upstream-to=origin/master master

#to run this script use: python -m streamlit run stocks_prediction.py
# Here I set time boundaries that can be specified by the user input
select_time_frame = st.sidebar.selectbox("Select the timeframe",["1day","1week","1month","1year","3years","5years","10years"],index=None)
time_frame_object = {
     "1day":1,
     "1week":7,
     "1month":30,
     "1year":365,
     "3years":365*3,
     "5years":365*5,
     "10years":365*10,
}
is_data_loaded = False
TODAY = date.today().strftime("%Y-%m-%d")
#calculating the timeframe using some basic arithmetic with timedelta (as td)
if(select_time_frame is not None):
    START = dt.strptime(TODAY, "%Y-%m-%d") - td(days=time_frame_object[select_time_frame]) 
    START = START.strftime("%Y-%m-%d")

st.title('AI Stocks Prediction')

st.sidebar.header("Parameters Dashboard:")
stocks = pd.read_csv("stocks_list.csv") #('GOOG', 'AAPL', 'MSFT', 'GME')
# here I set index to None in order to disable default initially selected value 
selected_stock = st.sidebar.selectbox('Select the stock you want to predict', stocks, index=None)
chart_types = ("Candlesticks", "Linechart", "Volume", "Rawdata")


#@st.cache_data # using this decorator we cache the data so we do not have to download it over and over again.
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    #will place dates in the first column
    data.reset_index(inplace=True)
    return data


# Add simple moving average (SMA) and exponential moving average(EMA) indicators
#This one needs troubleshooting in debug mode
def add_technical_indicators(data):
    
    #print(data["D"].head(50))
    data['SMA_20'] = ta.sma_indicator(data['Close'], window=20)
    data['EMA_20'] = ta.ema_indicator(data['Close'], window=20)
    return data


#Plot the stock price chart
def plot_candlesticks():
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['Date'],
    open=data['Open'], high=data['High'], low=data['Low'], close=data['Close']))
    print(data)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA_20'], name = 'EMA 20'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], name = 'SMA 20'))
    st.plotly_chart(fig)


# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

# Plot lineWithVolume
def plot_volume_chart():
    fig = ms.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=False)
    #Add Volume Chart to Row 2 of subplot
    # create an array and fill it with indecies of length of the data u need to fix this all shit
    index_arr = np.arange(len(data['Volume']))
    # Using DataFrame.insert() to add a column  df.insert(2, "Age", [21, 23, 24, 21], True)
    data.insert(1,"Index",index_arr, True)
    fig.add_trace(go.Bar(x=data['Index'], y=data['Volume']), row=2, col=1)
    st.plotly_chart(fig)



if(len(chart_types)> 0 and selected_stock is not None):
    selected_chart_type = st.sidebar.selectbox('Select the type of the chart you prefer', chart_types, index=None)
    with st.spinner('Wait, The data is loading......'):
        if(selected_chart_type is not None):
            n_years = st.sidebar.slider('Years of prediction:', 1, 10)
            period = n_years * 365
            data = load_data(selected_stock)
            add_technical_indicators(data)
            if(selected_chart_type=="Candlesticks" ):    
                plot_candlesticks()
            elif(selected_chart_type=="Linechart" ):
                plot_raw_data()
            elif(selected_chart_type=="Rawdata" ):
                st.subheader('Raw data')
                st.write(data.tail())
            elif(selected_chart_type=="Volume" ):
                plot_volume_chart()
            st.success("Done!")
    
    
    if(selected_chart_type is not None and st.sidebar.button('Generate Prediction')):
        with st.spinner('Wait for it...'):
            data_load_state = st.text('Generating the prediction...')
            # Predict forecast with Prophet.
            df_train = data[['Date','Close']]
            df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)

            # Show and plot forecast
            

                
            st.write(f'Forecast plot for {n_years} years')
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)

            st.subheader('Forecast data')
            st.write(forecast.tail())



            st.write("Forecast components")
            fig2 = m.plot_components(forecast)
            st.write(fig2)
        st.success("Done!")
        data_load_state.text('Done, the predicted data is successfully generated!')



        



