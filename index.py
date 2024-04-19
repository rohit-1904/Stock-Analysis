import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
import subprocess

# Install scikit-learn using pip
subprocess.check_call(["pip", "install", "scikit-learn"])



# Set page title
st.title('Stock Price Prediction')

# Upload CSV data
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

# Function to perform stock price prediction
def predict_stock_price(data):
    # Read data
    df = pd.read_csv(data)
    
    # Display uploaded data
    st.subheader('Uploaded dataset:')
    st.write(df)
    
    # Check if dataset has required columns
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error('Dataset must contain columns named "Date" and "Close"')
        return
    
    # Prepare data for model training
    X = np.arange(len(df)).reshape(-1, 1)  # Use index as feature
    y = df['Close'].values
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create line chart for actual and predicted prices
    line_chart = go.Figure()
    line_chart.add_trace(go.Scatter(x=df['Date'].iloc[-len(y_pred):], y=y_test,
                                    mode='lines',
                                    name='Actual Price'))
    line_chart.add_trace(go.Scatter(x=df['Date'].iloc[-len(y_pred):], y=y_pred,
                                    mode='lines',
                                    name='Predicted Price'))
    line_chart.update_layout(title='Actual vs Predicted Prices',
                             xaxis_title='Date',
                             yaxis_title='Price')
    
    # Create scatter plot for difference between actual and predicted prices
    scatter_plot = go.Figure()
    scatter_plot.add_trace(go.Scatter(x=df['Date'].iloc[-len(y_pred):], y=y_test - y_pred,
                                      mode='markers',
                                      name='Difference'))
    scatter_plot.update_layout(title='Difference between Actual and Predicted Prices',
                               xaxis_title='Date',
                               yaxis_title='Price Difference')
    
    # Show plots
    st.subheader('Prediction Results:')
    st.plotly_chart(line_chart)
    st.plotly_chart(scatter_plot)

    # Sunburst chart
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    sunburst_fig = px.sunburst(df, path=['Year', 'Month', 'Day'], values='Close')
    st.subheader('Sunburst Chart:')
    st.plotly_chart(sunburst_fig)

    # Line chart
    line_chart_fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    line_chart_fig.update_layout(title='Stock Price Candlestick Chart')
    st.subheader('Candlestick Chart:')
    st.plotly_chart(line_chart_fig)

# Perform prediction if file is uploaded
if uploaded_file is not None:
    predict_stock_price(uploaded_file)
