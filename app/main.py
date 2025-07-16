import streamlit as st
import plotly.graph_objects as go
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.stock_data import get_stock_data
from utils.news_sentiment import fetch_news_with_sentiment
from models.lstm_model import forecast_next_7_days





st.set_page_config(page_title="📈 Real-Time Stock Dashboard", layout="wide")

st.title("📈 Real-Time Stock Price Viewer")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY)", value="AAPL")

if ticker:
    with st.spinner(f"Fetching data for {ticker}..."):
        data = get_stock_data(ticker.upper(), period="6mo", interval="1d")
    
    if not data.empty:
        fig = go.Figure()
        data['Date'] = pd.to_datetime(data['Date'])
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))

        fig.update_layout(title=f"{ticker.upper()} Closing Price", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

        # 📰 NEWS SENTIMENT SECTION
# --------------------------
st.markdown("## 📰 News Headlines & Sentiment")
refresh = st.button("🔄 Refresh News")

if ticker and refresh:
    with st.spinner("Fetching latest headlines..."):
        df_news = fetch_news_with_sentiment(ticker.upper(), limit=20)

    if not df_news.empty:
        # 🟢🔴 Color-code sentiment
        def sentiment_color(val):
            color = "green" if val > 0 else "red" if val < 0 else "gray"
            return f"color: {color}"

        styled_df = df_news.style.applymap(sentiment_color, subset=["Sentiment Score"])
        st.dataframe(styled_df, use_container_width=True)

        avg_score = round(df_news["Sentiment Score"].mean(), 2)
        st.metric(label="Average News Sentiment", value=avg_score)
        st.caption("↑ Positive (> 0), ↓ Negative (< 0), ~ Neutral (≈ 0)")
    else:
        st.warning("No news found.")
else:
    st.info("Click the '🔄 Refresh News' button to load sentiment.")

st.markdown("## 🔮 LSTM Forecast (Next 7 Days)")

if ticker:
    with st.spinner(f"Training LSTM model for {ticker.upper()}..."):
        try:
            forecast , rmse = forecast_next_7_days(data.copy())

            # Check if forecast has enough values
            if len(forecast) < 2:
                st.warning("Forecast returned insufficient data. Please try again later.")
                st.stop()

            # Generate future dates
            future_dates = pd.date_range(
                start=data['Date'].iloc[-1] + pd.Timedelta(days=1),
                periods=len(forecast)
            )

            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecasted Price': forecast
            })
            # 📊 Percent Change Metric: Last Actual vs First Forecast
            last_actual_price = data['Close'].iloc[-1].item()
            first_forecast_price = forecast_df['Forecasted Price'].iloc[0].item()


            pct_change = ((first_forecast_price - last_actual_price) / last_actual_price) * 100
            pct_change = round(pct_change, 2)

            st.metric(
            label="Next Day Forecast Change",
            value=f"{pct_change:+.2f}%",
            delta=f"{last_actual_price:.2f} → {first_forecast_price:.2f} USD"
)
            # ✅ Debug logs (print to terminal)
            print("Actual prices (last 10):", data['Close'].tail(10).values.tolist())
            print("Forecasted values:", forecast)
            print("Forecast dates:", list(forecast_df['Date']))


            # 📅 User-selectable range for historical view
            recent_days = st.slider("Select number of previous days to show (before forecast)", 30, 180, 60)
            recent_actual = data.tail(recent_days)
            if st.checkbox("🔍 Show raw forecast/debug data"):
                st.subheader("🧪 Debug: Forecast Check")

         # Show forecasted DataFrame
                st.dataframe(forecast_df)

      # Show recent actual prices
                st.dataframe(recent_actual[['Date', 'Close']].tail(10))

      # Show value ranges
                st.write("Actual Close min/max:", recent_actual['Close'].min(), "/", recent_actual['Close'].max())
                st.write("Forecast min/max:", forecast_df['Forecasted Price'].min(), "/", forecast_df['Forecasted Price'].max())


            fig_forecast = go.Figure()

            # Actual Price Trace with hovertemplate
            fig_forecast.add_trace(go.Scatter(
            x=recent_actual['Date'],
            y=recent_actual['Close'],
            mode='lines',
            name='Actual Price',
            line=dict(color='skyblue'),
            hovertemplate='%{x}<br>%{y:.2f} USD'

))

# Forecast Price Trace with hovertemplate
            forecast_color = 'lime' if forecast_df['Forecasted Price'].iloc[-1] > forecast_df['Forecasted Price'].iloc[0] else 'red'
            fig_forecast.add_trace(go.Scatter(
    x=forecast_df['Date'],
    y=forecast_df['Forecasted Price'],
    mode='lines+markers',
    name='Forecasted Price',
    line=dict(color=forecast_color, width=3, dash='dash'),
    marker=dict(color=forecast_color, size=8),
    hovertemplate='%{x}<br>%{y:.2f} USD'

))

# Forecast Region Shading
            fig_forecast.add_vrect(
    x0=forecast_df['Date'].iloc[0],
    x1=forecast_df['Date'].iloc[-1],
    fillcolor="rgba(255, 0, 0, 0.1)" if forecast_color == 'red' else "rgba(0, 255, 0, 0.1)",
    layer="below",
    line_width=0,
    annotation_text="Forecast Region",
    annotation_position="top left",
    annotation=dict(font=dict(size=12))
)
  
       
    # Combine actual and forecast for Y-axis range
            combined_prices = pd.concat([
            recent_actual[['Date', 'Close']].rename(columns={'Close': 'Price'}),
            forecast_df.rename(columns={'Forecasted Price': 'Price'})
])

            # Combine actual & forecasted prices for dynamic y-axis range
            all_prices = pd.concat([
            recent_actual['Close'],
            forecast_df['Forecasted Price']
])
            y_min, y_max = all_prices.min(), all_prices.max()

            fig_forecast.update_layout(
            title=f"{ticker.upper()} - 7 Day Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis=dict(range=[recent_actual['Date'].iloc[0], forecast_df['Date'].iloc[-1]]),
            yaxis=dict(range=[y_min * 0.995, y_max * 1.005])  # small padding
)

            st.plotly_chart(fig_forecast, use_container_width=True)

        # 🧾 Forecast Table
            st.markdown("### 📋 Forecast Table (Next 7 Days)")
            st.dataframe(forecast_df)
            st.metric("📊 RMSE on Training Data", f"{rmse:.2f} USD")

            with st.expander("🧠 Model Info: LSTM Forecasting"):
                st.markdown("""
    This forecasting module uses a **Long Short-Term Memory (LSTM)** neural network built with Keras.

    - **Architecture**: 2 stacked LSTM layers with 50 units each, followed by a Dense output layer.
    - **Input**: Past 60 days of stock closing prices (scaled using MinMaxScaler).
    - **Training**: 10 epochs on 6 months of daily stock data.
    - **Output**: Forecast of the next 7 days of closing prices.

    **Why LSTM?**
    - LSTMs are a type of Recurrent Neural Network (RNN) good at learning temporal dependencies.
    - They're especially effective for time series forecasting because they retain memory over long sequences.

    **Metric Used**:
    - RMSE (Root Mean Squared Error) is displayed to assess model fit on training data.
    
    > ⚠️ Note: This model is trained per session and does not use advanced tuning, so results may vary.
    """)

        except Exception as e:
          import traceback
          st.error(" Forecasting failed.")
          st.code(traceback.format_exc())


