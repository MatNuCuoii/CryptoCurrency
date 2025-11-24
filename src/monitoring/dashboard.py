# src/monitoring/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
import os
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.visualization.visualizer import CryptoVisualizer
from src.assistant.rag_assistant import RAGCryptoAssistant
from dotenv import load_dotenv

# Load environment variables từ file .env
load_dotenv()

class MonitoringDashboard:
    """
    Comprehensive Streamlit dashboard for cryptocurrency prediction monitoring.
    Features:
    - Historical Data visualization from raw/train
    - Training Results and metrics
    - Future Predictions visualization
    """

    def __init__(self, data_dir: str = "data/raw/train", results_dir: str = "results"):
        st.set_page_config(
            page_title="Crypto Prediction Monitor", 
            page_icon="📈", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self.logger = self._setup_logger()
        self.visualizer = CryptoVisualizer()
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.coins = [
            "bitcoin", 
            "ethereum", 
            "litecoin",
            "binancecoin",
            "cardano",
            "solana",
            "pancakeswap",
            "axieinfinity",
            "thesandbox"
        ]
        self.ai_assistant = None

    def _setup_logger(self):
        logger = logging.getLogger("MonitoringDashboard")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_historical_data(self, coin: str) -> pd.DataFrame:
        """Load historical data from raw/train directory."""
        csv_files = list(self.data_dir.glob(f"{coin}_binance_*.csv"))
        if not csv_files:
            self.logger.warning(f"No historical data found for {coin}")
            return pd.DataFrame()
        
        # Load the most recent file
        latest_file = sorted(csv_files)[-1]
        try:
            df = pd.read_csv(latest_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error loading {latest_file}: {e}")
            return pd.DataFrame()

    def load_results(self, coin: str) -> dict:
        """Load training results for a specific coin."""
        result_files = list(self.results_dir.glob(f"{coin}_results_*.json"))
        if not result_files:
            return {}
        
        latest_file = sorted(result_files)[-1]
        try:
            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return {}

    def load_predictions(self, coin: str) -> dict:
        """Load future predictions for a specific coin."""
        pred_file = self.results_dir / "predictions" / f"{coin}_future_predictions.json"
        if not pred_file.exists():
            return {}
        
        try:
            with open(pred_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading predictions: {e}")
            return {}

    def plot_historical_data(self, df: pd.DataFrame, coin: str):
        """Plot historical OHLCV data."""
        if df.empty:
            st.warning(f"No historical data available for {coin}")
            return

        # 1. Main Candlestick chart with Volume
        st.subheader("Price & Volume Overview")
        fig_main = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{coin.capitalize()} Price (OHLC)', 'Volume'),
            row_heights=[0.7, 0.3]
        )

        # Candlestick
        fig_main.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        # Volume bars
        colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                  for i in range(len(df))]
        fig_main.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )

        fig_main.update_layout(
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig_main, use_container_width=True)

        # 2. Individual OHLC Line Charts
        st.subheader("OHLC Trends")
        fig_ohlc = go.Figure()
        
        fig_ohlc.add_trace(go.Scatter(
            x=df.index, y=df['open'], 
            name='Open', mode='lines',
            line=dict(color='blue', width=1)
        ))
        fig_ohlc.add_trace(go.Scatter(
            x=df.index, y=df['high'], 
            name='High', mode='lines',
            line=dict(color='green', width=1)
        ))
        fig_ohlc.add_trace(go.Scatter(
            x=df.index, y=df['low'], 
            name='Low', mode='lines',
            line=dict(color='red', width=1)
        ))
        fig_ohlc.add_trace(go.Scatter(
            x=df.index, y=df['close'], 
            name='Close', mode='lines',
            line=dict(color='black', width=2)
        ))
        
        fig_ohlc.update_layout(
            title=f"{coin.capitalize()} - OHLC Price Lines",
            xaxis_title="Date",
            yaxis_title="Price (USDT)",
            height=450,
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.15,
                xanchor='center',
                x=0.5
            ),
            margin=dict(b=100, t=60)
        )
        
        st.plotly_chart(fig_ohlc, use_container_width=True)

        # 3. Price Range (High-Low) Chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Daily Price Range")
            fig_range = go.Figure()
            
            # Calculate daily range
            df['range'] = df['high'] - df['low']
            df['range_pct'] = ((df['high'] - df['low']) / df['low']) * 100
            
            fig_range.add_trace(go.Bar(
                x=df.index, 
                y=df['range'],
                name='Price Range',
                marker_color='purple'
            ))
            
            fig_range.update_layout(
                title="Daily High-Low Range",
                xaxis_title="Date",
                yaxis_title="Range (USDT)",
                height=350
            )
            
            st.plotly_chart(fig_range, use_container_width=True)
        
        with col2:
            st.subheader("Volume Distribution")
            fig_vol_dist = go.Figure()
            
            fig_vol_dist.add_trace(go.Histogram(
                x=df['volume'],
                nbinsx=50,
                name='Volume Distribution',
                marker_color='lightblue'
            ))
            
            fig_vol_dist.update_layout(
                title="Volume Distribution",
                xaxis_title="Volume",
                yaxis_title="Frequency",
                height=350
            )
            
            st.plotly_chart(fig_vol_dist, use_container_width=True)

        # 4. Price Statistics Cards
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fig_close = go.Figure()
            fig_close.add_trace(go.Indicator(
                mode="number+delta",
                value=df['close'].iloc[-1],
                title={'text': "Close Price"},
                delta={'reference': df['close'].iloc[-2], 'relative': True},
                number={'prefix': "$"}
            ))
            fig_close.update_layout(height=200)
            st.plotly_chart(fig_close, use_container_width=True)

        with col2:
            fig_high = go.Figure()
            fig_high.add_trace(go.Indicator(
                mode="number+delta",
                value=df['high'].iloc[-1],
                title={'text': "High Price"},
                delta={'reference': df['high'].iloc[-2], 'relative': True},
                number={'prefix': "$"}
            ))
            fig_high.update_layout(height=200)
            st.plotly_chart(fig_high, use_container_width=True)
        
        with col3:
            fig_low = go.Figure()
            fig_low.add_trace(go.Indicator(
                mode="number+delta",
                value=df['low'].iloc[-1],
                title={'text': "Low Price"},
                delta={'reference': df['low'].iloc[-2], 'relative': True},
                number={'prefix': "$"}
            ))
            fig_low.update_layout(height=200)
            st.plotly_chart(fig_low, use_container_width=True)

        with col4:
            fig_vol_ind = go.Figure()
            fig_vol_ind.add_trace(go.Indicator(
                mode="number+delta",
                value=df['volume'].iloc[-1],
                title={'text': "Volume"},
                delta={'reference': df['volume'].mean(), 'relative': True}
            ))
            fig_vol_ind.update_layout(height=200)
            st.plotly_chart(fig_vol_ind, use_container_width=True)

        # 5. Box Plots for OHLC
        st.subheader("OHLC Distribution")
        fig_box = go.Figure()
        
        fig_box.add_trace(go.Box(y=df['open'], name='Open', marker_color='blue'))
        fig_box.add_trace(go.Box(y=df['high'], name='High', marker_color='green'))
        fig_box.add_trace(go.Box(y=df['low'], name='Low', marker_color='red'))
        fig_box.add_trace(go.Box(y=df['close'], name='Close', marker_color='black'))
        
        fig_box.update_layout(
            title=f"{coin.capitalize()} - OHLC Box Plot",
            yaxis_title="Price (USDT)",
            height=400,
            margin=dict(b=60, t=60)
        )
        
        st.plotly_chart(fig_box, use_container_width=True)

        # 6. Volume Analysis
        st.subheader("Volume Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Volume over time
            fig_vol_time = go.Figure()
            fig_vol_time.add_trace(go.Scatter(
                x=df.index,
                y=df['volume'],
                fill='tozeroy',
                name='Volume',
                line=dict(color='lightblue')
            ))
            # Add moving average
            df['vol_ma'] = df['volume'].rolling(window=7).mean()
            fig_vol_time.add_trace(go.Scatter(
                x=df.index,
                y=df['vol_ma'],
                name='7-Day MA',
                line=dict(color='darkblue', width=2)
            ))
            
            fig_vol_time.update_layout(
                title="Volume Over Time",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=350
            )
            st.plotly_chart(fig_vol_time, use_container_width=True)
        
        with col2:
            # Price vs Volume correlation
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=df['volume'],
                y=df['close'],
                mode='markers',
                name='Price vs Volume',
                marker=dict(
                    size=5,
                    color=df.index.astype('int64') / 10**9,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Time")
                )
            ))
            
            fig_corr.update_layout(
                title="Price vs Volume Correlation",
                xaxis_title="Volume",
                yaxis_title="Close Price (USDT)",
                height=350
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    def plot_training_results(self, results: dict, coin: str):
        """Plot training history and evaluation metrics."""
        if not results:
            st.warning(f"No training results available for {coin}")
            return

        # Metrics
        evaluation = results.get('evaluation', {})
        if evaluation:
            st.subheader("Model Performance Metrics")
            cols = st.columns(len(evaluation))
            for idx, (metric, value) in enumerate(evaluation.items()):
                with cols[idx]:
                    st.metric(
                        label=metric.upper().replace('_', ' '),
                        value=f"{value:.4f}"
                    )

        # Training history
        history = results.get('history', {})
        if history:
            st.subheader("Training History")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Loss', 'MAE'),
                horizontal_spacing=0.1
            )

            # Loss
            if 'loss' in history:
                fig.add_trace(
                    go.Scatter(y=history['loss'], name='Train Loss', mode='lines'),
                    row=1, col=1
                )
            if 'val_loss' in history:
                fig.add_trace(
                    go.Scatter(y=history['val_loss'], name='Val Loss', mode='lines'),
                    row=1, col=1
                )

            # MAE
            if 'mae' in history:
                fig.add_trace(
                    go.Scatter(y=history['mae'], name='Train MAE', mode='lines'),
                    row=1, col=2
                )
            if 'val_mae' in history:
                fig.add_trace(
                    go.Scatter(y=history['val_mae'], name='Val MAE', mode='lines'),
                    row=1, col=2
                )

            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_layout(height=400, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)

        # Predictions vs Actual (if available)
        if 'predictions' in results and 'actual' in results:
            st.subheader("Predictions vs Actual")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=results['actual'],
                name='Actual',
                mode='lines',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                y=results['predictions'],
                name='Predicted',
                mode='lines',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"{coin.capitalize()} - Test Set Predictions",
                xaxis_title="Time Step",
                yaxis_title="Price (USDT)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def plot_future_predictions(self, predictions: dict, coin: str):
        """Plot future price predictions."""
        if not predictions:
            st.warning(f"No future predictions available for {coin}")
            return

        st.subheader("Future Price Predictions")
        
        # Display prediction metadata
        if 'prediction_generated_at' in predictions:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Generated: {predictions['prediction_generated_at']}")
            with col2:
                st.info(f"Forecast Horizon: {predictions.get('forecast_horizon_days', 'N/A')} days")
        
        # Handle different prediction data formats
        pred_data = predictions.get('predictions', [])
        timestamps = predictions.get('timestamps', [])
        
        if not pred_data:
            st.warning("No prediction data found")
            return

        # Extract price values if pred_data contains dictionaries
        if isinstance(pred_data, list) and len(pred_data) > 0:
            if isinstance(pred_data[0], dict):
                # If predictions are in dict format like {'expected_price': value, 'day': ...}
                prices = [p.get('expected_price', p.get('price', p.get('predicted_price', 0))) for p in pred_data]
                if not timestamps:
                    # Try to get timestamps or day numbers
                    if 'timestamp' in pred_data[0]:
                        timestamps = [p['timestamp'] for p in pred_data]
                    elif 'day' in pred_data[0]:
                        timestamps = [f"Day {p['day']}" for p in pred_data]
            else:
                # If predictions are already numbers
                prices = pred_data
        else:
            st.warning("Invalid prediction data format")
            return

        # Create prediction chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps if timestamps else list(range(len(prices))),
            y=prices,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"{coin.capitalize()} - Future Price Forecast",
            xaxis_title="Date/Time",
            yaxis_title="Price (USDT)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Prediction statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Min", f"${min(prices):.2f}")
        with col2:
            st.metric("Predicted Max", f"${max(prices):.2f}")
        with col3:
            avg_price = sum(prices) / len(prices)
            st.metric("Predicted Avg", f"${avg_price:.2f}")
        
        # Show prediction trend
        if len(prices) > 1 and prices[0] != 0:
            trend = ((prices[-1] - prices[0]) / prices[0]) * 100
            trend_text = "Upward" if trend > 0 else "Downward"
            st.info(f"Overall Trend: {trend_text} ({trend:+.2f}%)")
        elif len(prices) > 1:
            change = prices[-1] - prices[0]
            trend_text = "Upward" if change > 0 else "Downward" if change < 0 else "Flat"
            st.info(f"Overall Trend: {trend_text} (Change: ${change:.2f})")
        
        # Display explanation if available
        if 'explanation' in predictions:
            with st.expander("About These Predictions"):
                st.write(predictions['explanation'])
    
    def load_ai_assistant(self, api_key: str):
        """Load AI Assistant với caching"""
        if 'ai_assistant' not in st.session_state:
            try:
                assistant = RAGCryptoAssistant(
                    api_key=api_key,
                    data_dir="data/raw",
                    model="gpt-4"
                )
                assistant.load_historical_data()
                st.session_state['ai_assistant'] = assistant
                return assistant
            except Exception as e:
                st.error(f"Lỗi khởi tạo AI: {str(e)}")
                return None
        return st.session_state['ai_assistant']
    
    def show_ai_assistant_page(self, selected_coin: str):
        """AI Assistant - Modern Professional Layout"""
        # Check API key
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
            return
        
        # Load AI Assistant
        assistant = self.load_ai_assistant(api_key)
        if not assistant:
            st.error("Failed to initialize AI Assistant")
            return
        
        # Initialize chat history
        chat_key = f'chat_history_{selected_coin}'
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []
        
        # Professional modern CSS
        st.markdown("""
            <style>
            /* AI Assistant specific styles */
            .ai-header {
                background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
                border: 1px solid #2a2a2a;
                border-radius: 8px;
                padding: 2rem;
                margin-bottom: 2rem;
                position: relative;
                overflow: hidden;
            }
            
            .ai-header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, #00ff88 0%, #00cc70 100%);
            }
            
            .coin-badge {
                display: inline-block;
                background: #00ff88;
                color: #000;
                padding: 0.4rem 1rem;
                border-radius: 20px;
                font-weight: 700;
                font-size: 0.85rem;
                letter-spacing: 0.05em;
            }
            
            .quick-action-card {
                background: #0a0a0a;
                border: 1px solid #2a2a2a;
                border-radius: 8px;
                padding: 1.25rem;
                cursor: pointer;
                transition: all 0.2s ease;
                text-align: center;
                height: 100%;
            }
            
            .quick-action-card:hover {
                border-color: #00ff88;
                background: #0f0f0f;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 255, 136, 0.1);
            }
            
            .quick-action-icon {
                font-size: 2rem;
                margin-bottom: 0.75rem;
            }
            
            .quick-action-title {
                color: #ffffff;
                font-weight: 600;
                font-size: 0.95rem;
                margin-bottom: 0.5rem;
            }
            
            .quick-action-desc {
                color: #666;
                font-size: 0.8rem;
                line-height: 1.4;
            }
            
            .chat-container {
                background: #0a0a0a;
                border: 1px solid #2a2a2a;
                border-radius: 8px;
                padding: 1.5rem;
                min-height: 500px;
                max-height: 600px;
                overflow-y: auto;
            }
            
            .chat-empty {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 450px;
                text-align: center;
            }
            
            .chat-empty-icon {
                font-size: 4rem;
                margin-bottom: 1rem;
                opacity: 0.3;
            }
            
            .section-title {
                color: #ffffff;
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.15em;
                margin-bottom: 1rem;
                padding-left: 0.5rem;
                border-left: 3px solid #00ff88;
            }
            
            .stats-card {
                background: #0a0a0a;
                border: 1px solid #2a2a2a;
                border-radius: 6px;
                padding: 1rem;
                text-align: center;
            }
            
            .stats-value {
                color: #00ff88;
                font-size: 1.5rem;
                font-weight: 700;
                font-family: 'JetBrains Mono', monospace;
            }
            
            .stats-label {
                color: #666;
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-top: 0.25rem;
            }
            
            /* Custom scrollbar for chat */
            .chat-container::-webkit-scrollbar {
                width: 6px;
            }
            
            .chat-container::-webkit-scrollbar-track {
                background: #0a0a0a;
            }
            
            .chat-container::-webkit-scrollbar-thumb {
                background: #2a2a2a;
                border-radius: 3px;
            }
            
            .chat-container::-webkit-scrollbar-thumb:hover {
                background: #3a3a3a;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Header Section
        st.markdown(f"""
            <div class='ai-header'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <div style='color: #999; font-size: 0.85rem; margin-bottom: 0.5rem;'>AI Investment Assistant</div>
                        <h1 style='color: #ffffff; margin: 0; font-size: 2rem; font-weight: 700;'>Chat with AI</h1>
                        <div style='margin-top: 1rem;'>
                            <span class='coin-badge'>{selected_coin.upper()}</span>
                        </div>
                    </div>
                    <div style='text-align: right;'>
                        <div style='color: #666; font-size: 0.8rem;'>Powered by</div>
                        <div style='color: #00ff88; font-size: 1.1rem; font-weight: 600;'>OpenAI GPT</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Layout: Sidebar + Main Chat
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            # Quick Actions Section
            st.markdown("<div class='section-title'>Quick Actions</div>", unsafe_allow_html=True)
            
            # Quick Action Cards
            if st.button("📊 Market Analysis", key="qa1", use_container_width=True):
                prompt = f"Provide comprehensive market analysis for {selected_coin} including current trends, volume, and technical indicators."
                st.session_state[chat_key].append({"role": "user", "content": prompt})
                st.rerun()
            
            if st.button("💰 Investment Strategy", key="qa2", use_container_width=True):
                prompt = f"Based on current data and predictions, what's the best investment strategy for {selected_coin}?"
                st.session_state[chat_key].append({"role": "user", "content": prompt})
                st.rerun()
            
            if st.button("📈 Price Prediction", key="qa3", use_container_width=True):
                prompt = f"What is your price forecast for {selected_coin}? Include timeframe and confidence level."
                st.session_state[chat_key].append({"role": "user", "content": prompt})
                st.rerun()
            
            if st.button("⚠️ Risk Assessment", key="qa4", use_container_width=True):
                prompt = f"What are the risks and opportunities for investing in {selected_coin} right now?"
                st.session_state[chat_key].append({"role": "user", "content": prompt})
                st.rerun()
            
            if st.button("🎯 Technical Signals", key="qa5", use_container_width=True):
                prompt = f"Analyze technical indicators (RSI, MACD, MA) for {selected_coin} and provide trading signals."
                st.session_state[chat_key].append({"role": "user", "content": prompt})
                st.rerun()
            
            # Chat Stats
            st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Session Stats</div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class='stats-card'>
                    <div class='stats-value'>{len(st.session_state[chat_key])}</div>
                    <div class='stats-label'>Messages</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            
            # Action Buttons
            if st.button("🗑️ Clear Chat", key="clear_btn", use_container_width=True):
                st.session_state[chat_key] = []
                st.rerun()
            
            if st.session_state[chat_key]:
                chat_text = "\n\n".join([
                    f"{'USER' if msg['role'] == 'user' else 'AI ASSISTANT'}:\n{msg['content']}"
                    for msg in st.session_state[chat_key]
                ])
                st.download_button(
                    label="💾 Export Chat",
                    data=chat_text,
                    file_name=f"ai_chat_{selected_coin}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="export_btn"
                )
        
        with col_right:
            # Chat Area
            st.markdown("<div class='section-title'>Conversation</div>", unsafe_allow_html=True)
            
            # Chat Container
            if not st.session_state[chat_key]:
                st.markdown("""
                    <div class='chat-container'>
                        <div class='chat-empty'>
                            <div class='chat-empty-icon'>💬</div>
                            <div style='color: #ffffff; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;'>Start a Conversation</div>
                            <div style='color: #666; font-size: 0.9rem;'>Ask me anything about """ + selected_coin.upper() + """</div>
                            <div style='color: #444; font-size: 0.85rem; margin-top: 0.5rem;'>Use quick actions on the left or type below</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Display messages using Streamlit's chat components
                chat_container = st.container()
                with chat_container:
                    for message in st.session_state[chat_key]:
                        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
                            st.markdown(message["content"])
            
            # Chat Input
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            user_input = st.chat_input(f"Type your question about {selected_coin.upper()}...", key="chat_input_main")
            
            if user_input:
                # Add user message
                st.session_state[chat_key].append({"role": "user", "content": user_input})
                
                # Get AI response
                with st.spinner("🤔 AI is analyzing..."):
                    try:
                        response = assistant.chat(
                            coin=selected_coin,
                            user_message=user_input,
                            conversation_history=st.session_state[chat_key][:-1]
                        )
                        
                        st.session_state[chat_key].append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"⚠️ Error: {str(e)}\n\nPlease try again or rephrase your question."
                        st.session_state[chat_key].append({"role": "assistant", "content": error_msg})
                
                st.rerun()
    
    def render_sidebar(self):
        """Render the sidebar navigation with new structure."""
        with st.sidebar:
            # Logo/Header - Ultra Professional
            st.markdown("""
                <div style='padding: 2rem 1.5rem 2rem 1.5rem; border-bottom: 1px solid #1a1a1a;'>
                    <div style='font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem; letter-spacing: 0.05em; text-transform: uppercase;'>
                        CRYPTO ANALYSIS
                    </div>
                    <div style='color: #666; font-size: 0.8rem; font-weight: 500; letter-spacing: 0.02em;'>
                        Deep Learning Platform
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Navigation Menu
            st.markdown("""
                <div style='margin: 2rem 0 1rem 1.5rem;'>
                    <h4 style='color: #666; margin: 0; font-weight: 700; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;'>
                        Navigation
                    </h4>
                </div>
            """, unsafe_allow_html=True)
            
            # Radio button navigation - No icons
            page = st.radio(
                "Chọn trang",
                [
                    "Overview",
                    "Price Analysis",
                    "Technical Indicators",
                    "Compare Coins",
                    "Predictions",
                    "AI Assistant"
                ],
                label_visibility="collapsed"
            )
            
            st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
            
            # Coin selector (chỉ hiện với các trang cần chọn coin)
            selected_coin = None
            if page in ["Price Analysis", "Technical Indicators", "Predictions", "AI Assistant"]:
                st.markdown("""
                    <div style='margin: 1rem 0;'>
                        <h4 style='color: #667eea; margin-bottom: 1rem; font-weight: bold; letter-spacing: 1px;'>
                            SELECT CRYPTOCURRENCY
                        </h4>
                    </div>
                """, unsafe_allow_html=True)
                
                selected_coin = st.selectbox(
                    "Cryptocurrency",
                    self.coins,
                    format_func=lambda x: x.upper(),
                    label_visibility="collapsed"
                )
                
                # Hiển thị icon coin với animation
                coin_icons = {
                    "bitcoin": "₿",
                    "ethereum": "Ξ",
                    "binancecoin": "BNB",
                    "cardano": "₳",
                    "solana": "◎",
                    "litecoin": "Ł",
                    "pancakeswap": "CAKE",
                    "axieinfinity": "AXS",
                    "thesandbox": "SAND"
                }
                
                coin_colors = {
                    "bitcoin": "linear-gradient(135deg, #f7931a 0%, #f2a900 100%)",
                    "ethereum": "linear-gradient(135deg, #627eea 0%, #8c9eff 100%)",
                    "binancecoin": "linear-gradient(135deg, #f3ba2f 0%, #ffd500 100%)",
                    "cardano": "linear-gradient(135deg, #0033ad 0%, #0055ff 100%)",
                    "solana": "linear-gradient(135deg, #00ffa3 0%, #dc1fff 100%)",
                    "litecoin": "linear-gradient(135deg, #345d9d 0%, #5b8dd3 100%)",
                    "pancakeswap": "linear-gradient(135deg, #d1884f 0%, #ffb347 100%)",
                    "axieinfinity": "linear-gradient(135deg, #0055d4 0%, #0099ff 100%)",
                    "thesandbox": "linear-gradient(135deg, #00adef 0%, #00d4ff 100%)"
                }
                
                if selected_coin:
                    icon = coin_icons.get(selected_coin, "COIN")
                    gradient = coin_colors.get(selected_coin, "linear-gradient(135deg, #667eea 0%, #764ba2 100%)")
                    
                    st.markdown(f"""
                        <div style='text-align: center; 
                                    padding: 1.5rem; 
                                    background: {gradient}; 
                                    border-radius: 15px; 
                                    margin: 1rem 0;
                                    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
                                    transform: translateY(0);
                                    transition: all 0.3s ease;'>
                            <h1 style='margin: 0; 
                                       color: white; 
                                       font-size: 3rem;
                                       text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
                                       animation: pulse 2s infinite;'>{icon}</h1>
                            <p style='color: white; 
                                     margin: 0.5rem 0 0 0; 
                                     font-weight: bold; 
                                     font-size: 1.1rem;
                                     letter-spacing: 2px;
                                     text-shadow: 1px 1px 3px rgba(0,0,0,0.3);'>{selected_coin.upper()}</p>
                        </div>
                        
                        <style>
                            @keyframes pulse {{
                                0%, 100% {{ transform: scale(1); }}
                                50% {{ transform: scale(1.05); }}
                            }}
                        </style>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
            
            # Settings & Info Section
            st.markdown("""
                <div style='margin: 2rem 0 1rem 1.5rem; padding-top: 2rem; border-top: 1px solid #1a1a1a;'>
                    <h4 style='color: #666; margin: 0; font-weight: 700; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;'>
                        Settings
                    </h4>
                </div>
            """, unsafe_allow_html=True)
            
            # Theme toggle & Refresh buttons with better styling
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Theme", use_container_width=True, help="Toggle Theme"):
                    st.info("Theme toggle (coming soon)")
            with col2:
                if st.button("Refresh", use_container_width=True, help="Refresh Data"):
                    st.rerun()
            
            # Export options
            with st.expander("Export Options", expanded=False):
                st.checkbox("Auto-save charts", value=False, key="auto_save")
                st.checkbox("Include raw data", value=True, key="include_raw")
                export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel", "PDF"], key="export_format")
                
                if st.button("Export Now", use_container_width=True):
                    st.success("Export feature coming soon!")
            
            st.markdown("---")
            
            # Data Status section
            st.markdown("""
                <div style='margin: 2rem 0 1rem 1.5rem; padding-top: 2rem; border-top: 1px solid #1a1a1a;'>
                    <h4 style='color: #666; margin: 0; font-weight: 700; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;'>
                        System Status
                    </h4>
                </div>
            """, unsafe_allow_html=True)
            
            # Quick stats with gradient cards
            total_coins = len(self.coins)
            
            st.markdown(f"""
                <div style='background: #0a0a0a;
                            padding: 1.25rem;
                            border-radius: 4px;
                            margin-bottom: 0.75rem;
                            border: 1px solid #1a1a1a;'>
                    <p style='color: #666; margin: 0; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;'>Total Assets</p>
                    <h2 style='color: white; margin: 0.5rem 0 0 0; font-weight: 700; font-size: 2rem; font-family: "JetBrains Mono", monospace;'>{total_coins}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Data quality indicator
            data_quality = 98.5  # Placeholder
            quality_color = "#00ff88" if data_quality >= 95 else "#ffd700" if data_quality >= 85 else "#ff6b6b"
            quality_emoji = "🟢" if data_quality >= 95 else "🟡" if data_quality >= 85 else "🔴"
            
            st.markdown(f"""
                <div style='background: #0a0a0a;
                            padding: 1.25rem;
                            border-radius: 4px;
                            margin-bottom: 0.75rem;
                            border: 1px solid #1a1a1a;'>
                    <p style='color: #666; margin: 0; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;'>Data Quality</p>
                    <h2 style='color: white; margin: 0.5rem 0 0 0; font-weight: 700; font-size: 2rem; font-family: "JetBrains Mono", monospace;'>{data_quality}%</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Last update time
            from datetime import datetime
            last_update = datetime.now().strftime("%d/%m/%Y %H:%M")
            
            st.markdown(f"""
                <div style='background: #0a0a0a;
                            padding: 1.25rem;
                            border-radius: 4px;
                            border: 1px solid #1a1a1a;'>
                    <p style='color: #666; margin: 0; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;'>Last Update</p>
                    <h4 style='color: #999; margin: 0.5rem 0 0 0; font-weight: 500; font-size: 0.85rem; font-family: "JetBrains Mono", monospace;'>{last_update}</h4>
                </div>
            """, unsafe_allow_html=True)
            
            # Footer - Minimal
            st.markdown("""
                <div style='padding: 2rem 1.5rem; 
                            margin-top: 3rem;
                            border-top: 1px solid #1a1a1a;'>
                    <p style='color: #666; margin: 0; font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;'>Version 2.0</p>
                    <p style='color: #444; margin: 0.5rem 0 0 0; font-size: 0.7rem;'>Enterprise Edition</p>
                </div>
            """, unsafe_allow_html=True)
            
        return page, selected_coin
    
    def show(self):
        """Main dashboard display."""
        # Ultra-Professional Dark Theme CSS - Enterprise Grade
        st.markdown("""
            <style>
            /* Import Professional Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
            
            /* Global Styles - Ultra Clean */
            * {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                letter-spacing: -0.01em;
            }
            
            /* Main container - Premium Dark */
            .main {
                background: #000000;
                padding: 3rem 4rem;
            }
            
            /* Sidebar - Premium Minimal */
            [data-testid="stSidebar"] {
                background: #0a0a0a;
                border-right: 1px solid #1a1a1a;
            }
            
            [data-testid="stSidebar"] > div:first-child {
                background: #0a0a0a;
            }
            
            /* Radio buttons - Ultra Clean */
            [data-testid="stSidebar"] .stRadio > label {
                background: transparent;
                padding: 0.75rem 1.25rem;
                border-radius: 4px;
                margin: 0.15rem 0;
                transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
                border-left: 2px solid transparent;
                color: #666;
                font-size: 0.9rem;
                font-weight: 500;
            }
            
            [data-testid="stSidebar"] .stRadio > label:hover {
                background: rgba(255, 255, 255, 0.03);
                color: #fff;
                border-left-color: #0066FF;
            }
            
            /* Selected radio button */
            [data-testid="stSidebar"] .stRadio [data-checked="true"] {
                background: rgba(0, 102, 255, 0.08);
                color: #fff;
                border-left-color: #0066FF;
            }
            
            /* Selectbox - Premium */
            [data-testid="stSidebar"] .stSelectbox > div > div {
                background: #0f0f0f;
                border: 1px solid #1a1a1a;
                border-radius: 4px;
                color: white;
                font-size: 0.9rem;
            }
            
            /* Buttons in sidebar */
            [data-testid="stSidebar"] .stButton > button {
                background: #0f0f0f;
                color: #999;
                border: 1px solid #1a1a1a;
                border-radius: 4px;
                font-weight: 500;
                padding: 0.6rem 1rem;
                transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
                font-size: 0.85rem;
            }
            
            [data-testid="stSidebar"] .stButton > button:hover {
                background: #1a1a1a;
                border-color: #0066FF;
                color: #fff;
            }
            
            /* Expander in sidebar */
            [data-testid="stSidebar"] .streamlit-expanderHeader {
                background: #0f0f0f;
                border-radius: 4px;
                color: #666;
                font-weight: 500;
                border: 1px solid #1a1a1a;
                font-size: 0.85rem;
            }
            
            /* Metrics - Premium */
            [data-testid="stMetricValue"] {
                font-size: 2rem;
                font-weight: 700;
                color: #fff;
                font-family: 'JetBrains Mono', monospace;
            }
            
            [data-testid="stMetricDelta"] {
                font-size: 0.85rem;
                font-weight: 600;
            }
            
            /* Metric containers - Premium cards */
            [data-testid="metric-container"] {
                background: #0a0a0a;
                border: 1px solid #1a1a1a;
                border-radius: 6px;
                padding: 1.5rem;
                transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            [data-testid="metric-container"]:hover {
                background: #0f0f0f;
                border-color: #252525;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            }
            
            /* Headers - Premium Typography */
            h1 {
                color: #ffffff;
                font-weight: 700;
                font-size: 2.25rem;
                letter-spacing: -0.03em;
                margin-bottom: 0.5rem;
            }
            
            h2 {
                color: #ffffff;
                font-weight: 600;
                font-size: 1.5rem;
                letter-spacing: -0.02em;
                margin-bottom: 0.75rem;
            }
            
            h3 {
                color: #cccccc;
                font-size: 1.1rem;
                font-weight: 600;
                letter-spacing: -0.01em;
                margin-bottom: 1rem;
            }
            
            /* Info/Warning/Success boxes */
            .stAlert {
                border-radius: 8px;
                border-left: 3px solid;
                background: #111111;
                border: 1px solid #1f1f1f;
            }
            
            [data-baseweb="notification"] {
                background: #111111;
                border-radius: 8px;
                border: 1px solid #1f1f1f;
            }
            
            /* Buttons in main area */
            .stButton>button {
                background: #0066FF;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: 600;
                padding: 0.65rem 1.75rem;
                transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            
            .stButton>button:hover {
                background: #0052CC;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 102, 255, 0.3);
            }
            
            /* Tabs styling - Premium */
            .stTabs [data-baseweb="tab-list"] {
                gap: 0;
                background: transparent;
                border-bottom: 1px solid #1a1a1a;
            }
            
            .stTabs [data-baseweb="tab"] {
                background: transparent;
                border-radius: 0;
                color: #666;
                font-weight: 600;
                padding: 1rem 2rem;
                border-bottom: 2px solid transparent;
                transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            
            .stTabs [data-baseweb="tab"]:hover {
                color: #999;
                background: transparent;
                border-bottom-color: #333;
            }
            
            .stTabs [aria-selected="true"] {
                background: transparent;
                color: #0066FF;
                border-bottom-color: #0066FF;
            }
            
            /* Dataframe styling */
            [data-testid="stDataFrame"] {
                border-radius: 6px;
                overflow: hidden;
                border: 1px solid #1a1a1a;
            }
            
            /* Plotly charts container */
            .js-plotly-plot {
                border-radius: 6px;
                background: #0a0a0a;
                border: 1px solid #1a1a1a;
            }
            
            /* Divider - Ultra Subtle */
            hr {
                border: none;
                height: 1px;
                background: #1a1a1a;
                margin: 3rem 0;
            }
            
            /* Scrollbar - Minimal */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: #0a0a0a;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #2a2a2a;
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #3a3a3a;
            }
            
            /* Expander - Premium */
            .streamlit-expanderHeader {
                background: #0a0a0a;
                border-radius: 4px;
                font-weight: 600;
                border: 1px solid #1a1a1a;
                color: #666;
                font-size: 0.9rem;
            }
            
            .streamlit-expanderHeader:hover {
                background: #0f0f0f;
                color: #fff;
                border-color: #252525;
            }
            
            /* Text input - Premium */
            .stTextInput > div > div > input {
                background: #0a0a0a;
                border: 1px solid #1a1a1a;
                border-radius: 4px;
                color: white;
                padding: 0.7rem 1rem;
                font-size: 0.9rem;
            }
            
            .stTextInput > div > div > input:focus {
                border-color: #0066FF;
                outline: none;
                box-shadow: 0 0 0 2px rgba(0, 102, 255, 0.1);
            }
            
            /* Checkbox */
            .stCheckbox {
                color: #e0e0e0;
            }
            
            /* Selectbox dropdown */
            [data-baseweb="select"] {
                background: #111111;
                border-radius: 6px;
            }
            
            /* Chat input */
            .stChatInput textarea {
                background: #0a0a0a;
                border: 1px solid #1a1a1a;
                border-radius: 6px;
                color: white;
                font-size: 0.9rem;
            }
            
            /* Chat message */
            .stChatMessage {
                background: #0a0a0a;
                border: 1px solid #1a1a1a;
                border-radius: 6px;
                padding: 1.25rem;
            }
            
            /* Text elements */
            p, span, div {
                color: #cccccc;
            }
            
            /* Labels - Premium */
            label {
                color: #666 !important;
                font-weight: 600;
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            
            /* Code blocks */
            code {
                background: #0f0f0f;
                border: 1px solid #1a1a1a;
                border-radius: 3px;
                padding: 0.25rem 0.5rem;
                color: #0066FF;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
            }
            
            /* Tables */
            table {
                background: #0a0a0a;
                border: 1px solid #1a1a1a;
            }
            
            th {
                background: #0f0f0f;
                color: #fff;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-size: 0.8rem;
            }
            
            td {
                border-color: #1a1a1a;
                color: #cccccc;
            }
            
            /* Alert boxes */
            .stAlert {
                background: #0a0a0a;
                border: 1px solid #1a1a1a;
                border-left: 3px solid #0066FF;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Render sidebar and get selected page
        page, selected_coin = self.render_sidebar()
        
        # Main content area based on selected page
        if page == "Overview":
            self.show_overview_page()
        
        elif page == "Price Analysis":
            if selected_coin:
                self.show_price_analysis_page(selected_coin)
            else:
                st.warning("Please select a cryptocurrency from the sidebar")
        
        elif page == "Technical Indicators":
            if selected_coin:
                self.show_technical_indicators_page(selected_coin)
            else:
                st.warning("Please select a cryptocurrency from the sidebar")
        
        elif page == "Compare Coins":
            self.show_comparison_page()
        
        elif page == "Predictions":
            if selected_coin:
                self.show_prediction_page(selected_coin)
            else:
                st.warning("Please select a cryptocurrency from the sidebar")
        
        elif page == "AI Assistant":
            if selected_coin:
                self.show_ai_assistant_page(selected_coin)
            else:
                st.warning("Please select a cryptocurrency from the sidebar")

        # Footer - Ultra Minimal
        st.markdown(
            """
            <div style='text-align: center; padding: 3rem 1rem; margin-top: 4rem; border-top: 1px solid #1a1a1a;'>
                <p style='color: #666; margin: 0; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;'>Crypto Analytics Platform</p>
                <p style='font-size: 0.7rem; color: #444; margin: 0.75rem 0 0 0; letter-spacing: 0.05em;'>© 2025 | Deep Learning Powered</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def show_overview_page(self):
        """Trang thông tin tổng quan - Implementation đầy đủ"""
        st.title("MARKET OVERVIEW")
        
        # Load tất cả data cho overview
        all_data = {}
        latest_prices = {}
        price_changes_24h = {}
        volumes_24h = {}
        
        for coin in self.coins:
            df = self.load_historical_data(coin)
            if not df.empty:
                all_data[coin] = df
                latest_prices[coin] = df['close'].iloc[-1]
                
                # Calculate 24h change (hoặc 1 day change)
                if len(df) > 1:
                    price_changes_24h[coin] = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                else:
                    price_changes_24h[coin] = 0
                    
                volumes_24h[coin] = df['volume'].iloc[-1] if len(df) > 0 else 0
        
        # === 1. KPIs SECTION ===
        st.markdown("### Market Snapshot")
        
        total_market_cap = sum(latest_prices.values()) * 1e9  # Giả định
        total_volume_24h = sum(volumes_24h.values())
        avg_change = sum(price_changes_24h.values()) / len(price_changes_24h) if price_changes_24h else 0
        
        # Find top gainer and loser
        top_gainer = max(price_changes_24h.items(), key=lambda x: x[1]) if price_changes_24h else ("N/A", 0)
        worst_performer = min(price_changes_24h.items(), key=lambda x: x[1]) if price_changes_24h else ("N/A", 0)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Market Cap", 
                f"${total_market_cap/1e12:.2f}T",
                f"{avg_change:+.2f}%"
            )
        with col2:
            st.metric(
                "24h Volume", 
                f"${total_volume_24h/1e9:.2f}B",
                "+15.2%"
            )
        with col3:
            st.metric(
                "Top Gainer", 
                top_gainer[0].upper(),
                f"+{top_gainer[1]:.2f}%",
                delta_color="normal"
            )
        with col4:
            st.metric(
                "Biggest Loser", 
                worst_performer[0].upper(),
                f"{worst_performer[1]:.2f}%",
                delta_color="inverse"
            )
        
        st.markdown("---")
        
        # === 2. PRICE CHANGE HEATMAP ===
        st.markdown("### Price Change Heatmap")
        
        # Calculate price changes for different periods
        periods = {
            '1D': 1,
            '7D': 7,
            '30D': 30,
            '90D': 90,
            'YTD': 'ytd',
            '1Y': 365
        }
        
        heatmap_data = []
        for coin in self.coins:
            if coin in all_data and not all_data[coin].empty:
                df = all_data[coin]
                row = {'Coin': coin.upper()}
                
                for period_name, period_days in periods.items():
                    if period_days == 'ytd':
                        # Year to date
                        current_year_data = df[df.index.year == df.index[-1].year]
                        if len(current_year_data) > 1:
                            change = ((current_year_data['close'].iloc[-1] - current_year_data['close'].iloc[0]) / 
                                     current_year_data['close'].iloc[0]) * 100
                        else:
                            change = 0
                    else:
                        if len(df) > period_days:
                            change = ((df['close'].iloc[-1] - df['close'].iloc[-period_days]) / 
                                     df['close'].iloc[-period_days]) * 100
                        else:
                            change = 0
                    
                    row[period_name] = change
                
                heatmap_data.append(row)
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            
            # Create styled dataframe
            def color_negative_red(val):
                if isinstance(val, (int, float)):
                    if val < -10:
                        return 'background-color: #ff6b6b; color: white'
                    elif val < 0:
                        return 'background-color: #ffa07a; color: white'
                    elif val < 5:
                        return 'background-color: #fff3cd; color: black'
                    elif val < 15:
                        return 'background-color: #90ee90; color: black'
                    else:
                        return 'background-color: #00ff88; color: black'
                return ''
            
            styled_df = heatmap_df.style.applymap(color_negative_red, subset=list(periods.keys()))
            styled_df = styled_df.format({col: '{:+.2f}%' for col in periods.keys()})
            
            st.dataframe(styled_df, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # === 3. RANKINGS ===
        st.markdown("### Rankings & Leaderboards")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Top Movers (24H)")
            top_movers = sorted(price_changes_24h.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for idx, (coin, change) in enumerate(top_movers, 1):
                st.markdown(f"""
                    <div style='background: rgba(0, 255, 136, 0.1); 
                                padding: 0.75rem; 
                                border-radius: 8px; 
                                margin-bottom: 0.5rem;
                                border-left: 3px solid #00ff88;'>
                        <span style='font-weight: bold; color: #00ff88;'>{idx}. {coin.upper()}</span>
                        <span style='float: right; color: #00ff88; font-weight: bold;'>+{change:.2f}% ↗</span>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Worst Performers")
            worst_performers = sorted(price_changes_24h.items(), key=lambda x: x[1])[:5]
            
            for idx, (coin, change) in enumerate(worst_performers, 1):
                st.markdown(f"""
                    <div style='background: rgba(255, 107, 107, 0.1); 
                                padding: 0.75rem; 
                                border-radius: 8px; 
                                margin-bottom: 0.5rem;
                                border-left: 3px solid #ff6b6b;'>
                        <span style='font-weight: bold; color: #ff6b6b;'>{idx}. {coin.upper()}</span>
                        <span style='float: right; color: #ff6b6b; font-weight: bold;'>{change:.2f}% ↘</span>
                    </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("#### Volume Leaders")
            volume_leaders = sorted(volumes_24h.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for idx, (coin, volume) in enumerate(volume_leaders, 1):
                st.markdown(f"""
                    <div style='background: rgba(99, 102, 241, 0.1); 
                                padding: 0.75rem; 
                                border-radius: 8px; 
                                margin-bottom: 0.5rem;
                                border-left: 3px solid #6366f1;'>
                        <span style='font-weight: bold; color: #a5b4fc;'>{idx}. {coin.upper()}</span>
                        <span style='float: right; color: #a5b4fc; font-weight: bold;'>${volume/1e9:.2f}B</span>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # === 4. MARKET OVERVIEW CHARTS ===
        st.markdown("### � Market Trends")
        
        # Total Market Volume (Stacked)
        if all_data:
            # Get common date range
            min_date = max([df.index.min() for df in all_data.values()])
            max_date = min([df.index.max() for df in all_data.values()])
            
            # Resample to daily and align all coins
            # Tách thành 2 tabs cho dễ nhìn
            tab1, tab2 = st.tabs(["Volume Analysis", "Price Performance"])
            
            with tab1:
                st.markdown("#### Daily Volume Comparison by Coin")
                st.info("Hover over bars to see detailed volume for each coin on specific dates")
                
                # Sử dụng màu sắc RIÊNG BIỆT rõ ràng cho mỗi coin
                coin_colors = {
                    'bitcoin': '#FF9500',      # Orange
                    'ethereum': '#627EEA',     # Blue
                    'litecoin': '#B8B8B8',     # Silver
                    'binancecoin': '#F3BA2F',  # Yellow
                    'cardano': '#0033AD',      # Dark Blue
                    'solana': '#00FFA3',       # Bright Green
                    'pancakeswap': '#D1884F',  # Brown
                    'axieinfinity': '#0055D5', # Light Blue
                    'thesandbox': '#00D4FF'    # Cyan
                }
                
                # Grouped bar chart - dễ nhìn hơn stacked
                fig_volume = go.Figure()
                
                # Lấy 30 ngày gần nhất để dễ nhìn
                recent_days = 30
                
                for coin in self.coins:
                    if coin in all_data:
                        df = all_data[coin]
                        df_filtered = df[(df.index >= min_date) & (df.index <= max_date)]
                        
                        if not df_filtered.empty:
                            # Lấy N ngày gần nhất
                            df_recent = df_filtered.tail(recent_days)
                            
                            fig_volume.add_trace(go.Bar(
                                x=df_recent.index,
                                y=df_recent['volume'],
                                name=coin.capitalize(),
                                marker_color=coin_colors.get(coin, '#667eea'),
                                hovertemplate='<b>%{fullData.name}</b><br>' +
                                            'Volume: %{y:,.0f}<br>' +
                                            '<extra></extra>'
                            ))
            
                fig_volume.update_layout(
                    title={
                        'text': f'� Daily Volume - Last {recent_days} Days',
                        'font': {'size': 18, 'color': 'white'}
                    },
                    xaxis_title="Date",
                    yaxis_title="Volume (USD)",
                    height=550,
                    barmode='group',  # Grouped bars - dễ so sánh hơn
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'},
                    xaxis={
                        'gridcolor': 'rgba(102, 126, 234, 0.2)',
                        'showgrid': True,
                        'tickangle': 0,
                        'tickformat': '%Y-%m-%d'
                    },
                    yaxis={
                        'gridcolor': 'rgba(102, 126, 234, 0.2)',
                        'showgrid': True,
                        'tickformat': ',.0f'
                    },
                    legend={
                        'orientation': 'h',
                        'yanchor': 'top',
                        'y': -0.25,
                        'xanchor': 'center',
                        'x': 0.5,
                        'bgcolor': 'rgba(0,0,0,0.8)',
                        'bordercolor': 'rgba(102, 126, 234, 0.3)',
                        'borderwidth': 1,
                        'font': {'size': 11}
                    },
                    margin=dict(b=150, t=80)  # More space for legend and title
                )
            
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # Thêm volume statistics
                st.markdown("#### Volume Statistics")
                vol_stats_cols = st.columns(3)
                
                total_volumes = {coin: all_data[coin]['volume'].sum() for coin in all_data}
                avg_volumes = {coin: all_data[coin]['volume'].mean() for coin in all_data}
                
                with vol_stats_cols[0]:
                    top_volume_coin = max(total_volumes.items(), key=lambda x: x[1])
                    st.metric(
                        "Highest Total Volume",
                        top_volume_coin[0].capitalize(),
                        f"{top_volume_coin[1]:,.0f}"
                    )
                
                with vol_stats_cols[1]:
                    avg_total = sum(avg_volumes.values())
                    st.metric(
                        "Market Avg Volume",
                        f"{avg_total:,.0f}",
                        "Per Coin"
                    )
                
                with vol_stats_cols[2]:
                    latest_total_vol = sum([all_data[coin]['volume'].iloc[-1] for coin in all_data])
                    st.metric(
                        "Latest Day Total Volume",
                        f"{latest_total_vol:,.0f}",
                        f"{((latest_total_vol - avg_total * len(all_data)) / (avg_total * len(all_data)) * 100):.2f}%"
                    )
            
            with tab2:
                # Price comparison (Normalized to 100)
                st.markdown("#### Normalized Price Performance (Base = 100)")
                st.info("All prices start at 100 for easy comparison of % change over time. Hover to see all coins on same date.")
            
                fig_normalized = go.Figure()
                
                # Sử dụng màu sắc GIỐNG với Volume chart
                coin_colors = {
                    'bitcoin': '#FF9500',      # Orange
                    'ethereum': '#627EEA',     # Blue
                    'litecoin': '#B8B8B8',     # Silver
                    'binancecoin': '#F3BA2F',  # Yellow
                    'cardano': '#0033AD',      # Dark Blue
                    'solana': '#00FFA3',       # Bright Green
                    'pancakeswap': '#D1884F',  # Brown
                    'axieinfinity': '#0055D5', # Light Blue
                    'thesandbox': '#00D4FF'    # Cyan
                }
                
                for coin in self.coins:
                    if coin in all_data:
                        df = all_data[coin]
                        df_filtered = df[(df.index >= min_date) & (df.index <= max_date)]
                        
                        if not df_filtered.empty and len(df_filtered) > 0:
                            # Normalize to 100 at start
                            normalized_prices = (df_filtered['close'] / df_filtered['close'].iloc[0]) * 100
                            
                            fig_normalized.add_trace(go.Scatter(
                                x=df_filtered.index,
                                y=normalized_prices,
                                name=coin.capitalize(),
                                mode='lines',
                                line={
                                    'width': 2.5,
                                    'color': coin_colors.get(coin, '#667eea')
                                },
                                # CHỈ hiển thị tên coin và giá index, KHÔNG hiển thị ngày ở từng coin
                                hovertemplate='%{fullData.name}: %{y:.2f}<extra></extra>'
                            ))
                
                # Add base line at 100
                fig_normalized.add_hline(
                    y=100, 
                    line_dash="dash", 
                    line_color="rgba(255,255,255,0.3)",
                    annotation_text="Base (100)",
                    annotation_position="right"
                )
            
                fig_normalized.update_layout(
                    title={
                        'text': 'Normalized Price Performance (All Coins)',
                        'font': {'size': 18, 'color': 'white'}
                    },
                    xaxis_title="Date",
                    yaxis_title="Index (Base = 100)",
                    height=500,
                    hovermode='x unified',  # Hiển thị tất cả coins cùng một lúc khi hover
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'},
                    xaxis={
                        'gridcolor': 'rgba(102, 126, 234, 0.2)',
                        'showgrid': True
                    },
                    yaxis={
                        'gridcolor': 'rgba(102, 126, 234, 0.2)',
                        'showgrid': True
                    },
                    legend={
                        'orientation': 'v',
                        'yanchor': 'top',
                        'y': 0.99,
                        'xanchor': 'left',
                        'x': 0.01,
                        'bgcolor': 'rgba(0,0,0,0.7)',
                        'bordercolor': 'rgba(102, 126, 234, 0.3)',
                        'borderwidth': 1
                    },
                    # Thêm hover label style để hiển thị date ở trên cùng
                    hoverlabel=dict(
                        bgcolor='rgba(0,0,0,0.8)',
                        font_size=13,
                        font_family="Inter",
                        font_color='white',
                        bordercolor='rgba(102, 126, 234, 0.5)'
                    )
                )
            
                st.plotly_chart(fig_normalized, use_container_width=True)
                
                # Thêm performance statistics
                st.markdown("#### Performance Statistics")
                perf_cols = st.columns(3)
                
                performances = {}
                for coin in all_data:
                    df = all_data[coin]
                    df_filtered = df[(df.index >= min_date) & (df.index <= max_date)]
                    if not df_filtered.empty and len(df_filtered) > 0:
                        perf = ((df_filtered['close'].iloc[-1] - df_filtered['close'].iloc[0]) / df_filtered['close'].iloc[0]) * 100
                        performances[coin] = perf
                
                if performances:
                    with perf_cols[0]:
                        best_perf = max(performances.items(), key=lambda x: x[1])
                        st.metric(
                            "Best Performer",
                            best_perf[0].capitalize(),
                            f"+{best_perf[1]:.2f}%"
                        )
                    
                    with perf_cols[1]:
                        worst_perf = min(performances.items(), key=lambda x: x[1])
                        st.metric(
                            "Worst Performer",
                            worst_perf[0].capitalize(),
                            f"{worst_perf[1]:.2f}%"
                        )
                    
                    with perf_cols[2]:
                        avg_perf = sum(performances.values()) / len(performances)
                        st.metric(
                            "Market Average",
                            "All Coins",
                            f"{avg_perf:+.2f}%"
                        )
                        st.metric(
                            "Market Average",
                            "All Coins",
                            f"{avg_perf:+.2f}%"
                        )
        
        st.markdown("---")
        
        # === 5. MARKET CAPITALIZATION & TRADING ACTIVITY ===
        st.markdown("### Market Capitalization & Trading Activity")
        st.info("Market share and trading volume distribution across cryptocurrencies based on latest prices and volumes.")
        
        if all_data and latest_prices:
            # Calculate market cap estimates (price * volume as proxy)
            market_caps = {}
            for coin in self.coins:
                if coin in all_data and coin in latest_prices:
                    # Simple market cap estimate: current price * 24h volume
                    df = all_data[coin]
                    latest_vol = df['volume'].iloc[-1]
                    market_caps[coin] = latest_prices[coin] * latest_vol
            
            # Create 2 columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Market Cap Pie Chart
                fig_pie = go.Figure()
                
                coins_list = list(market_caps.keys())
                caps_list = list(market_caps.values())
                
                # Màu sắc nhất quán
                coin_colors_list = [
                    '#FF9500',      # Bitcoin - Orange
                    '#627EEA',      # Ethereum - Blue
                    '#B8B8B8',      # Litecoin - Silver
                    '#F3BA2F',      # Binancecoin - Yellow
                    '#0033AD',      # Cardano - Dark Blue
                    '#00FFA3',      # Solana - Bright Green
                    '#D1884F',      # Pancakeswap - Brown
                    '#0055D5',      # Axieinfinity - Light Blue
                    '#00D4FF'       # Thesandbox - Cyan
                ]
                
                fig_pie.add_trace(go.Pie(
                    labels=[c.capitalize() for c in coins_list],
                    values=caps_list,
                    marker=dict(colors=coin_colors_list[:len(coins_list)]),
                    hole=0.4,  # Donut chart
                    textinfo='none',  # Bỏ text labels
                    hovertemplate='<b>%{label}</b><br>' +
                                'Market Cap: $%{value:,.0f}<br>' +
                                'Share: %{percent}<br>' +
                                '<extra></extra>'
                ))
                
                fig_pie.update_layout(
                    title={
                        'text': '🥧 Market Cap Distribution',
                        'font': {'size': 16, 'color': 'white'},
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'},
                    showlegend=True,
                    legend={
                        'orientation': 'v',
                        'yanchor': 'middle',
                        'y': 0.5,
                        'xanchor': 'left',
                        'x': 1.05,
                        'bgcolor': 'rgba(0,0,0,0.5)'
                    }
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Total Trading Volume Bar Chart
                fig_vol_bar = go.Figure()
                
                total_volumes = {}
                for coin in self.coins:
                    if coin in all_data:
                        total_volumes[coin] = all_data[coin]['volume'].sum()
                
                sorted_vols = sorted(total_volumes.items(), key=lambda x: x[1], reverse=True)
                coins_sorted = [c.capitalize() for c, v in sorted_vols]
                vols_sorted = [v for c, v in sorted_vols]
                
                # Gradient colors from high to low
                colors_gradient = ['#00FFA3', '#4ECDC4', '#45B7D1', '#667eea', 
                                 '#764ba2', '#F3BA2F', '#FF9500', '#FF6B6B', '#D1884F']
                
                fig_vol_bar.add_trace(go.Bar(
                    x=coins_sorted,
                    y=vols_sorted,
                    marker_color=colors_gradient[:len(coins_sorted)],
                    text=None,  # Bỏ text labels
                    hovertemplate='<b>%{x}</b><br>' +
                                'Total Volume: $%{y:,.0f}<br>' +
                                '<extra></extra>'
                ))
                
                fig_vol_bar.update_layout(
                    title={
                        'text': 'Total Trading Volume (All Time)',
                        'font': {'size': 16, 'color': 'white'}
                    },
                    xaxis_title="Cryptocurrency",
                    yaxis_title="Volume (USD)",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'},
                    xaxis={
                        'gridcolor': 'rgba(102, 126, 234, 0.2)',
                        'tickangle': -45
                    },
                    yaxis={
                        'gridcolor': 'rgba(102, 126, 234, 0.2)',
                        'showgrid': True,
                        'tickformat': ',.0f'
                    },
                    showlegend=False
                )
                
                st.plotly_chart(fig_vol_bar, use_container_width=True)
            
            # Market Statistics Cards
            st.markdown("#### Market Statistics")
            market_cols = st.columns(4)
            
            # Highest Market Cap
            if market_caps:
                top_cap = max(market_caps.items(), key=lambda x: x[1])
                with market_cols[0]:
                    st.metric(
                        "Largest Market Cap",
                        top_cap[0].capitalize(),
                        f"${top_cap[1]/1e9:.2f}B" if top_cap[1] > 1e9 else f"${top_cap[1]/1e6:.0f}M"
                    )
            
            # Highest Trading Volume
            if total_volumes:
                top_vol = max(total_volumes.items(), key=lambda x: x[1])
                with market_cols[1]:
                    st.metric(
                        "Most Traded",
                        top_vol[0].capitalize(),
                        f"${top_vol[1]/1e9:.2f}B" if top_vol[1] > 1e9 else f"${top_vol[1]/1e6:.0f}M"
                    )
            
            # Average Volume
            if total_volumes:
                avg_vol = sum(total_volumes.values()) / len(total_volumes)
                with market_cols[2]:
                    st.metric(
                        "Average Volume",
                        "All Coins",
                        f"${avg_vol/1e9:.2f}B" if avg_vol > 1e9 else f"${avg_vol/1e6:.0f}M"
                    )
            
            # Total Market Size
            total_market = sum(market_caps.values())
            with market_cols[3]:
                st.metric(
                    "Total Market Size",
                    "Estimated",
                    f"${total_market/1e9:.2f}B" if total_market > 1e9 else f"${total_market/1e6:.0f}M"
                )
        else:
            st.warning("No market data available.")
    
    def show_price_analysis_page(self, selected_coin: str):
        """Trang phân tích giá chi tiết - sẽ implement sau"""
        st.title(f"💹 PHÂN TÍCH GIÁ CHI TIẾT - {selected_coin.upper()}")("� Comparing prediction accuracy across all cryptocurrency models. Lower MAE/RMSE and higher R² indicate better performance.")
        
        # Load model performance metrics
        model_metrics = []
        for coin in self.coins:
            results = self.load_results(coin)
            if results and 'evaluation' in results:
                eval_metrics = results['evaluation']
                model_metrics.append({
                    'coin': coin.capitalize(),
                    'mae': eval_metrics.get('mae', 0),
                    'rmse': eval_metrics.get('rmse', 0),
                    'r2': eval_metrics.get('r2_score', 0),
                    'mape': eval_metrics.get('mape', 0)
                })
        
        if model_metrics:
            # Create 2 columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                # MAE & RMSE Comparison
                fig_errors = go.Figure()
                
                coins = [m['coin'] for m in model_metrics]
                mae_values = [m['mae'] for m in model_metrics]
                rmse_values = [m['rmse'] for m in model_metrics]
                
                fig_errors.add_trace(go.Bar(
                    x=coins,
                    y=mae_values,
                    name='MAE (Mean Absolute Error)',
                    marker_color='#FF6B6B',
                    text=[f'{v:.2f}' for v in mae_values],
                    textposition='outside'
                ))
                
                fig_errors.add_trace(go.Bar(
                    x=coins,
                    y=rmse_values,
                    name='RMSE (Root Mean Squared Error)',
                    marker_color='#4ECDC4',
                    text=[f'{v:.2f}' for v in rmse_values],
                    textposition='outside'
                ))
                
                fig_errors.update_layout(
                    title={
                        'text': 'Error Metrics (Lower is Better)',
                        'font': {'size': 16, 'color': 'white'}
                    },
                    xaxis_title="Cryptocurrency",
                    yaxis_title="Error Value",
                    height=400,
                    barmode='group',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'},
                    xaxis={
                        'gridcolor': 'rgba(102, 126, 234, 0.2)',
                        'tickangle': -45
                    },
                    yaxis={
                        'gridcolor': 'rgba(102, 126, 234, 0.2)',
                        'showgrid': True
                    },
                    legend={
                        'orientation': 'h',
                        'yanchor': 'bottom',
                        'y': 1.02,
                        'xanchor': 'right',
                        'x': 1,
                        'bgcolor': 'rgba(0,0,0,0.5)'
                    }
                )
                
                st.plotly_chart(fig_errors, use_container_width=True)
            
            with col2:
                # R² Score Comparison
                fig_r2 = go.Figure()
                
                r2_values = [m['r2'] * 100 for m in model_metrics]  # Convert to percentage
                colors = ['#00FFA3' if r2 >= 80 else '#F3BA2F' if r2 >= 60 else '#FF6B6B' for r2 in r2_values]
                
                fig_r2.add_trace(go.Bar(
                    x=coins,
                    y=r2_values,
                    name='R² Score',
                    marker_color=colors,
                    text=[f'{v:.1f}%' for v in r2_values],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>R² Score: %{y:.2f}%<extra></extra>'
                ))
                
                # Add reference lines
                fig_r2.add_hline(
                    y=80, 
                    line_dash="dash", 
                    line_color="rgba(0, 255, 163, 0.5)",
                    annotation_text="Excellent (80%)",
                    annotation_position="right"
                )
                
                fig_r2.add_hline(
                    y=60, 
                    line_dash="dash", 
                    line_color="rgba(243, 186, 47, 0.5)",
                    annotation_text="Good (60%)",
                    annotation_position="right"
                )
                
                fig_r2.update_layout(
                    title={
                        'text': 'R² Score - Accuracy (Higher is Better)',
                        'font': {'size': 16, 'color': 'white'}
                    },
                    xaxis_title="Cryptocurrency",
                    yaxis_title="R² Score (%)",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'},
                    xaxis={
                        'gridcolor': 'rgba(102, 126, 234, 0.2)',
                        'tickangle': -45
                    },
                    yaxis={
                        'gridcolor': 'rgba(102, 126, 234, 0.2)',
                        'showgrid': True,
                        'range': [0, 105]
                    },
                    showlegend=False
                )
                
                st.plotly_chart(fig_r2, use_container_width=True)
            
            # Performance Summary Cards
            st.markdown("#### Best Performing Models")
            perf_cols = st.columns(4)
            
            # Best R² Score
            best_r2 = max(model_metrics, key=lambda x: x['r2'])
            with perf_cols[0]:
                st.metric(
                    "🥇 Highest Accuracy",
                    best_r2['coin'],
                    f"R²: {best_r2['r2']*100:.1f}%"
                )
            
            # Lowest MAE
            best_mae = min(model_metrics, key=lambda x: x['mae'])
            with perf_cols[1]:
                st.metric(
                    "Lowest MAE",
                    best_mae['coin'],
                    f"{best_mae['mae']:.2f}"
                )
            
            # Lowest RMSE
            best_rmse = min(model_metrics, key=lambda x: x['rmse'])
            with perf_cols[2]:
                st.metric(
                    "Lowest RMSE",
                    best_rmse['coin'],
                    f"{best_rmse['rmse']:.2f}"
                )
            
            # Average Performance
            avg_r2 = sum([m['r2'] for m in model_metrics]) / len(model_metrics) * 100
            with perf_cols[3]:
                st.metric(
                    "Average R²",
                    "All Models",
                    f"{avg_r2:.1f}%"
                )
        else:
            st.warning("No market data available.")
    
    def show_price_analysis_page(self, selected_coin: str):
        """Trang phân tích giá chi tiết - Implementation đầy đủ"""
        st.title(f"PRICE ANALYSIS - {selected_coin.upper()}")
        
        # Load historical data
        hist_data = self.load_historical_data(selected_coin)
        
        if hist_data.empty:
            st.error(f"Không tìm thấy dữ liệu cho {selected_coin.upper()}")
            st.info("Vui lòng kiểm tra thư mục data/raw/train")
            return
        
        # === 1. KEY METRICS OVERVIEW ===
        st.markdown("### Key Metrics Overview")
        
        # Calculate key metrics
        current_price = hist_data['close'].iloc[-1]
        prev_price = hist_data['close'].iloc[-2] if len(hist_data) > 1 else current_price
        price_change = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
        
        high_24h = hist_data['high'].iloc[-1]
        low_24h = hist_data['low'].iloc[-1]
        volume_24h = hist_data['volume'].iloc[-1]
        
        # Calculate additional metrics
        avg_price_7d = hist_data['close'].tail(7).mean()
        avg_price_30d = hist_data['close'].tail(30).mean()
        volatility = hist_data['close'].pct_change().std() * 100
        
        # Get Market Cap from data (real data from CSV)
        current_market_cap = hist_data['market_cap'].iloc[-1] if 'market_cap' in hist_data.columns else (current_price * volume_24h)
        prev_market_cap = hist_data['market_cap'].iloc[-2] if 'market_cap' in hist_data.columns and len(hist_data) > 1 else current_market_cap
        market_cap_change = ((current_market_cap - prev_market_cap) / prev_market_cap * 100) if prev_market_cap != 0 else 0
        
        # Display metrics in cards - Row 1
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Current Price",
                f"${current_price:,.2f}",
                f"{price_change:+.2f}%"
            )
        
        with col2:
            st.metric(
                "24h High",
                f"${high_24h:,.2f}",
                delta=f"{((high_24h - current_price) / current_price * 100):+.2f}%"
            )
        
        with col3:
            st.metric(
                "24h Low",
                f"${low_24h:,.2f}",
                delta=f"{((low_24h - current_price) / current_price * 100):+.2f}%"
            )
        
        with col4:
            st.metric(
                "24h Volume",
                f"${volume_24h/1e9:.2f}B" if volume_24h > 1e9 else f"${volume_24h/1e6:.0f}M"
            )
        
        with col5:
            st.metric(
                "Volatility",
                f"{volatility:.2f}%"
            )
        
        # Display metrics in cards - Row 2 (Market Cap & More)
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Market Cap",
                f"${current_market_cap/1e12:.2f}T" if current_market_cap > 1e12 else f"${current_market_cap/1e9:.2f}B",
                f"{market_cap_change:+.2f}%"
            )
        
        with col2:
            price_range_24h = high_24h - low_24h
            st.metric(
                "24h Range",
                f"${price_range_24h:,.2f}",
                f"{(price_range_24h/low_24h*100):.2f}%"
            )
        
        with col3:
            st.metric(
                "7-Day Avg Price",
                f"${avg_price_7d:,.2f}",
                f"{((current_price - avg_price_7d)/avg_price_7d*100):+.2f}%"
            )
        
        with col4:
            st.metric(
                "30-Day Avg Price",
                f"${avg_price_30d:,.2f}",
                f"{((current_price - avg_price_30d)/avg_price_30d*100):+.2f}%"
            )
        
        with col5:
            # Calculate ATH (All-Time High) from available data
            ath_price = hist_data['high'].max()
            distance_from_ath = ((current_price - ath_price) / ath_price * 100)
            st.metric(
                "All-Time High",
                f"${ath_price:,.2f}",
                f"{distance_from_ath:+.2f}%"
            )
        
        st.markdown("---")
        
        # === 1.5 MARKET CAP TREND ===
        st.markdown("### Market Capitalization Trend")
        st.info("Market cap trend shows the actual market value over time from real data")
        
        # Use real market cap data from CSV
        if 'market_cap' in hist_data.columns:
            # Time range selector for Market Cap
            col1, col2 = st.columns([3, 1])
            
            with col1:
                mcap_time_range = st.selectbox(
                    "Select Time Range for Market Cap",
                    ["30 Days", "90 Days", "6 Months", "1 Year", "All Time"],
                    index=2,
                    key="mcap_time_range"
                )
            
            with col2:
                st.metric("Data Points", len(hist_data))
            
            # Filter data based on time range
            mcap_range_map = {
                "30 Days": 30,
                "90 Days": 90,
                "6 Months": 180,
                "1 Year": 365,
                "All Time": len(hist_data)
            }
            
            mcap_days = mcap_range_map[mcap_time_range]
            mcap_filtered_data = hist_data.tail(mcap_days)
            
            # Create market cap chart
            fig_mcap = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.6, 0.4],
                subplot_titles=(f'{selected_coin.upper()} Price', 'Market Cap')
            )
            
            # Price chart
            fig_mcap.add_trace(
                go.Scatter(
                    x=mcap_filtered_data.index,
                    y=mcap_filtered_data['close'],
                    name='Price',
                    line=dict(color='#667eea', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.2)'
                ),
                row=1, col=1
            )
            
            # Market cap chart (using real data)
            fig_mcap.add_trace(
                go.Scatter(
                    x=mcap_filtered_data.index,
                    y=mcap_filtered_data['market_cap'],
                    name='Market Cap',
                    line=dict(color='#00ff88', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 136, 0.2)'
                ),
                row=2, col=1
            )
            
            fig_mcap.update_layout(
                height=600,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                showlegend=True,
                title={
                    'text': f'{selected_coin.upper()} - {mcap_time_range}',
                    'font': {'size': 18, 'color': 'white'}
                }
            )
            
            fig_mcap.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_mcap.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)', title_text="Price (USD)", row=1, col=1)
            fig_mcap.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)', title_text="Market Cap (USD)", row=2, col=1)
            
            st.plotly_chart(fig_mcap, use_container_width=True)
            
            # Market Cap Statistics (based on filtered data)
            col1, col2, col3, col4 = st.columns(4)
            
            filtered_current_mcap = mcap_filtered_data['market_cap'].iloc[-1]
            filtered_avg_mcap = mcap_filtered_data['market_cap'].mean()
            filtered_max_mcap = mcap_filtered_data['market_cap'].max()
            filtered_min_mcap = mcap_filtered_data['market_cap'].min()
            
            with col1:
                st.metric(
                    "Current Market Cap",
                    f"${filtered_current_mcap/1e12:.2f}T" if filtered_current_mcap > 1e12 else f"${filtered_current_mcap/1e9:.2f}B"
                )
            
            with col2:
                st.metric(
                    "Average Market Cap",
                    f"${filtered_avg_mcap/1e12:.2f}T" if filtered_avg_mcap > 1e12 else f"${filtered_avg_mcap/1e9:.2f}B"
                )
            
            with col3:
                st.metric(
                    "Highest Market Cap",
                    f"${filtered_max_mcap/1e12:.2f}T" if filtered_max_mcap > 1e12 else f"${filtered_max_mcap/1e9:.2f}B"
                )
            
            with col4:
                mcap_change_vs_avg = ((filtered_current_mcap - filtered_avg_mcap) / filtered_avg_mcap * 100) if filtered_avg_mcap > 0 else 0
                st.metric(
                    "vs Average",
                    f"{mcap_change_vs_avg:+.2f}%",
                    "Above" if mcap_change_vs_avg > 0 else "Below"
                )
            
            # Additional Market Cap Analysis
            st.markdown("#### 📊 Market Cap Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Market Cap Volatility
                mcap_volatility = mcap_filtered_data['market_cap'].pct_change().std() * 100
                mcap_range = filtered_max_mcap - filtered_min_mcap
                mcap_range_pct = (mcap_range / filtered_min_mcap * 100) if filtered_min_mcap > 0 else 0
                
                st.markdown("**💹 Market Cap Volatility**")
                st.write(f"Volatility: {mcap_volatility:.2f}%")
                st.write(f"Range: ${mcap_range/1e9:.2f}B ({mcap_range_pct:.2f}%)")
                st.write(f"Min: ${filtered_min_mcap/1e9:.2f}B")
            
            with col2:
                # Market Cap Growth
                if len(mcap_filtered_data) > 1:
                    first_mcap = mcap_filtered_data['market_cap'].iloc[0]
                    last_mcap = mcap_filtered_data['market_cap'].iloc[-1]
                    mcap_growth = ((last_mcap - first_mcap) / first_mcap * 100) if first_mcap > 0 else 0
                    
                    st.markdown("**📈 Market Cap Growth**")
                    st.write(f"Period Growth: {mcap_growth:+.2f}%")
                    st.write(f"Start: ${first_mcap/1e9:.2f}B")
                    st.write(f"End: ${last_mcap/1e9:.2f}B")
                    
                    if mcap_growth > 0:
                        st.success(f"✅ Market cap increased by ${(last_mcap - first_mcap)/1e9:.2f}B")
                    else:
                        st.error(f"Market cap decreased by ${abs(last_mcap - first_mcap)/1e9:.2f}B")
        else:
            st.warning("Market cap data not available in the dataset")
        
        st.markdown("---")

        
        # === 2. PRICE CHART WITH TECHNICAL OVERLAYS ===
        st.markdown("### Price Chart & Volume")
        
        # Time range selector
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            time_range = st.selectbox(
                "Select Time Range",
                ["7 Days", "30 Days", "90 Days", "6 Months", "1 Year", "All Time"],
                index=2
            )
        
        with col2:
            chart_type = st.selectbox(
                "Chart Type",
                ["Candlestick", "Line", "Area"],
                index=0
            )
        
        with col3:
            show_volume = st.checkbox("Show Volume", value=True)
        
        # Filter data based on time range
        range_map = {
            "7 Days": 7,
            "30 Days": 30,
            "90 Days": 90,
            "6 Months": 180,
            "1 Year": 365,
            "All Time": len(hist_data)
        }
        
        days = range_map[time_range]
        filtered_data = hist_data.tail(days)
        
        # Create main price chart
        if show_volume:
            fig_main = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{selected_coin.upper()} Price', 'Volume')
            )
        else:
            fig_main = go.Figure()
        
        # Add price trace based on chart type
        if chart_type == "Candlestick":
            # Create custom hover text for better formatting
            hover_texts = []
            for idx in filtered_data.index:
                row = filtered_data.loc[idx]
                hover_text = (
                    f"Date: {idx.strftime('%Y-%m-%d')}<br>"
                    f"Open: ${row['open']:,.2f}<br>"
                    f"High: ${row['high']:,.2f}<br>"
                    f"Low: ${row['low']:,.2f}<br>"
                    f"Close: ${row['close']:,.2f}"
                )
                hover_texts.append(hover_text)
            
            if show_volume:
                fig_main.add_trace(
                    go.Candlestick(
                        x=filtered_data.index,
                        open=filtered_data['open'],
                        high=filtered_data['high'],
                        low=filtered_data['low'],
                        close=filtered_data['close'],
                        name='Price',
                        text=hover_texts,
                        hoverinfo='text',
                        increasing_line_color='#00ff88',
                        decreasing_line_color='#ff6b6b'
                    ),
                    row=1, col=1
                )
            else:
                fig_main.add_trace(
                    go.Candlestick(
                        x=filtered_data.index,
                        open=filtered_data['open'],
                        high=filtered_data['high'],
                        low=filtered_data['low'],
                        close=filtered_data['close'],
                        name='Price',
                        text=hover_texts,
                        hoverinfo='text',
                        increasing_line_color='#00ff88',
                        decreasing_line_color='#ff6b6b'
                    )
                )
        
        elif chart_type == "Line":
            trace_row = 1 if show_volume else None
            trace_col = 1 if show_volume else None
            
            if show_volume:
                fig_main.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data['close'],
                        name='Close Price',
                        line=dict(color='#667eea', width=2),
                        mode='lines'
                    ),
                    row=1, col=1
                )
            else:
                fig_main.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data['close'],
                        name='Close Price',
                        line=dict(color='#667eea', width=2),
                        mode='lines'
                    )
                )
        
        else:  # Area
            if show_volume:
                fig_main.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data['close'],
                        name='Close Price',
                        fill='tozeroy',
                        line=dict(color='#667eea', width=2),
                        fillcolor='rgba(102, 126, 234, 0.3)'
                    ),
                    row=1, col=1
                )
            else:
                fig_main.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data['close'],
                        name='Close Price',
                        fill='tozeroy',
                        line=dict(color='#667eea', width=2),
                        fillcolor='rgba(102, 126, 234, 0.3)'
                    )
                )
        
        # Add volume bars if enabled
        if show_volume:
            colors = ['#ff6b6b' if filtered_data['close'].iloc[i] < filtered_data['open'].iloc[i] 
                      else '#00ff88' for i in range(len(filtered_data))]
            
            fig_main.add_trace(
                go.Bar(
                    x=filtered_data.index,
                    y=filtered_data['volume'],
                    name='Volume',
                    marker_color=colors,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Update layout
        fig_main.update_layout(
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            title={
                'text': f'{selected_coin.upper()} - {time_range}',
                'font': {'size': 20, 'color': 'white'}
            }
        )
        
        if show_volume:
            fig_main.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)', row=1, col=1)
            fig_main.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)', row=2, col=1)
            fig_main.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)', row=1, col=1)
            fig_main.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)', row=2, col=1)
        else:
            fig_main.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_main.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
        
        st.plotly_chart(fig_main, use_container_width=True)
        
        st.markdown("---")
        
        # === 3. OHLC DETAILED ANALYSIS ===
        st.markdown("### OHLC Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["OHLC Trends", "Distribution", "Daily Range"])
        
        with tab1:
            st.markdown("#### Price Components Over Time")
            
            fig_ohlc = go.Figure()
            
            fig_ohlc.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['open'],
                name='Open',
                line=dict(color='#3b82f6', width=1.5),
                mode='lines'
            ))
            
            fig_ohlc.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['high'],
                name='High',
                line=dict(color='#00ff88', width=1.5),
                mode='lines'
            ))
            
            fig_ohlc.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['low'],
                name='Low',
                line=dict(color='#ff6b6b', width=1.5),
                mode='lines'
            ))
            
            fig_ohlc.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['close'],
                name='Close',
                line=dict(color='#fbbf24', width=2.5),
                mode='lines'
            ))
            
            fig_ohlc.update_layout(
                height=400,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis={'gridcolor': 'rgba(102, 126, 234, 0.2)'},
                yaxis={'gridcolor': 'rgba(102, 126, 234, 0.2)', 'title': 'Price (USD)'}
            )
            
            st.plotly_chart(fig_ohlc, use_container_width=True)
            
            # OHLC Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Open", f"${filtered_data['open'].mean():,.2f}")
            with col2:
                st.metric("Avg High", f"${filtered_data['high'].mean():,.2f}")
            with col3:
                st.metric("Avg Low", f"${filtered_data['low'].mean():,.2f}")
            with col4:
                st.metric("Avg Close", f"${filtered_data['close'].mean():,.2f}")
        
        with tab2:
            st.markdown("#### Price Distribution Analysis")
            
            fig_box = go.Figure()
            
            fig_box.add_trace(go.Box(
                y=filtered_data['open'],
                name='Open',
                marker_color='#3b82f6',
                boxmean='sd'
            ))
            
            fig_box.add_trace(go.Box(
                y=filtered_data['high'],
                name='High',
                marker_color='#00ff88',
                boxmean='sd'
            ))
            
            fig_box.add_trace(go.Box(
                y=filtered_data['low'],
                name='Low',
                marker_color='#ff6b6b',
                boxmean='sd'
            ))
            
            fig_box.add_trace(go.Box(
                y=filtered_data['close'],
                name='Close',
                marker_color='#fbbf24',
                boxmean='sd'
            ))
            
            fig_box.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                yaxis={'gridcolor': 'rgba(102, 126, 234, 0.2)', 'title': 'Price (USD)'},
                showlegend=True
            )
            
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Distribution stats
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Price Range**")
                st.write(f"Max High: ${filtered_data['high'].max():,.2f}")
                st.write(f"Min Low: ${filtered_data['low'].min():,.2f}")
                st.write(f"Range: ${filtered_data['high'].max() - filtered_data['low'].min():,.2f}")
            
            with col2:
                st.markdown("**Volatility Metrics**")
                st.write(f"Std Dev (Close): ${filtered_data['close'].std():,.2f}")
                st.write(f"Variance: ${filtered_data['close'].var():,.2f}")
                st.write(f"Coefficient of Variation: {(filtered_data['close'].std() / filtered_data['close'].mean() * 100):.2f}%")
        
        with tab3:
            st.markdown("#### Daily Price Range Analysis")
            
            # Calculate daily range
            filtered_data_copy = filtered_data.copy()
            filtered_data_copy['range'] = filtered_data_copy['high'] - filtered_data_copy['low']
            filtered_data_copy['range_pct'] = (filtered_data_copy['range'] / filtered_data_copy['low']) * 100
            
            # Range chart
            fig_range = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Daily Price Range (USD)', 'Daily Range (%)'),
                vertical_spacing=0.15
            )
            
            fig_range.add_trace(
                go.Bar(
                    x=filtered_data_copy.index,
                    y=filtered_data_copy['range'],
                    name='Range (USD)',
                    marker_color='#a855f7'
                ),
                row=1, col=1
            )
            
            fig_range.add_trace(
                go.Bar(
                    x=filtered_data_copy.index,
                    y=filtered_data_copy['range_pct'],
                    name='Range (%)',
                    marker_color='#ec4899'
                ),
                row=2, col=1
            )
            
            fig_range.update_layout(
                height=500,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            
            fig_range.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_range.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_range, use_container_width=True)
            
            # Range statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Range", f"${filtered_data_copy['range'].mean():,.2f}")
            with col2:
                st.metric("Max Range", f"${filtered_data_copy['range'].max():,.2f}")
            with col3:
                st.metric("Avg Range %", f"{filtered_data_copy['range_pct'].mean():.2f}%")
        
        st.markdown("---")
        
        # === 4. VOLUME ANALYSIS ===
        st.markdown("### Volume Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Volume Over Time")
            
            # Calculate volume moving average
            filtered_data_copy['vol_ma_7'] = filtered_data_copy['volume'].rolling(window=7).mean()
            filtered_data_copy['vol_ma_30'] = filtered_data_copy['volume'].rolling(window=30).mean()
            
            fig_vol = go.Figure()
            
            fig_vol.add_trace(go.Bar(
                x=filtered_data_copy.index,
                y=filtered_data_copy['volume'],
                name='Volume',
                marker_color='rgba(99, 102, 241, 0.6)'
            ))
            
            fig_vol.add_trace(go.Scatter(
                x=filtered_data_copy.index,
                y=filtered_data_copy['vol_ma_7'],
                name='7-Day MA',
                line=dict(color='#00ff88', width=2)
            ))
            
            fig_vol.add_trace(go.Scatter(
                x=filtered_data_copy.index,
                y=filtered_data_copy['vol_ma_30'],
                name='30-Day MA',
                line=dict(color='#ff6b6b', width=2)
            ))
            
            fig_vol.update_layout(
                height=400,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis={'gridcolor': 'rgba(102, 126, 234, 0.2)'},
                yaxis={'gridcolor': 'rgba(102, 126, 234, 0.2)', 'title': 'Volume'}
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col2:
            st.markdown("#### Volume Distribution")
            
            fig_vol_dist = go.Figure()
            
            fig_vol_dist.add_trace(go.Histogram(
                x=filtered_data['volume'],
                nbinsx=40,
                name='Volume Distribution',
                marker_color='#667eea'
            ))
            
            fig_vol_dist.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis={'gridcolor': 'rgba(102, 126, 234, 0.2)', 'title': 'Volume'},
                yaxis={'gridcolor': 'rgba(102, 126, 234, 0.2)', 'title': 'Frequency'}
            )
            
            st.plotly_chart(fig_vol_dist, use_container_width=True)
        
        # Volume statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Volume", f"${filtered_data['volume'].sum()/1e9:.2f}B")
        with col2:
            st.metric("Avg Volume", f"${filtered_data['volume'].mean()/1e6:.0f}M")
        with col3:
            st.metric("Max Volume", f"${filtered_data['volume'].max()/1e6:.0f}M")
        with col4:
            st.metric("Min Volume", f"${filtered_data['volume'].min()/1e6:.0f}M")
        
        st.markdown("---")
        
        # === 5. PRICE VS VOLUME CORRELATION ===
        st.markdown("### 🔗 Price vs Volume Correlation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_corr = go.Figure()
            
            fig_corr.add_trace(go.Scatter(
                x=filtered_data['volume'],
                y=filtered_data['close'],
                mode='markers',
                name='Price vs Volume',
                marker=dict(
                    size=8,
                    color=filtered_data.index.astype('int64') / 10**9,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Time", tickformat='d')
                ),
                hovertemplate='Volume: %{x:,.0f}<br>Price: $%{y:,.2f}<extra></extra>'
            ))
            
            # Add trend line
            import numpy as np
            z = np.polyfit(filtered_data['volume'], filtered_data['close'], 1)
            p = np.poly1d(z)
            
            fig_corr.add_trace(go.Scatter(
                x=filtered_data['volume'],
                y=p(filtered_data['volume']),
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_corr.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis={'gridcolor': 'rgba(102, 126, 234, 0.2)', 'title': 'Volume'},
                yaxis={'gridcolor': 'rgba(102, 126, 234, 0.2)', 'title': 'Close Price (USD)'}
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            # Correlation statistics
            correlation = filtered_data['volume'].corr(filtered_data['close'])
            
            st.markdown("#### Correlation Metrics")
            st.metric("Correlation Coefficient", f"{correlation:.4f}")
            
            if abs(correlation) > 0.7:
                strength = "Strong"
                color = "🟢"
            elif abs(correlation) > 0.4:
                strength = "Moderate"
                color = "🟡"
            else:
                strength = "Weak"
                color = "🔴"
            
            st.info(f"{color} **{strength}** {'positive' if correlation > 0 else 'negative'} correlation")
            
            st.markdown("---")
            
            # Additional stats
            st.markdown("**Statistical Summary**")
            st.write(f"Covariance: {filtered_data['volume'].cov(filtered_data['close']):,.2f}")
            st.write(f"R-squared: {correlation**2:.4f}")
        
        st.markdown("---")
        
        # === 6. RAW DATA TABLE ===
        st.markdown("### Raw Data")
        
        with st.expander("View Historical Data Table", expanded=False):
            # Format data for display
            display_data = filtered_data[['open', 'high', 'low', 'close', 'volume']].copy()
            display_data.index.name = 'Date'
            
            # Add percentage change
            display_data['Change %'] = display_data['close'].pct_change() * 100
            
            # Format columns
            st.dataframe(
                display_data.style.format({
                    'open': '${:,.2f}',
                    'high': '${:,.2f}',
                    'low': '${:,.2f}',
                    'close': '${:,.2f}',
                    'volume': '{:,.0f}',
                    'Change %': '{:+.2f}%'
                }),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = display_data.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{selected_coin}_historical_data.csv",
                mime="text/csv"
            )
    
    def show_technical_indicators_page(self, selected_coin: str):
        """Trang chỉ báo kỹ thuật - Implementation đầy đủ"""
        st.title(f"TECHNICAL INDICATORS - {selected_coin.upper()}")
        
        # Load historical data
        hist_data = self.load_historical_data(selected_coin)
        
        if hist_data.empty:
            st.error(f"Không tìm thấy dữ liệu cho {selected_coin.upper()}")
            st.info("Vui lòng kiểm tra thư mục data/raw/train")
            return
        
        # Calculate technical indicators using FeatureEngineer
        from src.preprocessing.feature_engineering import FeatureEngineer
        
        try:
            feature_engineer = FeatureEngineer()
            df_with_indicators = feature_engineer.add_technical_features(hist_data.copy())
        except Exception as e:
            st.error(f"Lỗi khi tính toán chỉ báo kỹ thuật: {e}")
            return
        
        # === 1. INDICATOR SELECTION ===
        st.markdown("### Select Technical Indicators")
        st.info("Choose which technical indicators you want to analyze")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            show_rsi = st.checkbox("RSI (Relative Strength Index)", value=True)
        with col2:
            show_macd = st.checkbox("MACD", value=True)
        with col3:
            show_bb = st.checkbox("Bollinger Bands", value=True)
        with col4:
            show_sma = st.checkbox("SMA (Simple Moving Average)", value=True)
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            show_roc = st.checkbox("ROC (Rate of Change)", value=True)
        with col6:
            show_volume = st.checkbox("Volume Analysis", value=True)
        with col7:
            show_price = st.checkbox("Price Action", value=True)
        with col8:
            show_summary = st.checkbox("Summary Table", value=False)
        
        st.markdown("---")
        
        # Time range selector
        col1, col2 = st.columns([3, 1])
        
        with col1:
            time_range = st.selectbox(
                "Select Time Range",
                ["30 Days", "90 Days", "6 Months", "1 Year", "All Time"],
                index=1
            )
        
        with col2:
            st.metric("Total Data Points", len(df_with_indicators))
        
        # Filter data based on time range
        range_map = {
            "30 Days": 30,
            "90 Days": 90,
            "6 Months": 180,
            "1 Year": 365,
            "All Time": len(df_with_indicators)
        }
        
        days = range_map[time_range]
        filtered_data = df_with_indicators.tail(days)
        
        st.markdown("---")
        
        # === 2. RSI (RELATIVE STRENGTH INDEX) ===
        if show_rsi and 'rsi' in filtered_data.columns:
            st.markdown("### RSI - Relative Strength Index")
            st.info("RSI measures the speed and magnitude of price changes. Values above 70 indicate overbought conditions, below 30 indicate oversold.")
            
            fig_rsi = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{selected_coin.upper()} Price', 'RSI (7-period)')
            )
            
            # Price chart
            fig_rsi.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['close'],
                    name='Close Price',
                    line=dict(color='#667eea', width=2)
                ),
                row=1, col=1
            )
            
            # RSI chart
            fig_rsi.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['rsi'],
                    name='RSI',
                    line=dict(color='#fbbf24', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(251, 191, 36, 0.2)'
                ),
                row=2, col=1
            )
            
            # Add overbought/oversold lines
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1, annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1, annotation_text="Oversold")
            fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
            
            fig_rsi.update_layout(
                height=600,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    bgcolor='rgba(0,0,0,0.8)',
                    bordercolor='rgba(102, 126, 234, 0.3)',
                    borderwidth=1
                ),
                margin=dict(b=100)
            )
            
            fig_rsi.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_rsi.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # RSI Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            current_rsi = filtered_data['rsi'].iloc[-1]
            avg_rsi = filtered_data['rsi'].mean()
            
            with col1:
                rsi_status = "🔴 Overbought" if current_rsi > 70 else "🟢 Oversold" if current_rsi < 30 else "🟡 Neutral"
                st.metric("Current RSI", f"{current_rsi:.2f}", rsi_status)
            
            with col2:
                st.metric("Average RSI", f"{avg_rsi:.2f}")
            
            with col3:
                overbought_count = len(filtered_data[filtered_data['rsi'] > 70])
                st.metric("Overbought Days", overbought_count)
            
            with col4:
                oversold_count = len(filtered_data[filtered_data['rsi'] < 30])
                st.metric("Oversold Days", oversold_count)
            
            st.markdown("---")
        
        # === 3. MACD (MOVING AVERAGE CONVERGENCE DIVERGENCE) ===
        if show_macd and all(col in filtered_data.columns for col in ['macd', 'macd_signal', 'macd_hist']):
            st.markdown("### MACD - Moving Average Convergence Divergence")
            st.info("MACD shows the relationship between two moving averages. Crossovers signal potential buy/sell opportunities.")
            
            fig_macd = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=(f'{selected_coin.upper()} Price', 'MACD Line & Signal', 'MACD Histogram')
            )
            
            # Price chart
            fig_macd.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['close'],
                    name='Close Price',
                    line=dict(color='#667eea', width=2)
                ),
                row=1, col=1
            )
            
            # MACD line and signal
            fig_macd.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['macd'],
                    name='MACD',
                    line=dict(color='#00ff88', width=2)
                ),
                row=2, col=1
            )
            
            fig_macd.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['macd_signal'],
                    name='Signal',
                    line=dict(color='#ff6b6b', width=2)
                ),
                row=2, col=1
            )
            
            # MACD histogram
            colors = ['#00ff88' if val >= 0 else '#ff6b6b' for val in filtered_data['macd_hist']]
            fig_macd.add_trace(
                go.Bar(
                    x=filtered_data.index,
                    y=filtered_data['macd_hist'],
                    name='Histogram',
                    marker_color=colors
                ),
                row=3, col=1
            )
            
            fig_macd.update_layout(
                height=700,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.12,
                    xanchor="center",
                    x=0.5,
                    bgcolor='rgba(0,0,0,0.8)',
                    bordercolor='rgba(102, 126, 234, 0.3)',
                    borderwidth=1
                ),
                margin=dict(b=100)
            )
            
            fig_macd.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_macd.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_macd, use_container_width=True)
            
            # MACD Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            current_macd = filtered_data['macd'].iloc[-1]
            current_signal = filtered_data['macd_signal'].iloc[-1]
            current_hist = filtered_data['macd_hist'].iloc[-1]
            
            with col1:
                trend = "🟢 Bullish" if current_macd > current_signal else "🔴 Bearish"
                st.metric("Current Trend", trend)
            
            with col2:
                st.metric("MACD Value", f"{current_macd:.4f}")
            
            with col3:
                st.metric("Signal Value", f"{current_signal:.4f}")
            
            with col4:
                hist_trend = "Growing" if current_hist > 0 else "Declining"
                st.metric("Histogram", f"{current_hist:.4f}", hist_trend)
            
            st.markdown("---")
        
        # === 4. BOLLINGER BANDS ===
        if show_bb and all(col in filtered_data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            st.markdown("### Bollinger Bands")
            st.info("Bollinger Bands measure volatility. Price touching upper band may indicate overbought, lower band may indicate oversold.")
            
            fig_bb = go.Figure()
            
            # Upper band
            fig_bb.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['bb_upper'],
                name='Upper Band',
                line=dict(color='#ff6b6b', width=1, dash='dash')
            ))
            
            # Middle band (SMA)
            fig_bb.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['bb_middle'],
                name='Middle Band (SMA)',
                line=dict(color='#fbbf24', width=2)
            ))
            
            # Lower band
            fig_bb.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['bb_lower'],
                name='Lower Band',
                line=dict(color='#00ff88', width=1, dash='dash')
            ))
            
            # Price
            fig_bb.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['close'],
                name='Close Price',
                line=dict(color='#667eea', width=2)
            ))
            
            # Fill between bands
            fig_bb.add_trace(go.Scatter(
                x=filtered_data.index.tolist() + filtered_data.index.tolist()[::-1],
                y=filtered_data['bb_upper'].tolist() + filtered_data['bb_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            ))
            
            fig_bb.update_layout(
                height=500,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                title={
                    'text': f'{selected_coin.upper()} - Bollinger Bands (10-period, 2 std)',
                    'font': {'size': 18, 'color': 'white'}
                },
                xaxis={'gridcolor': 'rgba(102, 126, 234, 0.2)'},
                yaxis={'gridcolor': 'rgba(102, 126, 234, 0.2)', 'title': 'Price (USD)'},
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    bgcolor='rgba(0,0,0,0.8)',
                    bordercolor='rgba(102, 126, 234, 0.3)',
                    borderwidth=1
                ),
                margin=dict(b=100, t=80)
            )
            
            st.plotly_chart(fig_bb, use_container_width=True)
            
            # Bollinger Bands Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = filtered_data['close'].iloc[-1]
            current_upper = filtered_data['bb_upper'].iloc[-1]
            current_lower = filtered_data['bb_lower'].iloc[-1]
            current_middle = filtered_data['bb_middle'].iloc[-1]
            
            bb_width = ((current_upper - current_lower) / current_middle) * 100
            price_position = ((current_price - current_lower) / (current_upper - current_lower)) * 100
            
            with col1:
                position = "🔴 Near Upper" if price_position > 80 else "🟢 Near Lower" if price_position < 20 else "🟡 Middle"
                st.metric("Price Position", position, f"{price_position:.1f}%")
            
            with col2:
                st.metric("Band Width", f"{bb_width:.2f}%")
            
            with col3:
                st.metric("Upper Band", f"${current_upper:,.2f}")
            
            with col4:
                st.metric("Lower Band", f"${current_lower:,.2f}")
            
            st.markdown("---")
        
        # === 5. SMA (SIMPLE MOVING AVERAGES) ===
        if show_sma:
            st.markdown("### SMA - Simple Moving Averages")
            st.info("SMAs smooth out price data to identify trends. Short-term MA crossing above long-term MA signals potential uptrend.")
            
            fig_sma = go.Figure()
            
            # Price
            fig_sma.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['close'],
                name='Close Price',
                line=dict(color='#667eea', width=2)
            ))
            
            # SMAs
            sma_colors = {
                'sma_10': '#00ff88',
                'sma_20': '#fbbf24'
            }
            
            for sma_col in ['sma_10', 'sma_20']:
                if sma_col in filtered_data.columns:
                    period = sma_col.split('_')[1]
                    fig_sma.add_trace(go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data[sma_col],
                        name=f'SMA {period}',
                        line=dict(color=sma_colors.get(sma_col, '#ffffff'), width=2, dash='dash')
                    ))
            
            fig_sma.update_layout(
                height=500,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                title={
                    'text': f'{selected_coin.upper()} - Simple Moving Averages',
                    'font': {'size': 18, 'color': 'white'}
                },
                xaxis={'gridcolor': 'rgba(102, 126, 234, 0.2)'},
                yaxis={'gridcolor': 'rgba(102, 126, 234, 0.2)', 'title': 'Price (USD)'}
            )
            
            st.plotly_chart(fig_sma, use_container_width=True)
            
            # SMA Statistics & Crossover Detection
            col1, col2, col3, col4 = st.columns(4)
            
            if 'sma_10' in filtered_data.columns and 'sma_20' in filtered_data.columns:
                current_sma10 = filtered_data['sma_10'].iloc[-1]
                current_sma20 = filtered_data['sma_20'].iloc[-1]
                prev_sma10 = filtered_data['sma_10'].iloc[-2] if len(filtered_data) > 1 else current_sma10
                prev_sma20 = filtered_data['sma_20'].iloc[-2] if len(filtered_data) > 1 else current_sma20
                
                # Detect crossover
                if prev_sma10 <= prev_sma20 and current_sma10 > current_sma20:
                    crossover = "Golden Cross (Bullish)"
                elif prev_sma10 >= prev_sma20 and current_sma10 < current_sma20:
                    crossover = "Death Cross (Bearish)"
                else:
                    crossover = "No Recent Cross"
                
                with col1:
                    st.metric("Crossover Signal", crossover)
                
                with col2:
                    trend = "Above" if current_price > current_sma20 else "Below"
                    st.metric("Price vs SMA20", trend)
                
                with col3:
                    st.metric("SMA 10", f"${current_sma10:,.2f}")
                
                with col4:
                    st.metric("SMA 20", f"${current_sma20:,.2f}")
            
            st.markdown("---")
        
        # === 6. ROC (RATE OF CHANGE) ===
        if show_roc:
            st.markdown("### ROC - Rate of Change")
            st.info("ROC measures the percentage change in price over a specific period. Positive values indicate upward momentum.")
            
            # Check if ROC columns exist and have valid data
            has_roc_3 = 'roc_3' in filtered_data.columns and filtered_data['roc_3'].notna().sum() > 0
            has_roc_5 = 'roc_5' in filtered_data.columns and filtered_data['roc_5'].notna().sum() > 0
            
            if not has_roc_3 and not has_roc_5:
                st.warning("ROC indicators are not available in the data. This may be due to insufficient data points or the data hasn't been processed with technical indicators yet.")
                st.info("ROC requires at least 5 data points to calculate. Please ensure your data has enough historical records.")
            else:
                fig_roc = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    row_heights=[0.6, 0.4],
                    subplot_titles=(f'{selected_coin.upper()} Price', 'ROC (3 & 5-period)')
                )
            
                # Price chart
                fig_roc.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data['close'],
                        name='Close Price',
                        line=dict(color='#667eea', width=2)
                    ),
                    row=1, col=1
                )
                
                # ROC charts - only add if data exists
                roc_colors = {
                    'roc_3': '#00ff88',
                    'roc_5': '#fbbf24'
                }
                
                roc_added = False
                for roc_col in ['roc_3', 'roc_5']:
                    if roc_col in filtered_data.columns and filtered_data[roc_col].notna().sum() > 0:
                        period = roc_col.split('_')[1]
                        # Remove NaN values for plotting
                        valid_data = filtered_data[[roc_col]].dropna()
                        if len(valid_data) > 0:
                            fig_roc.add_trace(
                                go.Scatter(
                                    x=valid_data.index,
                                    y=valid_data[roc_col],
                                    name=f'ROC {period}',
                                    line=dict(color=roc_colors.get(roc_col, '#ffffff'), width=2)
                                ),
                                row=2, col=1
                            )
                            roc_added = True
                
                if roc_added:
                    # Add zero line
                    fig_roc.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
                    
                    fig_roc.update_layout(
                        height=600,
                        hovermode='x unified',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': 'white'},
                        showlegend=True
                    )
                    
                    fig_roc.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
                    fig_roc.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
                    
                    st.plotly_chart(fig_roc, use_container_width=True)
                    
                    # ROC Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    if has_roc_3:
                        current_roc3 = filtered_data['roc_3'].dropna().iloc[-1] if len(filtered_data['roc_3'].dropna()) > 0 else 0
                        momentum3 = "Positive" if current_roc3 > 0 else "Negative"
                        
                        with col1:
                            st.metric("ROC 3 Momentum", momentum3, f"{current_roc3:.2f}%")
                    
                    if has_roc_5:
                        current_roc5 = filtered_data['roc_5'].dropna().iloc[-1] if len(filtered_data['roc_5'].dropna()) > 0 else 0
                        momentum5 = "Positive" if current_roc5 > 0 else "Negative"
                        
                        with col2:
                            st.metric("ROC 5 Momentum", momentum5, f"{current_roc5:.2f}%")
                    
                    if has_roc_3:
                        avg_roc3 = filtered_data['roc_3'].mean()
                        with col3:
                            st.metric("Avg ROC 3", f"{avg_roc3:.2f}%")
                    
                    if has_roc_5:
                        avg_roc5 = filtered_data['roc_5'].mean()
                        with col4:
                            st.metric("Avg ROC 5", f"{avg_roc5:.2f}%")
                else:
                    st.warning("No valid ROC data available for the selected time range.")
            
            st.markdown("---")
        
        # === 7. VOLUME ANALYSIS ===
        if show_volume:
            st.markdown("### Volume Analysis")
            st.info("Volume confirms price trends. Rising prices with high volume indicate strong trends.")
            
            fig_vol = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.6, 0.4],
                subplot_titles=(f'{selected_coin.upper()} Price', 'Volume with Moving Average')
            )
            
            # Price chart
            fig_vol.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['close'],
                    name='Close Price',
                    line=dict(color='#667eea', width=2)
                ),
                row=1, col=1
            )
            
            # Volume bars
            colors_vol = ['#00ff88' if filtered_data['close'].iloc[i] >= filtered_data['open'].iloc[i] 
                          else '#ff6b6b' for i in range(len(filtered_data))]
            
            fig_vol.add_trace(
                go.Bar(
                    x=filtered_data.index,
                    y=filtered_data['volume'],
                    name='Volume',
                    marker_color=colors_vol
                ),
                row=2, col=1
            )
            
            # Volume MA
            if 'volume_ma' in filtered_data.columns:
                fig_vol.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data['volume_ma'],
                        name='Volume MA (10)',
                        line=dict(color='#fbbf24', width=2)
                    ),
                    row=2, col=1
                )
            
            fig_vol.update_layout(
                height=600,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                showlegend=True
            )
            
            fig_vol.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_vol.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Volume Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            current_volume = filtered_data['volume'].iloc[-1]
            avg_volume = filtered_data['volume'].mean()
            volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 0
            
            with col1:
                vol_status = "High" if volume_ratio > 1.5 else "Low" if volume_ratio < 0.5 else "Normal"
                st.metric("Volume Status", vol_status, f"{volume_ratio:.2f}x avg")
            
            with col2:
                st.metric("Current Volume", f"${current_volume/1e6:.2f}M")
            
            with col3:
                st.metric("Average Volume", f"${avg_volume/1e6:.2f}M")
            
            if 'volume_roc' in filtered_data.columns:
                current_vol_roc = filtered_data['volume_roc'].iloc[-1]
                with col4:
                    st.metric("Volume Change", f"{current_vol_roc:+.2f}%")
            
            st.markdown("---")
        
        # === 8. PRICE ACTION ===
        if show_price:
            st.markdown("### Price Action Analysis")
            st.info("Comprehensive price movement analysis with candlestick patterns.")
            
            # Create candlestick chart
            fig_price = go.Figure()
            
            fig_price.add_trace(go.Candlestick(
                x=filtered_data.index,
                open=filtered_data['open'],
                high=filtered_data['high'],
                low=filtered_data['low'],
                close=filtered_data['close'],
                name='OHLC',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff6b6b'
            ))
            
            fig_price.update_layout(
                height=500,
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                title={
                    'text': f'{selected_coin.upper()} - Candlestick Chart',
                    'font': {'size': 18, 'color': 'white'}
                },
                xaxis={'gridcolor': 'rgba(102, 126, 234, 0.2)'},
                yaxis={'gridcolor': 'rgba(102, 126, 234, 0.2)', 'title': 'Price (USD)'}
            )
            
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Price Action Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate price changes
            price_change_1d = ((filtered_data['close'].iloc[-1] / filtered_data['close'].iloc[-2]) - 1) * 100 if len(filtered_data) > 1 else 0
            price_change_7d = ((filtered_data['close'].iloc[-1] / filtered_data['close'].iloc[-7]) - 1) * 100 if len(filtered_data) > 7 else 0
            
            # Count bullish/bearish candles
            bullish_candles = len(filtered_data[filtered_data['close'] > filtered_data['open']])
            bearish_candles = len(filtered_data[filtered_data['close'] < filtered_data['open']])
            
            with col1:
                st.metric("1-Day Change", f"{price_change_1d:+.2f}%")
            
            with col2:
                st.metric("7-Day Change", f"{price_change_7d:+.2f}%")
            
            with col3:
                st.metric("Bullish Days", bullish_candles, f"{(bullish_candles/len(filtered_data)*100):.1f}%")
            
            with col4:
                st.metric("Bearish Days", bearish_candles, f"{(bearish_candles/len(filtered_data)*100):.1f}%")
            
            st.markdown("---")
        
        # === 9. SUMMARY TABLE ===
        if show_summary:
            st.markdown("### Technical Indicators Summary Table")
            
            # Create summary data
            summary_data = {
                'Indicator': [],
                'Current Value': [],
                'Signal': [],
                'Interpretation': []
            }
            
            # RSI
            if 'rsi' in filtered_data.columns:
                current_rsi = filtered_data['rsi'].iloc[-1]
                summary_data['Indicator'].append('RSI (7)')
                summary_data['Current Value'].append(f"{current_rsi:.2f}")
                if current_rsi > 70:
                    summary_data['Signal'].append('🔴 Overbought')
                    summary_data['Interpretation'].append('Consider selling')
                elif current_rsi < 30:
                    summary_data['Signal'].append('🟢 Oversold')
                    summary_data['Interpretation'].append('Consider buying')
                else:
                    summary_data['Signal'].append('🟡 Neutral')
                    summary_data['Interpretation'].append('Hold position')
            
            # MACD
            if 'macd' in filtered_data.columns and 'macd_signal' in filtered_data.columns:
                current_macd = filtered_data['macd'].iloc[-1]
                current_signal = filtered_data['macd_signal'].iloc[-1]
                summary_data['Indicator'].append('MACD')
                summary_data['Current Value'].append(f"{current_macd:.4f}")
                if current_macd > current_signal:
                    summary_data['Signal'].append('🟢 Bullish')
                    summary_data['Interpretation'].append('Upward momentum')
                else:
                    summary_data['Signal'].append('🔴 Bearish')
                    summary_data['Interpretation'].append('Downward momentum')
            
            # Bollinger Bands
            if all(col in filtered_data.columns for col in ['bb_upper', 'bb_lower', 'close']):
                current_price = filtered_data['close'].iloc[-1]
                current_upper = filtered_data['bb_upper'].iloc[-1]
                current_lower = filtered_data['bb_lower'].iloc[-1]
                bb_position = ((current_price - current_lower) / (current_upper - current_lower)) * 100
                
                summary_data['Indicator'].append('Bollinger Bands')
                summary_data['Current Value'].append(f"{bb_position:.1f}%")
                if bb_position > 80:
                    summary_data['Signal'].append('🔴 Near Upper')
                    summary_data['Interpretation'].append('Potentially overbought')
                elif bb_position < 20:
                    summary_data['Signal'].append('🟢 Near Lower')
                    summary_data['Interpretation'].append('Potentially oversold')
                else:
                    summary_data['Signal'].append('🟡 Middle')
                    summary_data['Interpretation'].append('Normal range')
            
            # SMA Crossover
            if 'sma_10' in filtered_data.columns and 'sma_20' in filtered_data.columns:
                current_sma10 = filtered_data['sma_10'].iloc[-1]
                current_sma20 = filtered_data['sma_20'].iloc[-1]
                
                summary_data['Indicator'].append('SMA Cross (10/20)')
                summary_data['Current Value'].append(f"{current_sma10:.2f} / {current_sma20:.2f}")
                if current_sma10 > current_sma20:
                    summary_data['Signal'].append('🟢 Bullish')
                    summary_data['Interpretation'].append('Short-term above long-term')
                else:
                    summary_data['Signal'].append('🔴 Bearish')
                    summary_data['Interpretation'].append('Short-term below long-term')
            
            # Volume
            current_volume = filtered_data['volume'].iloc[-1]
            avg_volume = filtered_data['volume'].mean()
            volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 0
            
            summary_data['Indicator'].append('Volume')
            summary_data['Current Value'].append(f"${current_volume/1e6:.2f}M")
            if volume_ratio > 1.5:
                summary_data['Signal'].append('High')
                summary_data['Interpretation'].append('Strong interest')
            elif volume_ratio < 0.5:
                summary_data['Signal'].append('Low')
                summary_data['Interpretation'].append('Weak interest')
            else:
                summary_data['Signal'].append('Normal')
                summary_data['Interpretation'].append('Average activity')
            
            # Display table
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(
                summary_df,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary CSV",
                data=csv,
                file_name=f"{selected_coin}_technical_indicators_summary.csv",
                mime="text/csv"
            )
    
    def show_comparison_page(self):
        """Trang so sánh giữa các coin - Implementation đầy đủ"""
        st.title("CRYPTOCURRENCY COMPARISON")
        
        # === 1. COIN SELECTION ===
        st.markdown("### Select Coins to Compare")
        
        cols = st.columns(3)
        selected_coins = []
        
        for idx, coin in enumerate(self.coins):
            with cols[idx % 3]:
                if st.checkbox(coin.upper(), key=f"compare_{coin}"):
                    selected_coins.append(coin)
        
        if len(selected_coins) < 2:
            st.warning("Vui lòng chọn ít nhất 2 coins để so sánh")
            return
        
        st.success(f"Comparing {len(selected_coins)} coins: {', '.join([c.upper() for c in selected_coins])}")
        
        # Load data for all selected coins
        all_data = {}
        for coin in selected_coins:
            df = self.load_historical_data(coin)
            if not df.empty:
                all_data[coin] = df
        
        if len(all_data) < 2:
            st.error("Not enough data to compare. Please select different coins.")
            return
        
        st.markdown("---")
        
        # === 2. TIME RANGE SELECTOR ===
        col1, col2 = st.columns([3, 1])
        
        with col1:
            time_range = st.selectbox(
                "Select Time Range for Comparison",
                ["30 Days", "90 Days", "6 Months", "1 Year", "All Time"],
                index=2
            )
        
        with col2:
            st.metric("Coins Selected", len(selected_coins))
        
        # Filter data based on time range
        range_map = {
            "30 Days": 30,
            "90 Days": 90,
            "6 Months": 180,
            "1 Year": 365,
            "All Time": None
        }
        
        days = range_map[time_range]
        filtered_data = {}
        for coin, df in all_data.items():
            filtered_data[coin] = df.tail(days) if days else df
        
        st.markdown("---")
        
        # === 3. KEY METRICS COMPARISON ===
        st.markdown("### Key Metrics Comparison")
        
        # Create metrics dataframe
        metrics_data = []
        for coin in selected_coins:
            if coin in filtered_data:
                df = filtered_data[coin]
                current_price = df['close'].iloc[-1]
                price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100) if len(df) > 1 else 0
                volatility = df['close'].pct_change().std() * 100
                avg_volume = df['volume'].mean()
                high_price = df['high'].max()
                low_price = df['low'].min()
                
                metrics_data.append({
                    'Coin': coin.upper(),
                    'Current Price': f"${current_price:,.2f}",
                    f'{time_range} Change': f"{price_change:+.2f}%",
                    'Volatility': f"{volatility:.2f}%",
                    'Avg Volume': f"${avg_volume/1e9:.2f}B" if avg_volume > 1e9 else f"${avg_volume/1e6:.0f}M",
                    'High': f"${high_price:,.2f}",
                    'Low': f"${low_price:,.2f}",
                    'Range': f"${high_price - low_price:,.2f}"
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # === 4. PRICE COMPARISON CHARTS ===
        st.markdown("### Price Performance Comparison")
        
        tab1, tab2, tab3 = st.tabs(["Normalized Prices", "Absolute Prices", "Price Changes"])
        
        with tab1:
            st.markdown("#### Normalized Price Comparison (Base 100)")
            st.info("Normalized to 100 at start date for fair comparison regardless of absolute price levels")
            
            fig_norm = go.Figure()
            
            for coin in selected_coins:
                if coin in filtered_data:
                    df = filtered_data[coin]
                    # Normalize to 100
                    normalized_prices = (df['close'] / df['close'].iloc[0]) * 100
                    
                    fig_norm.add_trace(go.Scatter(
                        x=df.index,
                        y=normalized_prices,
                        name=coin.upper(),
                        mode='lines',
                        line=dict(width=2),
                        hovertemplate=f'<b>{coin.upper()}</b><br>Index: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>'
                    ))
            
            fig_norm.update_layout(
                height=500,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis_title="Date",
                yaxis_title="Normalized Price Index (Base 100)",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    bgcolor='rgba(0,0,0,0.8)',
                    bordercolor='rgba(102, 126, 234, 0.3)',
                    borderwidth=1
                ),
                margin=dict(r=120)
            )
            
            fig_norm.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_norm.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_norm, use_container_width=True)
            
            # Performance ranking
            performance_data = []
            for coin in selected_coins:
                if coin in filtered_data:
                    df = filtered_data[coin]
                    change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100) if len(df) > 1 else 0
                    performance_data.append({'Coin': coin.upper(), 'Performance': change})
            
            performance_df = pd.DataFrame(performance_data).sort_values('Performance', ascending=False)
            
            st.markdown("#### 🏆 Performance Ranking")
            col1, col2, col3 = st.columns(3)
            
            for idx, row in performance_df.iterrows():
                col_idx = idx % 3
                with [col1, col2, col3][col_idx]:
                    emoji = "1st" if idx == 0 else "2nd" if idx == 1 else "3rd" if idx == 2 else f"{idx+1}th"
                    delta_color = "normal" if row['Performance'] >= 0 else "inverse"
                    st.metric(
                        f"{emoji} {row['Coin']}",
                        f"{row['Performance']:+.2f}%",
                        delta_color=delta_color
                    )
        
        with tab2:
            st.markdown("#### Absolute Price Comparison")
            st.info("💡 Actual USD prices over time")
            
            fig_abs = go.Figure()
            
            for coin in selected_coins:
                if coin in filtered_data:
                    df = filtered_data[coin]
                    
                    fig_abs.add_trace(go.Scatter(
                        x=df.index,
                        y=df['close'],
                        name=coin.upper(),
                        mode='lines',
                        line=dict(width=2),
                        fill='tozeroy',
                        hovertemplate=f'<b>{coin.upper()}</b><br>Price: $%{{y:,.2f}}<br>Date: %{{x}}<extra></extra>'
                    ))
            
            fig_abs.update_layout(
                height=500,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    bgcolor='rgba(0,0,0,0.8)',
                    bordercolor='rgba(102, 126, 234, 0.3)',
                    borderwidth=1
                ),
                margin=dict(r=120)
            )
            
            fig_abs.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_abs.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_abs, use_container_width=True)
        
        with tab3:
            st.markdown("#### Daily Price Changes (%)")
            st.info("💡 Daily percentage changes to identify volatility patterns")
            
            fig_changes = go.Figure()
            
            for coin in selected_coins:
                if coin in filtered_data:
                    df = filtered_data[coin]
                    daily_changes = df['close'].pct_change() * 100
                    
                    fig_changes.add_trace(go.Scatter(
                        x=df.index,
                        y=daily_changes,
                        name=coin.upper(),
                        mode='lines',
                        line=dict(width=1.5),
                        hovertemplate=f'<b>{coin.upper()}</b><br>Change: %{{y:+.2f}}%<br>Date: %{{x}}<extra></extra>'
                    ))
            
            # Add zero line
            fig_changes.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            
            fig_changes.update_layout(
                height=500,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis_title="Date",
                yaxis_title="Daily Change (%)",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    bgcolor='rgba(0,0,0,0.8)',
                    bordercolor='rgba(102, 126, 234, 0.3)',
                    borderwidth=1
                ),
                margin=dict(r=120)
            )
            
            fig_changes.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_changes.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_changes, use_container_width=True)
        
        st.markdown("---")
        
        # === 5. VOLUME COMPARISON ===
        st.markdown("### 💰 Volume Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Trading Volume Over Time")
            
            fig_vol = go.Figure()
            
            for coin in selected_coins:
                if coin in filtered_data:
                    df = filtered_data[coin]
                    
                    fig_vol.add_trace(go.Bar(
                        x=df.index,
                        y=df['volume'],
                        name=coin.upper(),
                        hovertemplate=f'<b>{coin.upper()}</b><br>Volume: %{{y:,.0f}}<br>Date: %{{x}}<extra></extra>'
                    ))
            
            fig_vol.update_layout(
                height=450,
                barmode='group',
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis_title="Date",
                yaxis_title="Volume",
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    bgcolor='rgba(0,0,0,0.8)',
                    bordercolor='rgba(102, 126, 234, 0.3)',
                    borderwidth=1
                ),
                margin=dict(r=100, b=60)
            )
            
            fig_vol.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_vol.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col2:
            st.markdown("#### Average Volume Comparison")
            
            avg_volumes = []
            coin_names = []
            
            for coin in selected_coins:
                if coin in filtered_data:
                    df = filtered_data[coin]
                    avg_vol = df['volume'].mean()
                    avg_volumes.append(avg_vol)
                    coin_names.append(coin.upper())
            
            fig_avg_vol = go.Figure(data=[
                go.Bar(
                    x=coin_names,
                    y=avg_volumes,
                    marker=dict(
                        color=avg_volumes,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Volume")
                    ),
                    text=[f"${v/1e9:.2f}B" if v > 1e9 else f"${v/1e6:.0f}M" for v in avg_volumes],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Avg Volume: %{y:,.0f}<extra></extra>'
                )
            ])
            
            fig_avg_vol.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis_title="Cryptocurrency",
                yaxis_title="Average Volume",
                showlegend=False
            )
            
            fig_avg_vol.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_avg_vol.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_avg_vol, use_container_width=True)
        
        st.markdown("---")
        
        # === 6. VOLATILITY COMPARISON ===
        st.markdown("### 📊 Volatility & Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Volatility Comparison")
            
            volatilities = []
            coin_names = []
            
            for coin in selected_coins:
                if coin in filtered_data:
                    df = filtered_data[coin]
                    vol = df['close'].pct_change().std() * 100
                    volatilities.append(vol)
                    coin_names.append(coin.upper())
            
            fig_vol_comp = go.Figure(data=[
                go.Bar(
                    x=coin_names,
                    y=volatilities,
                    marker=dict(
                        color=volatilities,
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title="Volatility %")
                    ),
                    text=[f"{v:.2f}%" for v in volatilities],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Volatility: %{y:.2f}%<extra></extra>'
                )
            ])
            
            fig_vol_comp.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis_title="Cryptocurrency",
                yaxis_title="Volatility (%)",
                showlegend=False
            )
            
            fig_vol_comp.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_vol_comp.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_vol_comp, use_container_width=True)
        
        with col2:
            st.markdown("#### Price Range Comparison")
            
            ranges = []
            coin_names = []
            
            for coin in selected_coins:
                if coin in filtered_data:
                    df = filtered_data[coin]
                    price_range = df['high'].max() - df['low'].min()
                    range_pct = (price_range / df['low'].min() * 100) if df['low'].min() > 0 else 0
                    ranges.append(range_pct)
                    coin_names.append(coin.upper())
            
            fig_range = go.Figure(data=[
                go.Bar(
                    x=coin_names,
                    y=ranges,
                    marker=dict(
                        color=ranges,
                        colorscale='Blues',
                        showscale=True,
                        colorbar=dict(title="Range %")
                    ),
                    text=[f"{r:.2f}%" for r in ranges],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Price Range: %{y:.2f}%<extra></extra>'
                )
            ])
            
            fig_range.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis_title="Cryptocurrency",
                yaxis_title="Price Range (%)",
                showlegend=False
            )
            
            fig_range.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_range.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_range, use_container_width=True)
        
        st.markdown("---")
        
        # === 7. CORRELATION ANALYSIS ===
        st.markdown("### 🔗 Correlation Analysis")
        st.info("💡 Correlation matrix shows how closely the price movements of different cryptocurrencies are related")
        
        # Create correlation matrix
        price_data = pd.DataFrame()
        for coin in selected_coins:
            if coin in filtered_data:
                price_data[coin.upper()] = filtered_data[coin]['close']
        
        if not price_data.empty:
            correlation_matrix = price_data.corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 12},
                colorbar=dict(title="Correlation"),
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig_corr.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis_title="",
                yaxis_title="",
                title={
                    'text': 'Price Correlation Matrix',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': 'white'}
                }
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Correlation insights
            st.markdown("#### 💡 Correlation Insights")
            
            # Find highest and lowest correlations (excluding diagonal)
            corr_values = []
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    corr_values.append({
                        'pair': f"{correlation_matrix.index[i]} - {correlation_matrix.columns[j]}",
                        'correlation': correlation_matrix.iloc[i, j]
                    })
            
            if corr_values:
                corr_df = pd.DataFrame(corr_values).sort_values('correlation', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔗 Strongest Correlations**")
                    top_corr = corr_df.head(3)
                    for idx, row in top_corr.iterrows():
                        st.success(f"{row['pair']}: **{row['correlation']:.3f}**")
                
                with col2:
                    st.markdown("**🔀 Weakest Correlations**")
                    bottom_corr = corr_df.tail(3)
                    for idx, row in bottom_corr.iterrows():
                        st.info(f"{row['pair']}: **{row['correlation']:.3f}**")
        
        st.markdown("---")
        
        # === 8. DISTRIBUTION COMPARISON ===
        st.markdown("### 📦 Price Distribution Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Box Plot - Price Distribution")
            
            fig_box = go.Figure()
            
            for coin in selected_coins:
                if coin in filtered_data:
                    df = filtered_data[coin]
                    
                    fig_box.add_trace(go.Box(
                        y=df['close'],
                        name=coin.upper(),
                        boxmean='sd',
                        hovertemplate='<b>%{fullData.name}</b><br>Value: $%{y:,.2f}<extra></extra>'
                    ))
            
            fig_box.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                yaxis_title="Price (USD)",
                showlegend=False
            )
            
            fig_box.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            st.markdown("#### Histogram - Returns Distribution")
            
            fig_hist = go.Figure()
            
            for coin in selected_coins:
                if coin in filtered_data:
                    df = filtered_data[coin]
                    returns = df['close'].pct_change().dropna() * 100
                    
                    fig_hist.add_trace(go.Histogram(
                        x=returns,
                        name=coin.upper(),
                        opacity=0.7,
                        nbinsx=50,
                        hovertemplate='<b>%{fullData.name}</b><br>Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
                    ))
            
            fig_hist.update_layout(
                height=400,
                barmode='overlay',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis_title="Daily Returns (%)",
                yaxis_title="Frequency",
                showlegend=True
            )
            
            fig_hist.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_hist.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("---")
        
        # === 9. SUMMARY STATISTICS TABLE ===
        st.markdown("### 📋 Detailed Comparison Table")
        
        summary_data = []
        for coin in selected_coins:
            if coin in filtered_data:
                df = filtered_data[coin]
                
                current_price = df['close'].iloc[-1]
                start_price = df['close'].iloc[0]
                price_change = ((current_price - start_price) / start_price * 100) if start_price != 0 else 0
                
                high = df['high'].max()
                low = df['low'].min()
                avg_price = df['close'].mean()
                
                volatility = df['close'].pct_change().std() * 100
                avg_volume = df['volume'].mean()
                
                # Calculate Sharpe Ratio (simplified)
                returns = df['close'].pct_change().dropna()
                sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() != 0 else 0
                
                summary_data.append({
                    'Cryptocurrency': coin.upper(),
                    'Current Price': f"${current_price:,.2f}",
                    f'{time_range} Change': f"{price_change:+.2f}%",
                    'High': f"${high:,.2f}",
                    'Low': f"${low:,.2f}",
                    'Average': f"${avg_price:,.2f}",
                    'Volatility': f"{volatility:.2f}%",
                    'Avg Volume': f"${avg_volume/1e9:.2f}B" if avg_volume > 1e9 else f"${avg_volume/1e6:.0f}M",
                    'Sharpe Ratio': f"{sharpe:.2f}"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison Data (CSV)",
                data=csv,
                file_name=f"crypto_comparison_{time_range.replace(' ', '_').lower()}.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        
        # === 10. INSIGHTS & RECOMMENDATIONS ===
        st.markdown("### 💡 AI-Powered Insights")
        
        # Find best and worst performers
        if summary_data:
            best_performer = max(summary_data, key=lambda x: float(x[f'{time_range} Change'].rstrip('%')))
            worst_performer = min(summary_data, key=lambda x: float(x[f'{time_range} Change'].rstrip('%')))
            least_volatile = min(summary_data, key=lambda x: float(x['Volatility'].rstrip('%')))
            most_volatile = max(summary_data, key=lambda x: float(x['Volatility'].rstrip('%')))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"""
                **🏆 Best Performer: {best_performer['Cryptocurrency']}**
                - Change: {best_performer[f'{time_range} Change']}
                - Current Price: {best_performer['Current Price']}
                - Volatility: {best_performer['Volatility']}
                """)
                
                st.info(f"""
                **📉 Most Stable: {least_volatile['Cryptocurrency']}**
                - Volatility: {least_volatile['Volatility']}
                - Change: {least_volatile[f'{time_range} Change']}
                - Sharpe Ratio: {least_volatile['Sharpe Ratio']}
                """)
            
            with col2:
                st.error(f"""
                **📉 Worst Performer: {worst_performer['Cryptocurrency']}**
                - Change: {worst_performer[f'{time_range} Change']}
                - Current Price: {worst_performer['Current Price']}
                - Volatility: {worst_performer['Volatility']}
                """)
                
                st.warning(f"""
                **⚡ Most Volatile: {most_volatile['Cryptocurrency']}**
                - Volatility: {most_volatile['Volatility']}
                - Change: {most_volatile[f'{time_range} Change']}
                - Risk Level: High
                """)
    
    
    def show_prediction_page(self, selected_coin: str):
        """Trang dự đoán giá - Implementation đầy đủ"""
        st.title(f"PRICE PREDICTIONS - {selected_coin.upper()}")
        
        # Load data
        predictions = self.load_predictions(selected_coin)
        results = self.load_results(selected_coin)
        hist_data = self.load_historical_data(selected_coin)
        
        # === 1. OVERVIEW METRICS ===
        st.markdown("### 📊 Model & Prediction Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if results and 'evaluation' in results:
                mae = results['evaluation'].get('mae', 0)
                st.metric("Model MAE", f"${mae:.2f}", "Lower is better")
            else:
                st.metric("Model MAE", "N/A", "No data")
        
        with col2:
            if results and 'evaluation' in results:
                rmse = results['evaluation'].get('rmse', 0)
                st.metric("Model RMSE", f"${rmse:.2f}", "Lower is better")
            else:
                st.metric("Model RMSE", "N/A", "No data")
        
        with col3:
            if results and 'evaluation' in results:
                r2 = results['evaluation'].get('r2', 0) * 100
                st.metric("R² Score", f"{r2:.2f}%", "Higher is better")
            else:
                st.metric("R² Score", "N/A", "No data")
        
        with col4:
            if predictions and 'predictions' in predictions:
                pred_count = len(predictions['predictions'])
                st.metric("Predictions", f"{pred_count} days", "Forecast horizon")
            else:
                st.metric("Predictions", "N/A", "No data")
        
        st.markdown("---")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "🔮 Future Predictions",
            "📈 Training Results",
            "📊 Detailed Analysis"
        ])
        
        with tab1:
            self.show_future_predictions_tab(selected_coin, predictions, hist_data)
        
        with tab2:
            self.show_training_results_tab(selected_coin, results)
        
        with tab3:
            self.show_detailed_analysis_tab(selected_coin, predictions, results, hist_data)
    
    def show_future_predictions_tab(self, selected_coin: str, predictions: dict, hist_data: pd.DataFrame):
        """Tab hiển thị dự đoán tương lai"""
        st.markdown("### 🔮 Future Price Predictions")
        
        if not predictions or 'predictions' not in predictions:
            st.warning(f"⚠️ No future predictions available for {selected_coin.upper()}")
            st.info("💡 Run prediction: `python main.py --mode predict --coins " + selected_coin + "`")
            return
        
        # Extract prediction data
        pred_data = predictions['predictions']
        
        # Parse predictions
        if isinstance(pred_data, list) and len(pred_data) > 0:
            if isinstance(pred_data[0], dict):
                prices = [p.get('expected_price', p.get('price', 0)) for p in pred_data]
                days = [p.get('day', i+1) for i, p in enumerate(pred_data)]
            else:
                prices = pred_data
                days = list(range(1, len(prices) + 1))
        else:
            st.error("❌ Invalid prediction data format")
            return
        
        # Create proper timestamps for predictions
        # Start from the last date in historical data + 1 day
        if not hist_data.empty:
            last_date = hist_data.index[-1]
            timestamps = [last_date + pd.Timedelta(days=day) for day in days]
        else:
            # Fallback: use today + day offset
            from datetime import datetime, timedelta
            base_date = datetime.now()
            timestamps = [base_date + timedelta(days=day) for day in days]
        
        # === 1. PREDICTION CHART ===
        st.markdown("#### 📈 Predicted Price Trend")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig_pred = go.Figure()
            
            # Historical data (last 30 days for context)
            if not hist_data.empty:
                hist_recent = hist_data.tail(30)
                fig_pred.add_trace(go.Scatter(
                    x=hist_recent.index,
                    y=hist_recent['close'],
                    name='Historical Price',
                    mode='lines',
                    line=dict(color='#4169E1', width=2.5),  # Màu xanh dương đậm
                    hovertemplate='<b>Historical</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
                ))
                
                # Add connection point: last historical price as first prediction point
                last_hist_date = hist_recent.index[-1]
                last_hist_price = hist_recent['close'].iloc[-1]
                
                # Prepend last historical point to predictions for smooth connection
                pred_timestamps = [last_hist_date] + timestamps
                pred_prices = [last_hist_price] + prices
            else:
                pred_timestamps = timestamps
                pred_prices = prices
            
            # Predicted prices with connection
            fig_pred.add_trace(go.Scatter(
                x=pred_timestamps,
                y=pred_prices,
                name='Predicted Price',
                mode='lines',
                line=dict(color='#00FF00', width=2.5),  # Màu xanh lá cây nét liền
                hovertemplate='<b>Prediction</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
            ))
            
            # Confidence interval (if available)
            if 'confidence_lower' in predictions and 'confidence_upper' in predictions:
                fig_pred.add_trace(go.Scatter(
                    x=timestamps + timestamps[::-1],
                    y=predictions['confidence_upper'] + predictions['confidence_lower'][::-1],
                    fill='toself',
                    fillcolor='rgba(0, 255, 136, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True,
                    hoverinfo='skip'
                ))
            
            fig_pred.update_layout(
                height=500,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                title={
                    'text': f'{selected_coin.upper()} - Price Forecast',
                    'font': {'size': 18, 'color': 'white'}
                }
            )
            
            fig_pred.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_pred.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Prediction Stats")
            
            min_price = min(prices)
            max_price = max(prices)
            avg_price = sum(prices) / len(prices)
            
            st.metric("Min Predicted", f"${min_price:,.2f}")
            st.metric("Max Predicted", f"${max_price:,.2f}")
            st.metric("Avg Predicted", f"${avg_price:,.2f}")
            
            # Trend analysis
            if len(prices) > 1:
                if prices[0] != 0:
                    trend_pct = ((prices[-1] - prices[0]) / prices[0] * 100)
                else:
                    trend_pct = 0
                
                if trend_pct > 0:
                    st.success(f"📈 Uptrend\n+{trend_pct:.2f}%")
                elif trend_pct < 0:
                    st.error(f"📉 Downtrend\n{trend_pct:.2f}%")
                else:
                    st.info("➡️ Flat\n0.00%")
        
        st.markdown("---")
        
        # === 2. PREDICTION TABLE ===
        st.markdown("#### 📋 Detailed Predictions")
        
        # Create prediction dataframe
        pred_df_data = []
        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            pred_df_data.append({
                'Day': i + 1,
                'Date': ts,
                'Predicted Price': f"${price:,.2f}",
                'Change from Start': f"{((price - prices[0]) / prices[0] * 100):+.2f}%" if prices[0] != 0 else "N/A"
            })
        
        pred_df = pd.DataFrame(pred_df_data)
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions (CSV)",
            data=csv,
            file_name=f"{selected_coin}_predictions.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # === 3. PRICE DISTRIBUTION ===
        st.markdown("#### 📊 Prediction Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = go.Figure(data=[
                go.Histogram(
                    x=prices,
                    nbinsx=20,
                    marker=dict(
                        color='#00ff88',
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='Price: $%{x:,.2f}<br>Count: %{y}<extra></extra>'
                )
            ])
            
            fig_hist.update_layout(
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis_title="Predicted Price (USD)",
                yaxis_title="Frequency",
                title="Distribution of Predictions"
            )
            
            fig_hist.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_hist.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot
            fig_box = go.Figure(data=[
                go.Box(
                    y=prices,
                    name='Predictions',
                    marker=dict(color='#00ff88'),
                    boxmean='sd',
                    hovertemplate='Value: $%{y:,.2f}<extra></extra>'
                )
            ])
            
            fig_box.update_layout(
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                yaxis_title="Predicted Price (USD)",
                title="Price Range & Outliers",
                showlegend=False
            )
            
            fig_box.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_box, use_container_width=True)
    
    def show_training_results_tab(self, selected_coin: str, results: dict):
        """Tab hiển thị kết quả training"""
        st.markdown("### 📈 Training Results & Model Performance")
        
        if not results:
            st.warning(f"⚠️ No training results available for {selected_coin.upper()}")
            st.info("💡 Run training: `python main.py --mode train --coins " + selected_coin + "`")
            return
        
        # === 1. MODEL METRICS ===
        evaluation = results.get('evaluation', {})
        
        if evaluation:
            st.markdown("#### 📊 Model Performance Metrics")
            
            cols = st.columns(len(evaluation))
            
            for idx, (metric, value) in enumerate(evaluation.items()):
                with cols[idx]:
                    # Format metric name
                    metric_name = metric.upper().replace('_', ' ')
                    
                    # Determine if higher or lower is better
                    if metric.lower() in ['r2', 'r_squared', 'accuracy']:
                        delta_text = "Higher is better"
                        value_display = f"{value*100:.2f}%" if value < 1 else f"{value:.4f}"
                    else:
                        delta_text = "Lower is better"
                        value_display = f"${value:.2f}" if 'mae' in metric.lower() or 'rmse' in metric.lower() else f"{value:.4f}"
                    
                    st.metric(metric_name, value_display, delta_text)
        
        st.markdown("---")
        
        # === 2. TRAINING HISTORY ===
        history = results.get('history', {})
        
        if history:
            st.markdown("#### 📈 Training History")
            
            # Create subplots for loss and metrics
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Loss Over Epochs', 'MAE Over Epochs'),
                horizontal_spacing=0.12
            )
            
            # Loss plot
            if 'loss' in history:
                epochs = list(range(1, len(history['loss']) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history['loss'],
                        name='Training Loss',
                        mode='lines',
                        line=dict(color='#ff6b6b', width=2)
                    ),
                    row=1, col=1
                )
            
            if 'val_loss' in history:
                epochs = list(range(1, len(history['val_loss']) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history['val_loss'],
                        name='Validation Loss',
                        mode='lines',
                        line=dict(color='#00ff88', width=2)
                    ),
                    row=1, col=1
                )
            
            # MAE plot
            if 'mae' in history:
                epochs = list(range(1, len(history['mae']) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history['mae'],
                        name='Training MAE',
                        mode='lines',
                        line=dict(color='#667eea', width=2)
                    ),
                    row=1, col=2
                )
            
            if 'val_mae' in history:
                epochs = list(range(1, len(history['val_mae']) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history['val_mae'],
                        name='Validation MAE',
                        mode='lines',
                        line=dict(color='#ffd700', width=2)
                    ),
                    row=1, col=2
                )
            
            fig.update_xaxes(title_text="Epoch", gridcolor='rgba(102, 126, 234, 0.2)', row=1, col=1)
            fig.update_xaxes(title_text="Epoch", gridcolor='rgba(102, 126, 234, 0.2)', row=1, col=2)
            fig.update_yaxes(title_text="Loss", gridcolor='rgba(102, 126, 234, 0.2)', row=1, col=1)
            fig.update_yaxes(title_text="MAE", gridcolor='rgba(102, 126, 234, 0.2)', row=1, col=2)
            
            fig.update_layout(
                height=450,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Training insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'loss' in history:
                    initial_loss = history['loss'][0]
                    final_loss = history['loss'][-1]
                    improvement = ((initial_loss - final_loss) / initial_loss * 100)
                    st.success(f"""
                    **Training Loss Improvement**
                    - Initial: {initial_loss:.4f}
                    - Final: {final_loss:.4f}
                    - Improved: {improvement:.2f}%
                    """)
            
            with col2:
                if 'val_loss' in history and 'loss' in history:
                    overfitting = abs(history['loss'][-1] - history['val_loss'][-1])
                    if overfitting < 0.01:
                        st.info(f"""
                        **Model Fit Status**
                        - Difference: {overfitting:.4f}
                        - Status: ✅ Good fit
                        - Overfitting: Minimal
                        """)
                    else:
                        st.warning(f"""
                        **Model Fit Status**
                        - Difference: {overfitting:.4f}
                        - Status: ⚠️ Check needed
                        - Overfitting: Possible
                        """)
            
            with col3:
                if 'mae' in history:
                    total_epochs = len(history['mae'])
                    st.info(f"""
                    **Training Configuration**
                    - Total Epochs: {total_epochs}
                    - Final MAE: {history['mae'][-1]:.4f}
                    - Status: ✅ Completed
                    """)
        
        st.markdown("---")
        
        # === 3. PREDICTIONS VS ACTUAL ===
        if 'predictions' in results and 'actual' in results:
            st.markdown("#### 🎯 Predictions vs Actual (Test Set)")
            
            actual = results['actual']
            predicted = results['predictions']
            
            fig_pred_vs_actual = go.Figure()
            
            # Actual values
            fig_pred_vs_actual.add_trace(go.Scatter(
                y=actual,
                name='Actual Price',
                mode='lines',
                line=dict(color='#667eea', width=2),
                hovertemplate='Actual: $%{y:,.2f}<extra></extra>'
            ))
            
            # Predicted values
            fig_pred_vs_actual.add_trace(go.Scatter(
                y=predicted,
                name='Predicted Price',
                mode='lines',
                line=dict(color='#00ff88', width=2, dash='dash'),
                hovertemplate='Predicted: $%{y:,.2f}<extra></extra>'
            ))
            
            fig_pred_vs_actual.update_layout(
                height=400,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis_title="Time Step",
                yaxis_title="Price (USD)",
                title={
                    'text': f'{selected_coin.upper()} - Model Test Performance',
                    'font': {'size': 16, 'color': 'white'}
                }
            )
            
            fig_pred_vs_actual.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            fig_pred_vs_actual.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
            
            st.plotly_chart(fig_pred_vs_actual, use_container_width=True)
            
            # Error analysis
            errors = [abs(a - p) for a, p in zip(actual, predicted)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Error distribution
                fig_err = go.Figure(data=[
                    go.Histogram(
                        x=errors,
                        nbinsx=30,
                        marker=dict(color='#ff6b6b'),
                        hovertemplate='Error: $%{x:,.2f}<br>Count: %{y}<extra></extra>'
                    )
                ])
                
                fig_err.update_layout(
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'},
                    xaxis_title="Prediction Error (USD)",
                    yaxis_title="Frequency",
                    title="Error Distribution"
                )
                
                fig_err.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
                fig_err.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
                
                st.plotly_chart(fig_err, use_container_width=True)
            
            with col2:
                # Scatter plot
                fig_scatter = go.Figure(data=[
                    go.Scatter(
                        x=actual,
                        y=predicted,
                        mode='markers',
                        marker=dict(
                            size=6,
                            color='#00ff88',
                            opacity=0.6
                        ),
                        hovertemplate='Actual: $%{x:,.2f}<br>Predicted: $%{y:,.2f}<extra></extra>'
                    )
                ])
                
                # Add perfect prediction line
                min_val = min(min(actual), min(predicted))
                max_val = max(max(actual), max(predicted))
                fig_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='white', dash='dash', width=1),
                    name='Perfect Prediction',
                    showlegend=True
                ))
                
                fig_scatter.update_layout(
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'},
                    xaxis_title="Actual Price (USD)",
                    yaxis_title="Predicted Price (USD)",
                    title="Actual vs Predicted"
                )
                
                fig_scatter.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
                fig_scatter.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
                
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    def show_detailed_analysis_tab(self, selected_coin: str, predictions: dict, results: dict, hist_data: pd.DataFrame):
        """Tab phân tích chi tiết"""
        st.markdown("### 📊 Detailed Prediction Analysis")
        
        if not predictions or 'predictions' not in predictions:
            st.warning("⚠️ No prediction data available for detailed analysis")
            return
        
        # === 1. COMPARISON WITH HISTORICAL ===
        st.markdown("#### 🔍 Prediction vs Historical Comparison")
        
        if not hist_data.empty:
            # Get recent historical data
            hist_recent = hist_data.tail(90)
            
            # Extract prediction prices
            pred_data = predictions['predictions']
            if isinstance(pred_data[0], dict):
                pred_prices = [p.get('expected_price', p.get('price', 0)) for p in pred_data]
            else:
                pred_prices = pred_data
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                hist_avg = hist_recent['close'].mean()
                pred_avg = sum(pred_prices) / len(pred_prices)
                change = ((pred_avg - hist_avg) / hist_avg * 100)
                st.metric(
                    "Avg Price Change",
                    f"{change:+.2f}%",
                    f"Pred: ${pred_avg:,.2f} vs Hist: ${hist_avg:,.2f}"
                )
            
            with col2:
                hist_volatility = hist_recent['close'].pct_change().std() * 100
                pred_volatility = pd.Series(pred_prices).pct_change().std() * 100
                st.metric(
                    "Volatility Change",
                    f"{pred_volatility - hist_volatility:+.2f}%",
                    f"Pred: {pred_volatility:.2f}% vs Hist: {hist_volatility:.2f}%"
                )
            
            with col3:
                hist_high = hist_recent['high'].max()
                pred_high = max(pred_prices)
                st.metric(
                    "High Price",
                    f"${pred_high:,.2f}",
                    f"{((pred_high - hist_high) / hist_high * 100):+.2f}% vs Historical"
                )
            
            with col4:
                hist_low = hist_recent['low'].min()
                pred_low = min(pred_prices)
                st.metric(
                    "Low Price",
                    f"${pred_low:,.2f}",
                    f"{((pred_low - hist_low) / hist_low * 100):+.2f}% vs Historical"
                )
        
        st.markdown("---")
        
        # === 2. PREDICTION CONFIDENCE ===
        st.markdown("#### 🎯 Prediction Confidence Analysis")
        
        if results and 'evaluation' in results:
            evaluation = results['evaluation']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Model confidence gauge
                r2_score = evaluation.get('r2', 0) * 100
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=r2_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Model Confidence (R² Score)", 'font': {'size': 20, 'color': 'white'}},
                    delta={'reference': 80, 'suffix': '%'},
                    gauge={
                        'axis': {'range': [None, 100], 'tickcolor': 'white'},
                        'bar': {'color': "#00ff88"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "white",
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(255, 107, 107, 0.3)'},
                            {'range': [50, 75], 'color': 'rgba(255, 215, 0, 0.3)'},
                            {'range': [75, 100], 'color': 'rgba(0, 255, 136, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white', 'size': 14}
                )
                
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Error metrics visualization
                mae = evaluation.get('mae', 0)
                rmse = evaluation.get('rmse', 0)
                mape = evaluation.get('mape', 0) if 'mape' in evaluation else None
                
                fig_metrics = go.Figure()
                
                metrics_names = ['MAE', 'RMSE']
                metrics_values = [mae, rmse]
                
                if mape is not None:
                    metrics_names.append('MAPE')
                    metrics_values.append(mape)
                
                fig_metrics.add_trace(go.Bar(
                    x=metrics_names,
                    y=metrics_values,
                    marker=dict(
                        color=['#ff6b6b', '#ffd700', '#00ff88'][:len(metrics_names)],
                    ),
                    text=[f"${v:.2f}" if i < 2 else f"{v:.2f}%" for i, v in enumerate(metrics_values)],
                    textposition='auto',
                    hovertemplate='%{x}: %{text}<extra></extra>'
                ))
                
                fig_metrics.update_layout(
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'},
                    xaxis_title="Metric",
                    yaxis_title="Value",
                    title="Error Metrics (Lower is Better)",
                    showlegend=False
                )
                
                fig_metrics.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
                fig_metrics.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
                
                st.plotly_chart(fig_metrics, use_container_width=True)
        
        st.markdown("---")
        
        # === 3. RISK ASSESSMENT ===
        st.markdown("#### ⚠️ Risk Assessment")
        
        if predictions and 'predictions' in predictions:
            pred_data = predictions['predictions']
            if isinstance(pred_data[0], dict):
                pred_prices = [p.get('expected_price', p.get('price', 0)) for p in pred_data]
            else:
                pred_prices = pred_data
            
            # Calculate risk metrics
            pred_returns = pd.Series(pred_prices).pct_change().dropna()
            pred_volatility = pred_returns.std() * 100
            pred_sharpe = (pred_returns.mean() / pred_returns.std() * np.sqrt(365)) if pred_returns.std() != 0 else 0
            
            max_drawdown = 0
            peak = pred_prices[0]
            for price in pred_prices:
                if price > peak:
                    peak = price
                drawdown = ((peak - price) / peak * 100)
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if pred_volatility < 2:
                    st.success(f"""
                    **Volatility Risk: LOW**
                    - Predicted Volatility: {pred_volatility:.2f}%
                    - Risk Level: ✅ Low
                    - Suitable for conservative investors
                    """)
                elif pred_volatility < 5:
                    st.info(f"""
                    **Volatility Risk: MEDIUM**
                    - Predicted Volatility: {pred_volatility:.2f}%
                    - Risk Level: ⚠️ Moderate
                    - Suitable for balanced portfolios
                    """)
                else:
                    st.warning(f"""
                    **Volatility Risk: HIGH**
                    - Predicted Volatility: {pred_volatility:.2f}%
                    - Risk Level: 🔴 High
                    - Suitable for aggressive investors only
                    """)
            
            with col2:
                if max_drawdown < 10:
                    st.success(f"""
                    **Drawdown Risk: LOW**
                    - Max Drawdown: {max_drawdown:.2f}%
                    - Risk Level: ✅ Low
                    - Limited downside potential
                    """)
                elif max_drawdown < 20:
                    st.info(f"""
                    **Drawdown Risk: MEDIUM**
                    - Max Drawdown: {max_drawdown:.2f}%
                    - Risk Level: ⚠️ Moderate
                    - Moderate downside risk
                    """)
                else:
                    st.warning(f"""
                    **Drawdown Risk: HIGH**
                    - Max Drawdown: {max_drawdown:.2f}%
                    - Risk Level: 🔴 High
                    - Significant downside risk
                    """)
            
            with col3:
                if pred_sharpe > 1:
                    st.success(f"""
                    **Risk-Adjusted Return: GOOD**
                    - Sharpe Ratio: {pred_sharpe:.2f}
                    - Rating: ✅ Good
                    - Favorable risk/return profile
                    """)
                elif pred_sharpe > 0:
                    st.info(f"""
                    **Risk-Adjusted Return: FAIR**
                    - Sharpe Ratio: {pred_sharpe:.2f}
                    - Rating: ⚠️ Fair
                    - Acceptable risk/return trade-off
                    """)
                else:
                    st.warning(f"""
                    **Risk-Adjusted Return: POOR**
                    - Sharpe Ratio: {pred_sharpe:.2f}
                    - Rating: 🔴 Poor
                    - Unfavorable risk/return profile
                    """)


def main():
    dashboard = MonitoringDashboard()
    dashboard.show()


if __name__ == "__main__":
    main()