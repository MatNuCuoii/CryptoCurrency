# src/monitoring/pages/eda_correlation.py

"""
EDA: Correlation Analysis Page - Trang phân tích tương quan
"""

import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analysis.market_analyzer import (
    load_all_coins_data,
    calculate_correlation_matrix,
    calculate_rolling_correlation_with_btc
)


def render_correlation_page():
    """Render trang phân tích tương quan."""
    st.title("🔗 Phân Tích Tương Quan")
    
    # Page introduction
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h3 style='color: white; margin: 0;'>🔗 Mối Quan Hệ Giữa Các Coin</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>
                Phân tích tương quan giữa các coin để hiểu mức độ đa dạng hóa của danh mục.
                Coin có tương quan thấp giúp giảm rủi ro tổng thể.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Đang tải dữ liệu..."):
        data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        st.error("❌ Không có dữ liệu")
        return
    
    # Correlation Matrix
    st.subheader("📊 Ma Trận Tương Quan (Lợi Nhuận)")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📊 Ma Trận Này Cho Biết Gì?</h4>
            <p style='margin: 0; color: #ccc;'>
                Ma trận hiển thị hệ số tương quan giữa từng cặp coin. Giá trị từ -1 đến +1.
            </p>
            <h4 style='margin: 1rem 0 0.5rem 0; color: #667eea;'>💡 Cách Đọc</h4>
            <ul style='margin: 0; color: #ccc; padding-left: 1.5rem;'>
                <li><strong>+1.0 (Đỏ đậm)</strong>: Tương quan hoàn hảo - di chuyển cùng chiều 100%</li>
                <li><strong>0.0 (Trắng)</strong>: Không tương quan - di chuyển độc lập</li>
                <li><strong>-1.0 (Xanh đậm)</strong>: Tương quan nghịch - di chuyển ngược chiều</li>
                <li><strong>Đa dạng hóa tốt</strong>: Chọn coin có tương quan thấp (<0.5)</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    corr_matrix = calculate_correlation_matrix(data_dict, window=None)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[coin.upper() for coin in corr_matrix.columns],
        y=[coin.upper() for coin in corr_matrix.index],
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Tương Quan")
    ))
    
    fig.update_layout(
        title="Ma Trận Tương Quan (Toàn Bộ Thời Gian)",
        height=600,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Analysis Summary
    avg_corr = corr_matrix.mean().mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Tương Quan Trung Bình", f"{avg_corr:.2f}")
    with col2:
        max_corr = corr_matrix.where(corr_matrix != 1).max().max()
        st.metric("📈 Tương Quan Cao Nhất", f"{max_corr:.2f}")
    with col3:
        min_corr = corr_matrix.min().min()
        st.metric("📉 Tương Quan Thấp Nhất", f"{min_corr:.2f}")
    
    # Rolling Correlation with Bitcoin
    st.markdown("---")
    st.subheader("📈 Tương Quan Lăn Với Bitcoin (30 Ngày)")
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📊 Tại Sao So Với Bitcoin?</h4>
            <p style='margin: 0; color: #ccc;'>
                Bitcoin là coin dẫn dắt thị trường. Tương quan cao với BTC = coin theo sát thị trường chung.
                Tương quan thấp hoặc âm = coin có thể hoạt động khác biệt, tốt cho đa dạng hóa.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    rolling_corrs = calculate_rolling_correlation_with_btc(data_dict, window=30)
    
    if rolling_corrs:
        fig = go.Figure()
        
        colors = ['#667eea', '#00d4aa', '#ffc107', '#ff6b6b', '#17a2b8', '#28a745', '#fd7e14', '#6f42c1']
        
        for i, (coin, corr_series) in enumerate(rolling_corrs.items()):
            fig.add_trace(go.Scatter(
                x=corr_series.index,
                y=corr_series,
                name=coin.upper(),
                mode='lines',
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="yellow", 
                      annotation_text="Ngưỡng tương quan cao")
        
        fig.update_layout(
            title="Tương Quan Lăn 30 Ngày Với Bitcoin",
            xaxis_title="Ngày",
            yaxis_title="Hệ Số Tương Quan",
            height=500,
            hovermode='x unified',
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.markdown("---")
    st.subheader("💡 Nhận Định Tương Quan")
    
    if avg_corr > 0.7:
        st.warning(f"""
            ⚠️ **Tương Quan Cao** ({avg_corr:.2f})
            
            Các coin có xu hướng di chuyển cùng chiều mạnh. Điều này có nghĩa:
            - Đa dạng hóa trong danh mục này có lợi ích hạn chế
            - Khi thị trường giảm, phần lớn coin sẽ giảm cùng lúc
            - Cân nhắc thêm tài sản ngoài crypto để đa dạng hóa
        """)
    elif avg_corr < 0.3:
        st.success(f"""
            ✅ **Tương Quan Thấp** ({avg_corr:.2f})
            
            Các coin hoạt động khá độc lập. Điều này có nghĩa:
            - Tiềm năng đa dạng hóa tốt trong danh mục
            - Rủi ro tổng thể có thể được giảm thiểu
            - Các coin khác nhau có thể bù đắp lẫn nhau
        """)
    else:
        st.info(f"""
            ℹ️ **Tương Quan Vừa Phải** ({avg_corr:.2f})
            
            Các coin có mức độ liên kết trung bình. Điều này có nghĩa:
            - Có một số lợi ích đa dạng hóa
            - Khi thị trường biến động mạnh, các coin vẫn có xu hướng đi cùng chiều
            - Nên chọn lọc coin có tương quan thấp để tối ưu danh mục
        """)
    
    # Best pairs for diversification
    st.markdown("---")
    st.subheader("🎯 Cặp Coin Tốt Nhất Cho Đa Dạng Hóa")
    
    # Find lowest correlation pairs
    pairs = []
    for i, coin1 in enumerate(corr_matrix.columns):
        for j, coin2 in enumerate(corr_matrix.columns):
            if i < j:
                pairs.append((coin1, coin2, corr_matrix.loc[coin1, coin2]))
    
    pairs_sorted = sorted(pairs, key=lambda x: x[2])[:5]
    
    st.markdown("**5 Cặp Coin Có Tương Quan Thấp Nhất:**")
    for coin1, coin2, corr in pairs_sorted:
        color = "🟢" if corr < 0.3 else "🟡" if corr < 0.5 else "🟠"
        st.markdown(f"{color} **{coin1.upper()}** & **{coin2.upper()}**: Tương quan {corr:.2f}")
