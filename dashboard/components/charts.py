import pandas as pd
import plotly.graph_objects as go

# Cấu hình layout mặc định dùng chung cho tất cả các biểu đồ Plotly
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#94a3b8', size=12),
    xaxis=dict(gridcolor='#1e2535', linecolor='#1e2535'),
    yaxis=dict(gridcolor='#1e2535', linecolor='#1e2535'),
    margin=dict(l=16, r=16, t=40, b=16),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1e2535', borderwidth=1),
    hoverlabel=dict(bgcolor='#161b27', bordercolor='#1e2535', font_color='#e2e8f0'),
)

# Mã màu cho từng cửa hàng (Store)
STORE_COLORS = {
    'CA_1': '#3b82f6', 'CA_2': '#60a5fa', 'CA_3': '#93c5fd', 'CA_4': '#bfdbfe',
    'TX_1': '#8b5cf6', 'TX_2': '#a78bfa', 'TX_3': '#c4b5fd',
    'WI_1': '#06b6d4', 'WI_2': '#22d3ee', 'WI_3': '#67e8f9',
}

# Mã màu cho từng bang (State)
STATE_COLORS = {'CA': '#3b82f6', 'TX': '#8b5cf6', 'WI': '#06b6d4'}

def plot_item_forecast(df_history, df_actual, df_pred, item_id, store_id, calendar):
    """
    Vẽ biểu đồ forecast chi tiết cho một item (được gọi từ 02_item_detail.py)
    Bao gồm: Lịch sử, Actuals, Dự đoán, Vùng Confidence Interval và các Ngày Lễ.
    """
    df_hist_tail = df_history.tail(180) # Lấy 180 ngày gần nhất
    
    # Lấy thông tin ngày lễ trong khoảng thời gian dự đoán
    holiday_dates = calendar[
        (calendar['date'].isin(df_pred['ds'])) & 
        (calendar['event_name_1'].notna()) & 
        (calendar['event_name_1'] != 'No_Event')
    ][['date', 'event_name_1']].drop_duplicates()
    
    fig = go.Figure()
    
    # Vẽ vùng confidence interval (vùng bóng mờ biểu thị upper/lower bound)
    fig.add_trace(go.Scatter(
        x=pd.concat([df_pred['ds'], df_pred['ds'][::-1]]),
        y=pd.concat([df_pred['yhat_upper'], df_pred['yhat_lower'][::-1]]),
        fill='toself', 
        fillcolor='rgba(139, 92, 246, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval', 
        hoverinfo='skip'
    ))
    
    # Vẽ lịch sử (Historical Sales)
    fig.add_trace(go.Scatter(
        x=df_hist_tail['date'], 
        y=df_hist_tail['sales'],
        name='Historical', 
        mode='lines',
        line=dict(color='#475569', width=1.5),
        hovertemplate='<b>Historical</b><br>%{x|%Y-%m-%d}<br>Sales: %{y}<extra></extra>'
    ))
    
    # Vẽ dữ liệu thực tế 28 ngày cuối (Actual)
    if not df_actual.empty:
        fig.add_trace(go.Scatter(
            x=df_actual['date'], 
            y=df_actual['actual'],
            name='Actual (Last 28d)', 
            mode='lines+markers',
            line=dict(color='#06b6d4', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Actual</b><br>%{x|%Y-%m-%d}<br>Sales: %{y}<extra></extra>'
        ))
    
    # Vẽ dữ liệu dự đoán (Predicted)
    fig.add_trace(go.Scatter(
        x=df_pred['ds'], 
        y=df_pred['yhat'],
        name='Predicted', 
        mode='lines+markers',
        line=dict(color='#8b5cf6', width=2, dash='dot'),
        marker=dict(size=6),
        hovertemplate='<b>Predicted</b><br>%{x|%Y-%m-%d}<br>Sales: %{y:.2f}<extra></extra>'
    ))
    
    # Đánh dấu các ngày lễ bằng ngôi sao (Holidays)
    if not holiday_dates.empty:
        fig.add_trace(go.Scatter(
            x=holiday_dates['date'],
            y=[df_pred['yhat'].max() * 1.05] * len(holiday_dates),
            name='Holidays',
            mode='markers',
            marker=dict(color='#fbbf24', size=8, symbol='star'),
            text=holiday_dates['event_name_1'],
            hovertemplate='<b>Holiday</b><br>%{x|%Y-%m-%d}<br>%{text}<extra></extra>'
        ))
        
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig