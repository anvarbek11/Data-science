import plotly.express as px
import numpy as np
import pandas as pd


def plot_metrics(metrics):
    """Create a radar chart of model metrics"""
    df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Score': list(metrics.values()),
        'Full_Score': [1.0] * len(metrics)
    })

    fig = px.line_polar(
        df,
        r='Score',
        theta='Metric',
        line_close=True,
        range_r=[0, 1],
        template="plotly_white"
    )
    fig.update_traces(fill='toself')
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_confusion_matrix(cm):
    """Visualize confusion matrix"""
    labels = ['Legitimate', 'Fraud']
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        aspect="auto"
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        coloraxis_showscale=False
    )
    fig.update_traces(
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}"
    )
    return fig


def plot_feature_distributions():
    """Show feature distributions (simplified)"""
    # This would normally use your actual feature distributions
    np.random.seed(42)
    data = {
        'Feature': [f'V{i}' for i in range(1, 31)],
        'Mean': np.random.normal(0, 1, 30),
        'StdDev': np.abs(np.random.normal(0.5, 0.2, 30))
    }
    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x='Feature',
        y='Mean',
        error_y='StdDev',
        title="Feature Distributions (Mean ± Std Dev)",
        color='Mean',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(
        xaxis_tickangle=-90,
        showlegend=False,
        height=600
    )
    return fig


def generate_simulation_results(avg_transaction, fraud_rate, volume, multiplier):
    """Generate business impact simulation results"""
    potential_losses = avg_transaction * volume * fraud_rate * multiplier
    estimated_savings = potential_losses * 0.87 * 0.5  # 87% recall, 50% effectiveness
    savings_percentage = estimated_savings / potential_losses
    roi = estimated_savings / (avg_transaction * volume * 0.0001)  # Assuming small cost

    return {
        'potential_losses': potential_losses,
        'estimated_savings': estimated_savings,
        'savings_percentage': savings_percentage,
        'roi': roi
    }