import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def analyze_prompt_performance(feedback_data):
    """Analyze which prompting styles perform best."""
    if not feedback_data or len(feedback_data) < 5:
        return "Not enough feedback data collected yet."
    
    # Convert to DataFrame
    df = pd.DataFrame(feedback_data)
    
    # Calculate helpful percentage by style
    style_performance = df.groupby('style')['rating'].apply(
        lambda x: (x == 'helpful').mean() * 100).reset_index()
    style_performance.columns = ['Style', 'Helpful %']
    
    # Sort by performance
    style_performance = style_performance.sort_values('Helpful %', ascending=False)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(style_performance['Style'], style_performance['Helpful %'])
    ax.set_ylabel('Helpfulness Score (%)')
    ax.set_title('Prompt Style Performance')
    plt.xticks(rotation=45)
    
    return fig, style_performance