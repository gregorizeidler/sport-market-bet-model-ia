import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import os
from openai import OpenAI
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Configure your OpenAI API key (replace with your actual key or set via environment variables)
client = OpenAI(api_key="")

##############################################
# Adicionar estilos globais para melhorar a legibilidade com texto branco puro
##############################################
# Definindo CSS personalizado para melhorar a legibilidade em todo o aplicativo
def apply_custom_css():
    st.markdown("""
    <style>
        /* Definir cores explícitas em vez de usar variáveis CSS */
        :root {
            --primary-color: #18A558;
            --secondary-color: #1E3A8A;
            --accent-color: #e84855;
            --background: #121212;
            --card-background: #1E1E1E;
            --text-color: #FFFFFF;
            --light-gray: #CCCCCC;
        }

        /* Garantir que texto em toda a aplicação seja branco puro */
        .stMarkdown, .stText, p, span, h1, h2, h3, h4, h5, h6, div {
            color: #FFFFFF !important;
        }
        
        /* Garantir que código HTML seja renderizado e não exibido como texto */
        .element-container div.stMarkdown {
            overflow: visible !important;
        }
        
        /* Resolver problema com blocos de HTML em cartões */
        pre code {
            display: none !important;
        }
        
        /* Ocultar qualquer código que esteja sendo mostrado como texto */
        .language-html, .language-css {
            display: none !important;
        }
        
        /* Estilos para o sidebar mais profissional */
        [data-testid="stSidebar"] {
            background-color: #0F0F0F !important;
            border-right: 1px solid rgba(255,255,255,0.05) !important;
            box-shadow: 5px 0 15px rgba(0,0,0,0.2) !important;
        }
        
        /* Estilo para itens do sidebar */
        div[data-testid="stRadio"] > div {
            background-color: #1A1A1A !important;
            border-radius: 8px !important;
            padding: 10px !important;
            margin-bottom: 8px !important;
            border-left: 3px solid transparent !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        }
        
        div[data-testid="stRadio"] > div:hover {
            background-color: #252525 !important;
            border-left: 3px solid #18A558 !important;
            transform: translateX(3px) !important;
        }
        
        /* Estilo para botões */
        .stButton > button {
            background: linear-gradient(90deg, #18A558, #18A558cc) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 3px 5px rgba(0,0,0,0.2) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(24, 165, 88, 0.3) !important;
        }
        
        /* Estilos específicos para cartões de informação */
        .info-card {
            background-color: #1A1A1A !important; 
            border-left: 5px solid #18A558 !important; 
            padding: 20px !important; 
            margin: 20px 0 !important; 
            border-radius: 10px !important;
            color: #FFFFFF !important;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
            transition: transform 0.3s ease !important;
        }
        
        .info-card:hover {
            transform: translateY(-5px) !important;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3) !important;
        }
        
        .info-card h4 {
            color: #18A558 !important; 
            margin-top: 0 !important;
            font-weight: bold !important;
            font-size: 1.4em !important;
            margin-bottom: 15px !important;
        }
        
        .info-card p {
            color: #FFFFFF !important;
            margin-bottom: 15px !important;
            line-height: 1.6 !important;
        }
        
        .info-card strong {
            color: #18A558 !important;
            font-weight: 600 !important;
        }
        
        /* Design para cabeçalhos e textos introdutórios */
        .section-header {
            background: linear-gradient(135deg, #18A558, #126e3b) !important;
            color: #FFFFFF !important;
            padding: 20px 25px !important;
            border-radius: 12px !important;
            margin: 0 0 30px 0 !important;
            box-shadow: 0 8px 20px rgba(24, 165, 88, 0.2) !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .section-header::before {
            content: "" !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            bottom: 0 !important;
            background-image: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px) !important;
            background-size: 20px 20px !important;
            opacity: 0.3 !important;
            z-index: 0 !important;
        }
        
        .section-header h1 {
            position: relative !important;
            z-index: 1 !important;
        }
        
        .upload-container {
            background-color: #1A1A1A !important;
            padding: 30px !important;
            border-radius: 15px !important;
            border-left: 5px solid #18A558 !important;
            margin-bottom: 30px !important;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3) !important;
            transition: transform 0.3s ease !important;
        }
        
        .upload-container:hover {
            transform: translateY(-5px) !important;
        }
        
        /* Melhorar a legibilidade de tabelas e dataframes */
        .dataframe {
            color: #FFFFFF !important;
            border-radius: 10px !important;
            overflow: hidden !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
        }
        
        .dataframe th {
            background-color: #18A558 !important;
            color: white !important;
            padding: 12px 15px !important;
            text-align: left !important;
        }
        
        .dataframe td {
            padding: 10px 15px !important;
            border-bottom: 1px solid rgba(255,255,255,0.05) !important;
        }
        
        .dataframe tr:hover td {
            background-color: rgba(24, 165, 88, 0.1) !important;
        }
        
        /* Corrigir códigos de amostra e blocos de texto */
        span[style*="padding: 6px 12px; border-radius: 4px;"] {
            background-color: rgba(24, 165, 88, 0.15) !important;
            color: #FFFFFF !important;
            font-family: monospace !important;
            padding: 8px 15px !important;
            border-radius: 6px !important;
            border: 1px solid rgba(24, 165, 88, 0.3) !important;
        }
        
        /* Melhorar visibilidade de texto em caixas de métricas */
        [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }
        
        [data-testid="stMetricDelta"] {
            background-color: rgba(24, 165, 88, 0.1) !important;
            padding: 5px 10px !important;
            border-radius: 20px !important;
            font-weight: 600 !important;
        }
        
        /* Melhorar o contraste e design da caixa de dicas */
        .info-card .tip-box {
            background: linear-gradient(135deg, #1E3A8A, #13255a) !important; 
            padding: 15px !important; 
            border-radius: 10px !important;
            margin-top: 15px !important;
            position: relative !important;
            overflow: hidden !important;
            box-shadow: 0 5px 15px rgba(30, 58, 138, 0.3) !important;
        }
        
        .info-card .tip-box::before {
            content: "" !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            bottom: 0 !important;
            background-image: radial-gradient(circle, rgba(255,255,255,0.05) 1px, transparent 1px) !important;
            background-size: 15px 15px !important;
            opacity: 0.2 !important;
        }
        
        .info-card .tip-box p {
            color: #FFFFFF !important; 
            margin: 0 !important;
            font-weight: 500 !important;
            position: relative !important;
            z-index: 1 !important;
        }
        
        .info-card .tip-box p span {
            color: #FFFFFF !important; 
        }
        
        .info-card .tip-box .dica-label,
        .info-card .tip-box span:first-child {
            color: #FFEB3B !important; 
            font-weight: bold !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
        }
    </style>
    """, unsafe_allow_html=True)

##############################################
# Utility Functions
##############################################

# Function to get consistent theme colors for charts and visualizations
def get_theme_colors():
    """
    Returns a dictionary of theme colors for consistency across the application
    """
    colors = {
        'primary_color': '#18A558',  # Green
        'secondary_color': '#1E2761',  # Dark Blue
        'accent_color': '#e84855',  # Red accent
        'background': '#121212',  # Dark background
        'card_background': '#1E1E1E',  # Slightly lighter background
        'text': '#FFFFFF',  # Texto mais claro - alterado de EBEBEB para branco puro
        'light_gray': '#CCCCCC',  # Cinza mais claro para melhor contraste - alterado de A0A0A0
        'home_win': '#18A558',  # Green for home wins
        'draw': '#5D8BF4',  # Azul mais claro para empates - alterado de 1E2761
        'away_win': '#FF5252',  # Vermelho mais vivo para vitórias fora - alterado de e84855
        'grid': 'rgba(255,255,255,0.1)'  # Grid color for charts
    }
    return colors

# Function to apply theme to plotly figures
def apply_theme_to_plotly(fig):
    """
    Applies a consistent theme to Plotly figures
    """
    colors = get_theme_colors()
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text'], size=14),  # Aumentando o tamanho da fonte e garantindo cor branca
        title=dict(font=dict(color=colors['text'], size=18)),  # Título em branco e maior
        legend=dict(
            font=dict(color=colors['text']),
            bgcolor=colors['card_background'],
            bordercolor=colors['primary_color']
        ),
        xaxis=dict(
            gridcolor=colors['grid'],
            zerolinecolor=colors['grid'],
            title_font=dict(color=colors['text'], size=14),  # Garantindo texto do eixo X em branco
            tickfont=dict(color=colors['text'])  # Texto dos ticks em branco
        ),
        yaxis=dict(
            gridcolor=colors['grid'],
            zerolinecolor=colors['grid'],
            title_font=dict(color=colors['text'], size=14),  # Garantindo texto do eixo Y em branco
            tickfont=dict(color=colors['text'])  # Texto dos ticks em branco
        ),
        colorway=[colors['primary_color'], colors['secondary_color'], colors['accent_color'], 
                 colors['home_win'], colors['draw'], colors['away_win']]
    )
    
    # Garantir que anotações e outros textos também estejam em branco
    if 'annotations' in fig.layout:
        for annotation in fig.layout.annotations:
            annotation.font.color = colors['text']
    
    return fig

from sklearn.decomposition import PCA
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, roc_curve, 
                             precision_recall_curve, f1_score, precision_score, recall_score, 
                             brier_score_loss)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB  # Bayesian model

# For LIME (optional)
try:
    import lime
    import lime.lime_tabular
except ImportError:
    st.warning("Install 'lime' (pip install lime) to use LIME explanations.")

# Function to load data from uploaded file or sample data
def load_data(file):
    """
    Load and preprocess data from uploaded file or sample data
    Supports .xlsx, .xls, .xlsm, .csv
    """
    try:
        if isinstance(file, str):  # For sample data filepath
            if file.endswith('.csv'):
                df = pd.read_csv(file, delimiter=',', encoding='utf-8', na_values=['NA', 'N/A', ''], keep_default_na=True)
            elif file.endswith(('.xlsx', '.xls', '.xlsm')):
                df = pd.read_excel(file)
            else:
                raise ValueError("Unsupported file format. Please upload .csv, .xlsx, .xls, or .xlsm files.")
        else:  # For uploaded file
            file_extension = file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(file, delimiter=',', encoding='utf-8', na_values=['NA', 'N/A', ''], keep_default_na=True)
            elif file_extension in ['xlsx', 'xls', 'xlsm']:
                df = pd.read_excel(file)
            else:
                raise ValueError("Unsupported file format. Please upload .csv, .xlsx, .xls, or .xlsm files.")
        
        # Handle data format compatibility - check for common sports betting dataset formats
        # Format 1: Standard format with match_id, match_outcome, odds columns
        if all(col in df.columns for col in ['match_id', 'match_outcome', 'home_win_odds', 'draw_odds', 'away_win_odds']):
            # Standard format already
            pass
            
        # Format 2: Sports market CSV format with HOME, AWAY, ODD columns
        elif all(col in df.columns for col in ['HOME', 'AWAY', 'ODD Fecho H', 'ODD Fecho D', 'ODD Fecho A']):
            # Create match_id if not exists
            if 'match_id' not in df.columns:
                df['match_id'] = df.apply(lambda row: f"{row['PAÍS']}_{row['LIGA']}_{row['HOME']}_{row['AWAY']}_{row['DATA']}" 
                                          if all(col in df.columns for col in ['PAÍS', 'LIGA', 'HOME', 'AWAY', 'DATA']) 
                                          else f"{row['HOME']}_{row['AWAY']}_{df.index.get_loc(row.name)}", axis=1)
            
            # Map odds columns
            df['home_win_odds'] = pd.to_numeric(df['ODD Fecho H'].astype(str).str.replace(',', '.'), errors='coerce')
            df['draw_odds'] = pd.to_numeric(df['ODD Fecho D'].astype(str).str.replace(',', '.'), errors='coerce')
            df['away_win_odds'] = pd.to_numeric(df['ODD Fecho A'].astype(str).str.replace(',', '.'), errors='coerce')
            
            # Determine match outcome based on goals
            if all(col in df.columns for col in ['Gols_H', 'Gols_A']):
                # 1: Home win, 0: Draw, 2: Away win
                df['match_outcome'] = df.apply(lambda row: 
                                              1 if row['Gols_H'] > row['Gols_A'] else 
                                              (0 if row['Gols_H'] == row['Gols_A'] else 2), axis=1)
            else:
                # If no goal data, try to infer from WR column or set as missing
                if 'WR' in df.columns:
                    df['match_outcome'] = df['WR'].astype(int)
                else:
                    raise ValueError("Cannot determine match outcome. Please ensure your data includes home goals (Gols_H) and away goals (Gols_A) columns.")
            
            # Map additional columns if available
            col_mapping = {
                'home_team': 'HOME',
                'away_team': 'AWAY',
                'league': 'LIGA',
                'country': 'PAÍS',
                'date': 'DATA',
                'home_goals': 'Gols_H',
                'away_goals': 'Gols_A',
                'home_corners': 'corners_h',
                'away_corners': 'corners_a',
            }
            
            for target_col, source_col in col_mapping.items():
                if source_col in df.columns and target_col not in df.columns:
                    df[target_col] = df[source_col]
                    
            # Add calculated columns
            if 'home_team_rank' not in df.columns and all(col in df.columns for col in ['EV', 'Linha > 0']):
                # Use these as proxies for team ranking
                df['home_team_rank'] = pd.to_numeric(df['EV'].astype(str).str.replace(',', '.'), errors='coerce')
                df['away_team_rank'] = pd.to_numeric(df['Linha > 0'].astype(str).str.replace(',', '.'), errors='coerce')
                
            if 'home_recent_wins' not in df.columns and 'Linha >= 0' in df.columns:
                # Use as proxy for recent performance
                df['home_recent_wins'] = pd.to_numeric(df['Linha >= 0'].astype(str).str.replace(',', '.'), errors='coerce')
                df['away_recent_wins'] = 1 - df['home_recent_wins']  # Inverse relationship
                
        else:
            # Try to identify common column patterns and make best effort to map
            odds_patterns = {
                'home_win_odds': ['home_odds', 'h_odds', 'odds_h', 'home_win', 'odd_home', 'odds_home', 'odds_1'],
                'draw_odds': ['draw_odds', 'x_odds', 'odds_x', 'draw', 'odd_draw', 'odds_draw', 'odds_x'],
                'away_win_odds': ['away_odds', 'a_odds', 'odds_a', 'away_win', 'odd_away', 'odds_away', 'odds_2']
            }
            
            for target, patterns in odds_patterns.items():
                if target not in df.columns:
                    for pattern in patterns:
                        matches = [col for col in df.columns if pattern.lower() in col.lower()]
                        if matches:
                            df[target] = pd.to_numeric(df[matches[0]].astype(str).str.replace(',', '.'), errors='coerce')
                            break
            
            # Create match_id if not present
            if 'match_id' not in df.columns:
                df['match_id'] = df.index
            
            # Unable to find required columns
            missing_cols = []
            for col in ['match_id', 'home_win_odds', 'draw_odds', 'away_win_odds']:
                if col not in df.columns:
                    missing_cols.append(col)
                    
            if missing_cols:
                raise ValueError(f"Unable to find required columns: {', '.join(missing_cols)}. Please ensure your file has these columns or compatible alternatives.")
                
        # Calculate implied probabilities and overround
        df['home_implied_prob'] = 1 / df['home_win_odds']
        df['draw_implied_prob'] = 1 / df['draw_odds']
        df['away_implied_prob'] = 1 / df['away_win_odds']
        
        # Calculate overround (bookmaker margin)
        df['overround'] = df['home_implied_prob'] + df['draw_implied_prob'] + df['away_implied_prob']
        
        # Calculate true probabilities adjusted for overround
        df['true_home_prob'] = df['home_implied_prob'] / df['overround']
        df['true_draw_prob'] = df['draw_implied_prob'] / df['overround']
        df['true_away_prob'] = df['away_implied_prob'] / df['overround']
        
        # Add additional features if missing
        if 'win_form_diff' not in df.columns and 'home_recent_wins' in df.columns and 'away_recent_wins' in df.columns:
            df['win_form_diff'] = df['home_recent_wins'] - df['away_recent_wins']
            
        if 'goal_form_diff' not in df.columns and 'home_recent_goals' in df.columns and 'away_recent_goals' in df.columns:
            df['goal_form_diff'] = df['home_recent_goals'] - df['away_recent_goals']
            
        if 'rank_diff' not in df.columns and 'home_team_rank' in df.columns and 'away_team_rank' in df.columns:
            df['rank_diff'] = df['away_team_rank'] - df['home_team_rank']  # Lower rank is better (1 is best)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise e

# Set page configuration
st.set_page_config(
    page_title="Big Data Bet Sports Market",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved styling
st.markdown("""
<style>
    /* Base colors */
    :root {
        --primary-color: #18A558;     /* Verde esportivo */
        --secondary-color: #1E2761;   /* Azul escuro */
        --accent-color: #F15A24;      /* Laranja para destaque */
        --background-color: #1F1F1F;  /* Fundo escuro */
        --text-color: #FFFFFF;        /* Texto branco */
        --card-background: #2D2D2D;   /* Fundo de cartões */
        --light-gray: #EBEBEB;        /* Cinza claro para textos secundários */
    }

    /* Main structure styling */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid var(--primary-color);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-color);
    }
    
    /* Main title styling */
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        color: var(--text-color);
        background-color: var(--secondary-color);
        padding: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 2.5em;
        letter-spacing: 2px;
        margin-bottom: 10px;
        border-radius: 5px;
        border-left: 5px solid var(--primary-color);
    }
    
    /* Subtitle styling */
    .subtitle {
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 20px;
        font-style: italic;
        font-weight: 500;
        font-size: 1.2em;
    }
    
    /* Container for logo and title */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: var(--secondary-color);
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Graph in logo styling */
    .graph-line {
        color: var(--primary-color);
        font-size: 24px;
    }
    
    /* Section title styling */
    .section-title {
        background-color: var(--primary-color);
        color: var(--text-color);
        padding: 12px 15px;
        border-radius: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
        font-weight: bold;
        font-size: 1.3em;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Card for key metrics */
    .metric-card {
        background-color: var(--card-background);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        border-left: 3px solid var(--primary-color);
        margin: 10px 0;
    }
    
    /* Value in metric card */
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: var(--primary-color);
    }
    
    /* Label in metric card */
    .metric-label {
        font-size: 14px;
        color: var(--light-gray);
        margin-top: 5px;
    }
    
    /* Custom button styling */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #128A48; /* Verde mais escuro no hover */
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Widget labels */
    .stSelectbox label, .stSlider label, .stFileUploader label, .stMultiselect label {
        color: var(--light-gray) !important;
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        background-color: #3D3D3D !important;
        color: var(--text-color) !important;
        border: 1px solid #444 !important;
    }
    
    /* Multiselect dropdown */
    .stMultiselect>div>div>div {
        background-color: #3D3D3D !important;
        color: var(--text-color) !important;
        border: 1px solid #444 !important;
    }
    
    /* Select dropdown */
    .stSelectbox>div>div>div {
        background-color: #3D3D3D !important;
        color: var(--text-color) !important;
        border: 1px solid #444 !important;
    }
    
    /* Dataframe/table styling */
    .stDataFrame {
        background-color: var(--card-background);
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #444;
    }
    
    .stDataFrame [data-testid="StyledDataFrameDataCell"] {
        color: var(--text-color);
    }
    
    .stDataFrame [data-testid="StyledDataFrameRowHeader"] {
        color: var(--text-color);
    }
    
    .stDataFrame [data-testid="StyledDataFrameColumnHeader"] {
        color: var(--primary-color);
        background-color: #222;
    }
    
    /* Tab styling */
    button[data-baseweb="tab"] {
        background-color: var(--card-background);
        color: var(--light-gray);
        border-radius: 4px 4px 0 0;
        margin-right: 2px;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--primary-color);
        color: var(--text-color);
    }
    
    /* Charts and visualization containers */
    [data-testid="stPlotlyChart"], .element-container iframe {
        background-color: var(--card-background);
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #444;
    }
    
    /* Info containers */
    div.stInfo {
        background-color: var(--card-background);
        border-left: 5px solid var(--accent-color);
    }
    
    /* Success containers */
    div.stSuccess {
        background-color: var(--card-background);
        border-left: 5px solid var(--primary-color);
    }

    /* Text elements */
    h1, h2, h3, h4, h5, h6, p {
        color: var(--text-color);
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: var(--card-background);
        padding: 15px;
        border-radius: 5px;
        border: 1px dashed var(--primary-color);
    }
    
    /* Radio buttons */
    .stRadio > div {
        padding: 10px !important;
        background-color: var(--card-background) !important;
        border-radius: 5px !important;
    }
    
    .stRadio label {
        color: var(--text-color) !important;
    }
    
    /* Navigation item in sidebar styling */
    .sidebar-item {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: var(--card-background);
        border-left: 3px solid transparent;
        transition: all 0.2s ease;
    }
    
    .sidebar-item:hover, .sidebar-item.active {
        background-color: #3D3D3D;
        border-left: 3px solid var(--primary-color);
    }
    
    /* Stats and indicators */
    .stat-indicator {
        display: flex;
        align-items: center;
        margin: 5px 0;
    }
    
    .stat-indicator .icon {
        color: var(--primary-color);
        margin-right: 10px;
        font-size: 20px;
    }
    
    .stat-indicator .value {
        font-weight: bold;
        font-size: 16px;
    }
    
    .stat-indicator .label {
        margin-left: 10px;
        color: var(--light-gray);
        font-size: 14px;
    }
    
    /* Empty state styling */
    .empty-state {
        background-color: var(--card-background);
        padding: 40px;
        border-radius: 10px;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border: 1px dashed #444;
    }
    
    .empty-state h3 {
        color: var(--primary-color);
        margin-bottom: 15px;
    }
    
    .empty-state p {
        color: var(--light-gray);
        font-size: 16px;
        line-height: 1.5;
    }
    
    .empty-state .icon {
        font-size: 60px;
        color: var(--primary-color);
        margin: 20px 0;
        opacity: 0.8;
    }
    
    /* Metric formatting */
    [data-testid="stMetric"] {
        background-color: var(--card-background);
        padding: 15px;
        border-radius: 8px;
        border-left: 3px solid var(--primary-color);
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }
    
    [data-testid="stMetric"] > div {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    [data-testid="stMetric"] label {
        color: var(--light-gray) !important;
        font-size: 1em !important;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--primary-color) !important;
        font-size: 1.8em !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: var(--accent-color) !important;
    }
    
    /* Upload area styling */
    [data-testid="stFileUploader"] > section {
        background-color: rgba(24, 165, 88, 0.05) !important;
        border: 2px dashed var(--primary-color) !important;
        padding: 30px !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] p {
        color: var(--text-color) !important;
        font-weight: 500 !important;
    }
    
    /* Animation for transitions */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .st-emotion-cache-1mnsv9e {
        animation: fadeIn 0.5s ease-in-out;
    }
    
    /* Estilização de card informativo */
    .info-card {
        background-color: #1A1A1A !important; 
        border-left: 5px solid #18A558 !important; 
        padding: 15px !important; 
        margin: 15px 0 !important; 
        border-radius: 5px !important;
        animation: fadeIn 0.8s ease-in !important;
        color: #FFFFFF !important; /* Garantir que o texto seja branco */
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }
    
    .info-card h4 {
        color: #18A558 !important; 
        margin-top: 0 !important;
        font-weight: bold !important;
    }
    
    .info-card p {
        color: #FFFFFF !important; /* Garantir que o texto do parágrafo seja branco */
        margin-bottom: 10px !important;
    }
    
    .info-card strong {
        color: #18A558 !important; /* Destacar o texto em negrito com a cor da marca */
        font-weight: bold !important;
    }
    
    .info-card .tip-box {
        background-color: #1E3A8A !important; 
        padding: 12px !important; 
        border-radius: 5px !important;
        margin-top: 10px !important;
    }
    
    .info-card .tip-box p {
        color: #FFFFFF !important; /* Garantir que o texto na caixa de dica seja branco */
        margin: 0 !important;
        font-weight: 500 !important;
    }
    
    .info-card .tip-box p span {
        color: #FFFFFF !important; /* TODOS os spans dentro da caixa de dica serão brancos */
    }
    
    .info-card .tip-box .dica-label,
    .info-card .tip-box span:first-child {
        color: #FFEB3B !important; /* Amarelo brilhante APENAS para o label da dica */
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# Logo and Header
def render_logo():
    # Logo com design moderno e impactante
    st.markdown("""
    <div style="text-align: center; padding: 30px 0; margin-bottom: 30px; position: relative; overflow: hidden;">
        <!-- Gradiente de fundo animado -->
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(135deg, #121212, #1E3A8A, #1E1E1E, #18A558); 
             background-size: 400% 400%; z-index: -1; opacity: 0.7; animation: gradient 15s ease infinite; border-radius: 15px;">
        </div>
        
        <!-- Padrão de grid geométrico para dar profundidade -->
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
             background-image: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px); 
             background-size: 20px 20px; z-index: -1; opacity: 0.3;"></div>
        
        <!-- Título principal com efeito de sombra -->
        <h1 style="font-size: 3.2em; font-weight: 800; margin: 0; color: #FFFFFF; text-shadow: 0 4px 8px rgba(0,0,0,0.3); 
                 letter-spacing: 2px; position: relative; display: inline-block;">
            <span style="background: linear-gradient(90deg, #FFFFFF, #18A558); -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                  position: relative; z-index: 2;">BIG DATA BET</span>
        </h1>
        
        <!-- Subtítulo com linha decorativa -->
        <div style="text-align: center; color: #FFFFFF; font-size: 1.5em; font-weight: 300; margin-top: 10px; position: relative;">
            <span style="position: relative; padding: 0 10px; z-index: 2;">SPORTS MARKET ANALYSIS</span>
        </div>
        
        <!-- Linha decorativa animada -->
        <div style="width: 80%; height: 3px; margin: 25px auto; position: relative; overflow: hidden; border-radius: 3px;">
            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
                 background: linear-gradient(90deg, transparent, #18A558, #1E3A8A, #18A558, transparent); 
                 background-size: 200% 100%; animation: shimmer 3s infinite; z-index: 1;"></div>
        </div>
        
        <!-- Texto descritivo com ícones -->
        <div style="display: flex; justify-content: center; align-items: center; margin-top: 15px; flex-wrap: wrap; gap: 20px;">
            <div style="display: flex; align-items: center; background-color: rgba(24, 165, 88, 0.1); padding: 8px 15px; border-radius: 50px; backdrop-filter: blur(5px);">
                <span style="margin-right: 8px; font-size: 1.2em;">🎲</span>
                <span style="color: #FFFFFF; font-weight: 500;">Modelo Preditivo</span>
            </div>
            <div style="display: flex; align-items: center; background-color: rgba(30, 58, 138, 0.1); padding: 8px 15px; border-radius: 50px; backdrop-filter: blur(5px);">
                <span style="margin-right: 8px; font-size: 1.2em;">📊</span>
                <span style="color: #FFFFFF; font-weight: 500;">Análise de Dados</span>
            </div>
            <div style="display: flex; align-items: center; background-color: rgba(232, 72, 85, 0.1); padding: 8px 15px; border-radius: 50px; backdrop-filter: blur(5px);">
                <span style="margin-right: 8px; font-size: 1.2em;">🔍</span>
                <span style="color: #FFFFFF; font-weight: 500;">Explicabilidade ML</span>
            </div>
        </div>
    </div>
    
    <style>
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    @keyframes shimmer {
        0% {background-position: 200% 0;}
        100% {background-position: -200% 0;}
    }
    </style>
    """, unsafe_allow_html=True)

# Render the logo at the top of the page
render_logo()

# Aplica o CSS personalizado para garantir que todos os textos sejam brancos
apply_custom_css()

##############################################
# Function to Generate Report for Download (HTML)
##############################################
def generate_report(result_df, metrics, report_text, fig_list):
    buffer = BytesIO()
    content = "<h1>Sports Betting Analysis Report</h1>"
    content += "<h2>Model Comparison</h2>" + result_df.to_html()
    content += "<h2>Additional Metrics</h2><ul>"
    for k, v in metrics.items():
        content += f"<li>{k}: {v:.3f}</li>"
    content += "</ul>" + report_text
    # Save figures to report as images (converted to base64)
    for idx, fig in enumerate(fig_list):
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches="tight")
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        content += f"<h3>Figure {idx+1}</h3><img src='data:image/png;base64,{img_str}' />"
    buffer.write(content.encode('utf-8'))
    return buffer

##############################################
# Utility: Record explanation history for auditing
##############################################
def record_explanation(match_id, explanation):
    if 'explanation_history' not in st.session_state:
        st.session_state['explanation_history'] = {}
    st.session_state['explanation_history'][match_id] = explanation

##############################################
# Sidebar Navigation
##############################################
st.sidebar.markdown("""
<div style='background-color: var(--secondary-color); padding: 15px; border-radius: 5px; margin-bottom: 20px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.2);'>
    <h2 style='color: var(--text-color); margin: 0;'>Navigation</h2>
    <div style='width: 50px; height: 3px; background-color: var(--primary-color); margin: 10px auto;'></div>
</div>
""", unsafe_allow_html=True)

# Style for sidebar items
sidebar_style = """
<style>
    div[data-testid="stRadio"] > div {
        background-color: var(--card-background);
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        border-left: 3px solid transparent;
        transition: all 0.2s ease;
    }
    div[data-testid="stRadio"] > div:hover {
        background-color: #3D3D3D;
        border-left: 3px solid var(--primary-color);
    }
    div[data-testid="stRadio"] label {
        color: var(--text-color) !important;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        background-color: var(--primary-color);
        border-color: var(--primary-color);
    }
</style>
"""
st.sidebar.markdown(sidebar_style, unsafe_allow_html=True)

# Add a divider
st.sidebar.markdown("<hr style='height:3px;border:none;background-color:var(--primary-color);margin:20px 0;opacity:0.5;'>", unsafe_allow_html=True)

# Main navigation
page = st.sidebar.radio("", 
                      ("Data Upload & EDA", 
                       "Modeling & Performance", 
                       "Advanced Visualizations", 
                       "Advanced Feature Engineering", 
                       "Dynamic Hyperparameter Tuning", 
                       "Advanced Explainability", 
                       "Interactive Explainability & Model Comparison",
                       "Export Report", 
                       "Advanced What-if Simulation",
                       "Advanced Interpretability Visualizations",
                       "Additional Evaluation Metrics",
                       "ROI Analysis & Simulations",
                       "Bankroll Simulator",
                       "Arbitrage Calculator",
                       "Value Bet Detector",
                       "Relatório",
                       "Virtual Assistant",
                       "Sobre"))

# Add legend with information about the section
section_info = {
    "Data Upload & EDA": "Upload and explore your sports betting data",
    "Modeling & Performance": "Train and evaluate prediction models",
    "Advanced Visualizations": "Visualize data and model results",
    "Advanced Feature Engineering": "Create and select features for modeling",
    "Dynamic Hyperparameter Tuning": "Tune model parameters for better performance",
    "Advanced Explainability": "Understand model predictions",
    "Interactive Explainability & Model Comparison": "Compare different models",
    "Export Report": "Generate and download analysis reports",
    "Advanced What-if Simulation": "Simulate different betting scenarios",
    "Advanced Interpretability Visualizations": "Visualize model interpretation",
    "Additional Evaluation Metrics": "Explore additional model metrics",
    "ROI Analysis & Simulations": "Analyze return on investment and run betting simulations",
    "Bankroll Simulator": "Simulate bankroll growth with different strategies",
    "Arbitrage Calculator": "Find and calculate arbitrage opportunities",
    "Value Bet Detector": "Identify value bets with positive expected value",
    "Relatório": "Automated AI analysis of all visualizations and data",
    "Virtual Assistant": "Ask questions about your betting data",
    "Sobre": "About the project"
}

# Display section information
st.sidebar.markdown(f"""
<div style='background-color: var(--card-background); padding:15px; border-radius:8px; margin-top:20px; border-left: 3px solid var(--accent-color);'>
    <span style='color:var(--accent-color); font-weight:bold;'>Current Section:</span><br/>
    <span style='color:var(--text-color);'>{section_info.get(page, "")}</span>
</div>
""", unsafe_allow_html=True)

# Add metrics at the bottom of sidebar if data is available
if 'data' in st.session_state:
    st.sidebar.markdown("<hr style='height:3px;border:none;background-color:var(--primary-color);margin:20px 0;opacity:0.5;'>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='text-align:center;'><span style='color:var(--primary-color);font-weight:bold;'>Session Stats</span></div>", unsafe_allow_html=True)
    
    # Add some metrics
    data = st.session_state['data']
    total_matches = len(data)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{total_matches}</div>
            <div class='metric-label'>Matches</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add more metrics if available
    if 'match_outcome' in data.columns:
        home_wins = (data['match_outcome'] == 1).sum()
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{home_wins}</div>
                <div class='metric-label'>Home Wins</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Adicionar mais métricas em uma nova linha
        col3, col4 = st.sidebar.columns(2)
        draws = (data['match_outcome'] == 0).sum()
        away_wins = (data['match_outcome'] == 2).sum()
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{draws}</div>
                <div class='metric-label'>Draws</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{away_wins}</div>
                <div class='metric-label'>Away Wins</div>
            </div>
            """, unsafe_allow_html=True)

# Add credit at the bottom
st.sidebar.markdown("""
<div style='position:fixed;bottom:0;left:0;width:100%;background-color:var(--secondary-color);padding:10px;text-align:center;color:var(--text-color);font-size:12px;border-top: 2px solid var(--primary-color);'>
    © 2024 Big Data Bet | Sports Market Analysis
</div>
""", unsafe_allow_html=True)

##############################################
# Section 1: Data Upload & EDA
##############################################
if page == "Data Upload & EDA":
    # Section title with improved custom styling
    st.markdown("""
    <div class="section-header">
        <h1 style="margin: 0; padding: 0; font-size: 1.8em; font-weight: 600;">
            Data Upload & Exploratory Data Analysis
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a nicer container for the file upload with improved visual design
    st.markdown("""
    <div class="upload-container">
        <div style="display: flex; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #444; padding-bottom: 15px;">
            <div style="background-color: #18A558; color: white; padding: 8px; 
                 border-radius: 8px; margin-right: 15px; font-weight: bold; font-size: 1.2em;">
                01
            </div>
            <h2 style="color: #18A558; margin: 0; font-size: 1.6em; font-weight: 600;">
                Upload Your Data
            </h2>
        </div>
        
        <p style="color: #FFFFFF; margin: 15px 0; line-height: 1.6; font-size: 1.05em;">
            Upload your CSV or Excel file containing sports betting data to begin your analysis journey. 
            Our platform will help you uncover patterns and insights to improve your betting strategy.
        </p>
        
        <div style="background-color: rgba(24, 165, 88, 0.08); border-left: 4px solid #18A558; 
             padding: 15px; margin: 20px 0; border-radius: 0 8px 8px 0;">
            <div style="color: #18A558; font-weight: 600; margin-bottom: 8px; font-size: 1.1em;">
                Required Columns
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px;">
                <span style="background-color: #333333; padding: 6px 12px; border-radius: 4px; color: #18A558; font-family: monospace;">match_id</span>
                <span style="background-color: #333333; padding: 6px 12px; border-radius: 4px; color: #18A558; font-family: monospace;">match_outcome</span>
                <span style="background-color: #333333; padding: 6px 12px; border-radius: 4px; color: #18A558; font-family: monospace;">home_win_odds</span>
                <span style="background-color: #333333; padding: 6px 12px; border-radius: 4px; color: #18A558; font-family: monospace;">draw_odds</span>
                <span style="background-color: #333333; padding: 6px 12px; border-radius: 4px; color: #18A558; font-family: monospace;">away_win_odds</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for upload options with better styling
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls", "xlsm"], label_visibility="visible")
    
    with col2:
        st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)  # Add some spacing
        sample_data_button = st.button("Use Sample Data", key="sample_data_btn", help="Load our sample dataset with 100 sports betting matches")
    
    # Handle file upload or sample data
    if uploaded_file is not None:
        if uploaded_file.getbuffer().nbytes == 0:
            st.error("The file is empty. Please upload a valid file.")
        else:
            # Show a spinner while loading
            with st.spinner("Processing your data..."):
                data = load_data(uploaded_file)
                
            if data is not None:
                # Success message with animation
                st.markdown("""
                <div style="background-color: rgba(24, 165, 88, 0.1); border-left: 4px solid var(--primary-color); 
                     padding: 16px; margin: 20px 0; border-radius: 0 8px 8px 0; display: flex; align-items: center;">
                    <div style="background-color: var(--primary-color); color: white; padding: 8px; 
                         border-radius: 50%; margin-right: 15px; font-weight: bold; font-size: 1.2em;">
                        ✓
                    </div>
                    <div>
                        <div style="color: var(--primary-color); font-weight: 600; font-size: 1.1em;">
                            Success
                        </div>
                        <div style="color: var(--text-color); margin-top: 5px;">
                            Data loaded successfully. You can now explore and analyze your betting data.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Data overview in a nice card with improved styling
                st.markdown("""
                <div style="background-color: var(--card-background); padding: 25px; 
                     border-radius: 10px; margin: 25px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                     
                    <div style="display: flex; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #444; padding-bottom: 15px;">
                        <div style="background-color: var(--accent-color); color: white; padding: 8px; 
                             border-radius: 8px; margin-right: 15px; font-weight: bold; font-size: 1.2em;">
                            02
                        </div>
                        <h2 style="color: var(--accent-color); margin: 0; font-size: 1.6em; font-weight: 600;">
                            Data Preview
                        </h2>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Improved dataframe display with hover styles
                st.markdown("""
                <style>
                    .dataframe-container {
                        border-radius: 8px;
                        overflow: hidden;
                        border: 1px solid #444;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    }
                    .dataframe-container [data-testid="stDataFrame"] {
                        border: none !important;
                    }
                </style>
                <div class="dataframe-container">
                """, unsafe_allow_html=True)
                
                st.dataframe(data.head(), use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Key stats with improved pure HTML/CSS card layout
                st.markdown("""
                <div style="background-color: var(--card-background); padding: 25px; 
                     border-radius: 10px; margin: 25px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                     
                    <div style="display: flex; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #444; padding-bottom: 15px;">
                        <div style="background-color: var(--primary-color); color: white; padding: 8px; 
                             border-radius: 8px; margin-right: 15px; font-weight: bold; font-size: 1.2em;">
                            03
                        </div>
                        <h2 style="color: var(--primary-color); margin: 0; font-size: 1.6em; font-weight: 600;">
                            Key Statistics
                        </h2>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a card grid for the metrics with pure CSS/HTML
                st.markdown("""
                <style>
                    .stats-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                        gap: 20px;
                        margin-bottom: 30px;
                    }
                    
                    .stat-card {
                        background-color: var(--card-background);
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                        border-left: 4px solid var(--primary-color);
                        transition: transform 0.2s ease, box-shadow 0.2s ease;
                    }
                    
                    .stat-card:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
                    }
                    
                    .stat-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 15px;
                        padding-bottom: 10px;
                        border-bottom: 1px solid rgba(255,255,255,0.1);
                    }
                    
                    .stat-title {
                        color: var(--light-gray);
                        font-size: 0.9em;
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    }
                    
                    .stat-badge {
                        background-color: rgba(24, 165, 88, 0.1);
                        color: var(--primary-color);
                        padding: 5px 10px;
                        border-radius: 12px;
                        font-size: 0.8em;
                        font-weight: 600;
                    }
                    
                    .stat-value {
                        font-size: 2.4em;
                        font-weight: 700;
                        color: var(--text-color);
                        margin: 0;
                        line-height: 1;
                    }
                    
                    .stat-trend {
                        display: flex;
                        align-items: center;
                        margin-top: 10px;
                    }
                    
                    .trend-positive {
                        color: var(--primary-color);
                        font-weight: 600;
                        font-size: 0.9em;
                    }
                    
                    .trend-negative {
                        color: var(--accent-color);
                        font-weight: 600;
                        font-size: 0.9em;
                    }
                    
                    .trend-arrow {
                        margin-right: 5px;
                    }
                </style>
                
                <div class="stats-grid">
                """, unsafe_allow_html=True)
                
                # Total Matches metric
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-header">
                        <div class="stat-title">Total Matches</div>
                        <div class="stat-badge">ALL DATA</div>
                    </div>
                    <div class="stat-value">{len(data)}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Home Wins metric
                if 'match_outcome' in data.columns:
                    home_wins = (data['match_outcome'] == 1).sum()
                    home_win_pct = home_wins/len(data)*100
                    
                    st.markdown(f"""
                    <div class="stat-card" style="border-left-color: #18A558;">
                        <div class="stat-header">
                            <div class="stat-title">Home Wins</div>
                            <div class="stat-badge" style="background-color: rgba(24, 165, 88, 0.1); color: #18A558;">HOME</div>
                        </div>
                        <div class="stat-value" style="color: #18A558;">{home_wins}</div>
                        <div class="stat-trend">
                            <span class="trend-positive">
                                <span class="trend-arrow">&#x25B2;</span> {home_win_pct:.1f}% of matches
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Draws metric
                if 'match_outcome' in data.columns:
                    draws = (data['match_outcome'] == 0).sum()
                    draw_pct = draws/len(data)*100
                    
                    st.markdown(f"""
                    <div class="stat-card" style="border-left-color: #5D8BF4;">
                        <div class="stat-header">
                            <div class="stat-title">Draws</div>
                            <div class="stat-badge" style="background-color: rgba(93, 139, 244, 0.1); color: #5D8BF4;">DRAW</div>
                        </div>
                        <div class="stat-value" style="color: #5D8BF4;">{draws}</div>
                        <div class="stat-trend">
                            <span class="trend-positive" style="color: #5D8BF4;">
                                <span class="trend-arrow">=</span> {draw_pct:.1f}% of matches
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Away Wins metric
                if 'match_outcome' in data.columns:
                    away_wins = (data['match_outcome'] == 2).sum()
                    away_win_pct = away_wins/len(data)*100
                    
                    st.markdown(f"""
                    <div class="stat-card" style="border-left-color: #FF5252;">
                        <div class="stat-header">
                            <div class="stat-title">Away Wins</div>
                            <div class="stat-badge" style="background-color: rgba(255, 82, 82, 0.1); color: #FF5252;">AWAY</div>
                        </div>
                        <div class="stat-value" style="color: #FF5252;">{away_wins}</div>
                        <div class="stat-trend">
                            <span class="trend-positive" style="color: #FF5252;">
                                <span class="trend-arrow">&#x25B2;</span> {away_win_pct:.1f}% of matches
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add a metric for average odds if available
                if 'home_win_odds' in data.columns:
                    avg_home_odds = data['home_win_odds'].mean()
                    
                    st.markdown(f"""
                    <div class="stat-card" style="border-left-color: #9370DB;">
                        <div class="stat-header">
                            <div class="stat-title">Avg Home Win Odds</div>
                            <div class="stat-badge" style="background-color: rgba(147, 112, 219, 0.1); color: #9370DB;">ODDS</div>
                        </div>
                        <div class="stat-value" style="color: #9370DB;">{avg_home_odds:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Close the stats grid
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Optional dynamic filters in an improved container
                st.markdown("""
                <div style="background-color: var(--card-background); padding: 20px; border-radius: 10px; 
                         border-left: 5px solid var(--accent-color); margin: 25px 0; 
                         box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                    <h3 style="color: var(--accent-color); margin-top: 0; display: flex; align-items: center;">
                        <span style="background-color: var(--accent-color); color: white; border-radius: 50%; width: 30px; height: 30px; 
                              display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">
                            <span style="font-size: 16px;">🔍</span>
                        </span>
                        Filter Data
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                filter_cols = st.columns([1, 1])
                
                with filter_cols[0]:
                    if 'league' in data.columns:
                        leagues = ['All Leagues'] + sorted(data['league'].unique().tolist())
                        selected_league = st.selectbox("Filter by League", leagues, label_visibility="visible")
                        if selected_league != 'All Leagues':
                            data = data[data['league'] == selected_league]
                
                with filter_cols[1]:
                    if 'home_team_rank' in data.columns:
                        min_rank = int(data['home_team_rank'].min())
                        max_rank = int(data['home_team_rank'].max())
                        rank_range = st.slider("Filter by Home Team Rank", min_rank, max_rank, (min_rank, max_rank), label_visibility="visible")
                        data = data[(data['home_team_rank'] >= rank_range[0]) & (data['home_team_rank'] <= rank_range[1])]
                
                # Visualizations with tabs
                st.markdown("""
                <div style="background-color: var(--card-background); padding: 25px; 
                     border-radius: 10px; margin: 25px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                     
                    <div style="display: flex; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #444; padding-bottom: 15px;">
                        <div style="background-color: var(--primary-color); color: white; padding: 8px; 
                             border-radius: 8px; margin-right: 15px; font-weight: bold; font-size: 1.2em;">
                            04
                        </div>
                        <h2 style="color: var(--primary-color); margin: 0; font-size: 1.6em; font-weight: 600;">
                            Exploratory Visualizations
                        </h2>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Custom CSS for better tab styling without SVG elements
                st.markdown("""
                <style>
                    /* Custom tabs styling */
                    .stTabs [data-baseweb="tab-list"] {
                        gap: 8px;
                        margin-bottom: 15px;
                        background-color: #252525;
                        padding: 10px;
                        border-radius: 8px;
                    }
                    
                    .stTabs [data-baseweb="tab"] {
                        background-color: #2D2D2D !important;
                        border-radius: 6px !important;
                        padding: 10px 20px !important;
                        border: none !important;
                        color: #EBEBEB !important;
                        font-weight: 500 !important;
                        transition: all 0.2s ease;
                    }
                    
                    .stTabs [data-baseweb="tab"][aria-selected="true"] {
                        background-color: #18A558 !important;
                        color: white !important;
                        font-weight: 600 !important;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                    }
                    
                    .stTabs [data-baseweb="tab"]:hover {
                        background-color: #3D3D3D !important;
                        color: white !important;
                    }
                    
                    .stTabs [data-baseweb="tab-panel"] {
                        background-color: var(--card-background);
                        border-radius: 8px;
                        padding: 20px;
                        border: 1px solid #444;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                    }
                    
                    /* Graph container styling */
                    .graph-container {
                        padding: 20px;
                        background-color: #2A2A2A;
                        border-radius: 8px;
                        margin: 10px 0 20px 0;
                        border: 1px solid #444;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                    }
                    
                    /* Section heading styling */
                    .viz-section-heading {
                        color: var(--primary-color);
                        font-size: 1.4em;
                        font-weight: 600;
                        margin: 5px 0 20px 0;
                        padding-bottom: 10px;
                        border-bottom: 2px solid rgba(24, 165, 88, 0.3);
                    }
                    
                    /* Insights container styling */
                    .insights-container {
                        background-color: rgba(24, 165, 88, 0.05);
                        border-left: 4px solid var(--primary-color);
                        padding: 15px 20px;
                        border-radius: 0 8px 8px 0;
                        margin: 20px 0;
                    }
                    
                    .insights-title {
                        color: var(--primary-color);
                        font-weight: 600;
                        font-size: 1.1em;
                        margin-bottom: 10px;
                    }
                    
                    .insights-list {
                        color: var(--text-color);
                        margin: 0;
                        padding-left: 20px;
                    }
                    
                    .insights-list li {
                        margin-bottom: 8px;
                    }
                    
                    .insights-list li b {
                        color: var(--primary-color);
                    }
                </style>
                """, unsafe_allow_html=True)
                
                tabs = st.tabs(["Match Outcomes", "Odds Analysis", "Correlations"])
                
                with tabs[0]:
                    # Match outcome distribution
                    if 'match_outcome' in data.columns:
                        st.markdown("<div class='viz-section-heading'>Match Outcome Distribution</div>", unsafe_allow_html=True)
                        
                        # Container for the graph
                        st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
                        
                        outcome_counts = data['match_outcome'].value_counts().reset_index()
                        outcome_counts.columns = ['Outcome', 'Count']
                        outcome_map = {1: 'Home Win', 0: 'Draw', 2: 'Away Win'}
                        outcome_counts['Outcome'] = outcome_counts['Outcome'].map(outcome_map)
                        
                        # Get colors from theme for consistency
                        colors = get_theme_colors()
                        custom_colors = [colors['home_win'], colors['draw'], colors['away_win']]
                        
                        fig = px.bar(outcome_counts, x='Outcome', y='Count', 
                                    title="Match Outcome Distribution",
                                    color='Outcome', color_discrete_sequence=custom_colors)
                        
                        # Apply theme to figure
                        fig = apply_theme_to_plotly(fig)
                        
                        # Add value labels on top of bars
                        fig.update_traces(texttemplate='%{y}', textposition='outside')
                        
                        # Update the title format
                        fig.update_layout(
                            title={
                                'text': 'Match Outcome Distribution',
                                'y':0.95,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top',
                                'font': {'size': 22, 'color': colors['text']}
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Add some context and insights
                        st.markdown(f"""
                        <div class="insights-container">
                            <div class="insights-title">Distribution Insights</div>
                            <ul class="insights-list">
                                <li>Home teams won <b>{home_win_pct:.1f}%</b> of all matches</li>
                                <li>Away teams won <b>{away_win_pct:.1f}%</b> of all matches</li>
                                <li>Matches ended in a draw <b>{draw_pct:.1f}%</b> of the time</li>
                                <li>The home field advantage is <b>{round(home_win_pct - away_win_pct, 1)}%</b> (difference between home and away wins)</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                with tabs[1]:
                    # Odds vs. Outcomes visualization
                    if all(col in data.columns for col in ['home_win_odds', 'match_outcome']):
                        st.markdown("<div class='viz-section-heading'>Odds vs. Outcomes</div>", unsafe_allow_html=True)
                        
                        # Container for the graph
                        st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
                        
                        colors = get_theme_colors()
                        
                        home_odds = data[data['match_outcome'] == 1]['home_win_odds']
                        away_odds = data[data['match_outcome'] == 2]['away_win_odds']
                        draw_odds = data[data['match_outcome'] == 0]['draw_odds']
                        
                        odds_fig = go.Figure()
                        odds_fig.add_trace(go.Box(y=home_odds, name='Home Win Odds (Home Won)', marker_color=colors['home_win']))
                        odds_fig.add_trace(go.Box(y=away_odds, name='Away Win Odds (Away Won)', marker_color=colors['away_win']))
                        odds_fig.add_trace(go.Box(y=draw_odds, name='Draw Odds (Draw)', marker_color=colors['draw']))
                        
                        # Apply theme to odds figure
                        odds_fig = apply_theme_to_plotly(odds_fig)
                        odds_fig.update_layout(
                            title={
                                'text': 'Distribution of Winning Odds',
                                'y':0.95,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top',
                                'font': {'size': 22, 'color': colors['text']}
                            }
                        )
                        
                        st.plotly_chart(odds_fig, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Add a second visualization - Odds distribution
                        st.markdown("<div class='viz-section-heading'>Odds Distribution Histogram</div>", unsafe_allow_html=True)
                        
                        # Container for the second graph
                        st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
                        
                        hist_fig = go.Figure()
                        hist_fig.add_trace(go.Histogram(x=data['home_win_odds'], name='Home Win Odds', marker_color=colors['home_win'], opacity=0.7, nbinsx=20))
                        hist_fig.add_trace(go.Histogram(x=data['draw_odds'], name='Draw Odds', marker_color=colors['draw'], opacity=0.7, nbinsx=20))
                        hist_fig.add_trace(go.Histogram(x=data['away_win_odds'], name='Away Win Odds', marker_color=colors['away_win'], opacity=0.7, nbinsx=20))
                        
                        # Apply theme
                        hist_fig = apply_theme_to_plotly(hist_fig)
                        hist_fig.update_layout(
                            title={
                                'text': 'Distribution of All Odds',
                                'y':0.95,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top',
                                'font': {'size': 22, 'color': colors['text']}
                            },
                            barmode='overlay'
                        )
                        
                        st.plotly_chart(hist_fig, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Add some insights
                        avg_home_odds = data['home_win_odds'].mean()
                        avg_away_odds = data['away_win_odds'].mean()
                        avg_draw_odds = data['draw_odds'].mean()
                        
                        st.markdown(f"""
                        <div class="insights-container">
                            <div class="insights-title">Odds Insights</div>
                            <ul class="insights-list">
                                <li>Average Home Win Odds: <b>{avg_home_odds:.2f}</b></li>
                                <li>Average Draw Odds: <b>{avg_draw_odds:.2f}</b></li>
                                <li>Average Away Win Odds: <b>{avg_away_odds:.2f}</b></li>
                                <li>Lower odds generally indicate higher probability of an outcome</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                with tabs[2]:
                    # Correlation matrix for numeric features with improved styling
                    st.markdown("<div class='viz-section-heading'>Feature Correlations</div>", unsafe_allow_html=True)
                    
                    # Container for the graph
                    st.markdown("<div class='graph-container' style='padding: 25px;'>", unsafe_allow_html=True)
                    
                    numeric_data = data.select_dtypes(include=np.number)
                    # Define corr_cols here to avoid the name error
                    corr_cols = []
                    # Limit to 15 columns for better visualization
                    if numeric_data.shape[1] > 15:
                        corr_cols = ['home_win_odds', 'draw_odds', 'away_win_odds', 'home_team_rank', 
                                    'away_team_rank', 'home_recent_wins', 'away_recent_wins', 
                                    'home_recent_goals', 'away_recent_goals', 'home_possession',
                                    'away_possession', 'match_outcome', 'overround', 'true_home_prob', 'true_away_prob']
                        corr_cols = [col for col in corr_cols if col in numeric_data.columns]
                        numeric_data = numeric_data[corr_cols]
                    else:
                        # If fewer than 15 columns, use all numeric columns
                        corr_cols = numeric_data.columns.tolist()
                    
                    # Check if we have any columns to work with
                    if len(corr_cols) > 1:  # Need at least 2 columns for correlation
                        corr = numeric_data.corr()
                        mask = np.triu(np.ones_like(corr, dtype=bool))
                        
                        # Set style for dark theme
                        plt.style.use('dark_background')
                        
                        # Use a custom colormap that fits the theme
                        colors = get_theme_colors()
                        custom_cmap = sns.diverging_palette(200, 110, s=75, l=40, as_cmap=True)
                        
                        fig, ax = plt.subplots(figsize=(12, 10), facecolor=colors['background'])
                        sns.heatmap(
                            corr, 
                            mask=mask,
                            annot=True, 
                            cmap=custom_cmap, 
                            ax=ax, 
                            fmt=".2f", 
                            linewidths=0.8,
                            cbar_kws={"shrink": .8},
                            annot_kws={"size": 10, "color": "white"}
                        )
                        
                        # Improve heatmap appearance
                        ax.set_facecolor(colors['background'])
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', color='white')
                        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color='white')
                        ax.set_title('Correlation Matrix of Numeric Features', fontsize=16, pad=20, color='white')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Add correlation insights
                        if 'match_outcome' in corr_cols:
                            # Find top correlations with match_outcome
                            outcome_corr = corr['match_outcome'].drop('match_outcome').sort_values(ascending=False)
                            top_pos = outcome_corr.head(3)
                            top_neg = outcome_corr.tail(3)
                            
                            st.markdown(f"""
                            <div class="insights-container">
                                <div class="insights-title">Correlation Insights</div>
                                <p style="color: var(--text-color); margin-bottom: 10px;">
                                    The correlation matrix shows relationships between different variables. Values closer to 1 or -1 indicate stronger relationships.
                                </p>
                                <h5 style="color: var(--text-color); margin: 10px 0;">Top Positive Correlations with Match Outcome:</h5>
                                <ul class="insights-list">
                                    {' '.join([f"<li><b>{col}</b>: {val:.2f}</li>" for col, val in top_pos.items()])}
                                </ul>
                                <h5 style="color: var(--text-color); margin: 10px 0;">Top Negative Correlations with Match Outcome:</h5>
                                <ul class="insights-list">
                                    {' '.join([f"<li><b>{col}</b>: {val:.2f}</li>" for col, val in top_neg.items()])}
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Not enough numeric columns to create a correlation matrix. Please ensure your data has multiple numeric columns.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Store data in session state for other pages
                st.session_state['data'] = data
                
                # Add download button with improved styling - no SVG
                st.markdown("""
                <div style="background-color: var(--card-background); padding: 25px; 
                     border-radius: 10px; margin: 30px 0; text-align: center; 
                     box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                    
                    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
                        <div style="background-color: var(--accent-color); color: white; padding: 8px; 
                             border-radius: 8px; margin-right: 15px; font-weight: bold; font-size: 1.2em;">
                            05
                        </div>
                        <h2 style="color: var(--accent-color); margin: 0; font-size: 1.6em; font-weight: 600;">
                            Download Processed Data
                        </h2>
                    </div>
                    
                    <p style="color: var(--text-color); margin: 15px 0;">
                        Download your processed sports betting data as a CSV file for further analysis in other tools.
                    </p>
                </div>
                """, unsafe_allow_html=True)

##############################################
# Section 2: Modeling & Performance
##############################################
if page == "Modeling & Performance":
    st.title("Modeling & Performance")
    if 'data' not in st.session_state:
        st.error("Please upload the data in the 'Data Upload & EDA' section first!")
    else:
        data = st.session_state['data']
        
        # Feature selection interface
        st.subheader("Feature Selection")
        
        # Get list of numeric features
        numeric_features = data.select_dtypes(include=np.number).columns.tolist()
        # Exclude match_id, any ID columns, and the target variable
        exclude_cols = ['match_id', 'match_outcome']
        feature_options = [f for f in numeric_features if f not in exclude_cols]
        
        # Let user select features
        selected_features = st.multiselect(
            "Select features for modeling", 
            options=feature_options,
            default=[f for f in ['home_win_odds', 'away_win_odds', 'draw_odds', 
                                'home_team_rank', 'away_team_rank', 
                                'home_recent_wins', 'away_recent_wins',
                                'home_recent_goals', 'away_recent_goals',
                                'win_form_diff', 'goal_form_diff', 'rank_diff'] 
                     if f in feature_options],
            label_visibility="visible"
        )
        
        if not selected_features:
            st.warning("Please select at least one feature for modeling.")
            st.stop()
        
        # Set target variable
        target_variable = 'match_outcome'
        if target_variable not in data.columns:
            st.error(f"Required column '{target_variable}' not found in the data.")
            st.stop()
        
        # Prepare features and target
        try:
            X = data[selected_features]
            y = data[target_variable]
            
            # Display class distribution
            st.subheader("Class Distribution")
            class_dist = y.value_counts().reset_index()
            class_dist.columns = ['Outcome', 'Count']
            class_map = {1: 'Home Win', 0: 'Draw', 2: 'Away Win'}
            class_dist['Outcome'] = class_dist['Outcome'].map(class_map)
            st.write(class_dist)
            
            # Balance options
            st.subheader("Data Balancing Options")
            balance_method = st.radio(
                "Select data balancing method:",
                ["None", "SMOTE (Synthetic Minority Over-sampling)"],
                label_visibility="visible"
            )
            
            # Train-test split
            test_size = st.slider("Test Size (% of data)", 10, 40, 20, label_visibility="visible") / 100
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            
            # Apply balancing if selected
            if balance_method == "SMOTE (Synthetic Minority Over-sampling)":
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                st.success(f"Applied SMOTE balancing. New training data shape: {X_train.shape}")
            
            # Dynamic parameter tuning for RandomForest
            st.sidebar.subheader("RandomForest Parameters")
            n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 200, step=50)
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
            
            # Define models
            models = {
                "XGBoost": xgb.XGBClassifier(objective='multi:softprob', num_class=3, 
                                            use_label_encoder=False, random_state=42),
                "RandomForest": RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
                "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'),
                "SVC": SVC(probability=True, random_state=42),
                "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
                "GaussianNB": GaussianNB()
            }
            
            # Model selection
            selected_models = st.multiselect(
                "Select models to evaluate",
                list(models.keys()),
                default=["RandomForest", "XGBoost"],
                label_visibility="visible"
            )
            
            if not selected_models:
                st.warning("Please select at least one model for evaluation.")
                st.stop()
            
            # Save models in session_state for later use
            st.session_state['models'] = {}
            results = []
            best_model = None
            best_accuracy = 0
            
            st.subheader("Model Performance")
            
            # Train and evaluate selected models
            for name in selected_models:
                model = models[name]
                # Training
                with st.spinner(f"Training {name}..."):
                    model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # For multiclass, use macro-average F1 score
                f1 = f1_score(y_test, y_pred, average='macro')
                
                # Store results
                results.append({
                    "Model": name, 
                    "Accuracy": accuracy, 
                    "F1 Score": f1
                })
                
                st.write(f"Model: {name} | Accuracy: {accuracy:.3f} | F1 Score: {f1:.3f}")
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                labels = ['Draw', 'Home Win', 'Away Win']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=labels, yticklabels=labels)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"{name} Confusion Matrix")
                st.pyplot(fig)
                
                # Save model
                st.session_state['models'][name] = model
                
                # Track best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
            
            # Display results table
            result_df = pd.DataFrame(results)
            st.write("Model Comparison Table:")
            st.dataframe(result_df)
            
            # Save best model for later use
            if best_model is not None:
                with open('best_model.pkl', 'wb') as f:
                    pickle.dump(best_model, f)
                st.success(f"Best model saved with accuracy: {best_accuracy:.3f}")
            
            # Cross-validation
            if st.checkbox("Perform Cross-Validation"):
                st.subheader("Cross-Validation Results")
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                for name in selected_models:
                    model = models[name]
                    cv_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
                    st.write(f"{name} CV Accuracy: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
            
            # Save data and models for other sections
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['best_model'] = best_model
            st.session_state['selected_features'] = selected_features
            st.session_state['result_df'] = result_df
            
        except Exception as e:
            st.error(f"An error occurred during modeling: {e}")
            st.stop()

##############################################
# Section 3: Advanced Visualizations
##############################################
if page == "Advanced Visualizations":
    st.title("Advanced Visualizations")
    if 'X_test' not in st.session_state:
        st.error("Please complete the modeling in the 'Modeling & Performance' section first!")
    else:
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        best_model = st.session_state['best_model']
        X_train = st.session_state['X_train']
        
        figs_to_report = []  # List for saving figures for the report
        
        # 1. Confusion Matrix using seaborn
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(y_test, best_model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["No Default", "Defaulted"],
                    yticklabels=["No Default", "Defaulted"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        figs_to_report.append(fig)
        
        # 2. Interactive ROC Curve with Plotly - adaptado para classificação multiclasse
        st.subheader("Interactive ROC Curve")
        
        # Obter probabilidades de todas as classes
        y_proba_all = best_model.predict_proba(X_test)
        
        # Criar uma curva ROC para cada classe usando a abordagem One-vs-Rest
        # Verifica se temos uma classificação multiclasse
        n_classes = len(np.unique(y_test))
        
        if n_classes == 2:  # Classificação binária - abordagem padrão
            y_proba_best = y_proba_all[:, 1]  # Probabilidade da classe positiva
            fpr, tpr, _ = roc_curve(y_test, y_proba_best)
            roc_auc = roc_auc_score(y_test, y_proba_best)
            
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                     name=f'ROC Curve (AUC = {roc_auc:.3f})'))
            roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', 
                                     name='Baseline', line=dict(dash='dash')))
            
        else:  # Classificação multiclasse - usamos One-vs-Rest
            # Binarize a saída
            y_bin = label_binarize(y_test, classes=np.unique(y_test))
            
            # Calcular ROC AUC macro
            try:
                # Tenta calcular o AUC diretamente
                roc_auc = roc_auc_score(y_bin, y_proba_all, multi_class='ovr', average='macro')
                auc_label = f"Macro-Average AUC = {roc_auc:.3f}"
            except:
                auc_label = "Couldn't calculate Macro-Average AUC"
            
            # Plotar uma curva ROC para cada classe
            roc_fig = go.Figure()
            
            colors = px.colors.qualitative.Plotly
            class_names = {i: f"Class {i}" for i in range(n_classes)}
            
            # Mapear valores numéricos para nomes significativos se possível
            if hasattr(best_model, 'classes_'):
                if 'match_outcome' in locals() or 'match_outcome' in globals():
                    # Para dados de apostas esportivas
                    match_outcome_map = {1: 'Home Win', 0: 'Draw', 2: 'Away Win'}
                    if all(c in match_outcome_map for c in best_model.classes_):
                        class_names = {i: match_outcome_map[c] for i, c in enumerate(best_model.classes_)}
            
            for i in range(n_classes):
                if y_bin.shape[1] > i:  # Verificar se temos essa coluna em y_bin
                    # Computar a curva ROC para a classe i
                    if y_proba_all.shape[1] > i:  # Verificar se temos probabilidades para essa classe
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba_all[:, i])
                        
                        # Calcular AUC para essa classe
                        try:
                            roc_auc_i = roc_auc_score(y_bin[:, i], y_proba_all[:, i])
                            auc_text = f"AUC = {roc_auc_i:.3f}"
                        except:
                            auc_text = "AUC N/A"
                        
                        # Adicionar a curva para essa classe
                        roc_fig.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode='lines',
                            name=f'{class_names[i]} ({auc_text})',
                            line=dict(color=colors[i % len(colors)])
                        ))
            
            # Adicionar linha de base
            roc_fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                name='Baseline', line=dict(color='gray', dash='dash')
            ))
            
            # Adicionar anotação para o AUC médio
            roc_fig.add_annotation(
                x=0.5, y=0.02, text=auc_label,
                showarrow=False, font=dict(size=14)
            )
        
        roc_fig.update_layout(
            title="ROC Curve (One-vs-Rest para classificação multiclasse)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True
        )
        
        st.plotly_chart(roc_fig)
        
        # 3. Interactive Calibration Curve - adaptado para classificação multiclasse
        st.subheader("Calibration Curve")
        
        # Para classificação multiclasse, vamos usar a classe com maior probabilidade
        if n_classes == 2:
            y_proba_best = y_proba_all[:, 1]  # Classe positiva para binário
        else:
            # Para multiclasse, usamos a máxima probabilidade como confiança do modelo
            y_proba_best = np.max(y_proba_all, axis=1)
        
        prob_bins = np.linspace(0, 1, 11)
        bin_ids = np.digitize(y_proba_best, prob_bins) - 1
        
        # Para calibração multiclasse, calculamos a acurácia por bin
        bin_acc = []
        for i in range(len(prob_bins)):
            bin_mask = (bin_ids == i)
            if np.sum(bin_mask) > 0:
                # Verifica se a classe prevista está correta
                y_pred_bin = np.argmax(y_proba_all[bin_mask], axis=1)
                y_true_bin = y_test[bin_mask]
                bin_acc.append((y_pred_bin == y_true_bin).mean())
            else:
                bin_acc.append(np.nan)
        
        calib_fig = px.line(x=prob_bins, y=bin_acc, markers=True, 
                          labels={'x': 'Max Predicted Probability', 'y': 'Observed Accuracy'})
        calib_fig.add_scatter(x=[0,1], y=[0,1], mode='lines', name='Ideal', line=dict(dash='dash'))
        calib_fig.update_layout(title="Calibration Curve (Multiclass)")
        st.plotly_chart(calib_fig)
        
        # 4. Cumulative Accuracy Profile (CAP) - adaptado para multiclasse
        st.subheader("Cumulative Accuracy Profile (CAP)")
        
        # Para CAP multiclasse, usamos a probabilidade da classe correta
        if n_classes == 2:
            # Binário - usamos a probabilidade da classe positiva para amostras positivas
            pos_mask = (y_test == 1)
            if np.any(pos_mask):
                y_proba_correct = y_proba_all[pos_mask, 1]
                y_true_cap = y_test[pos_mask]
            else:
                y_proba_correct = y_proba_all[:, 1]
                y_true_cap = y_test
        else:
            # Multiclasse - extraímos a probabilidade da classe verdadeira para cada amostra
            # Convertemos y_test para array numpy para indexação direta
            y_test_array = np.array(y_test)
            y_proba_correct = np.array([
                y_proba_all[i, int(y_test_array[i])] if i < len(y_proba_all) and int(y_test_array[i]) < y_proba_all.shape[1] 
                else 0 for i in range(len(y_test_array))
            ])
            y_true_cap = np.ones_like(y_test_array)  # Todas as amostras são "positivas" para sua classe
        
        # Ordenar por probabilidade
        df_cap = pd.DataFrame({"y_true": y_true_cap, "y_proba": y_proba_correct})
        df_cap.sort_values(by="y_proba", ascending=False, inplace=True)
        df_cap["cum_true"] = df_cap["y_true"].cumsum()
        df_cap["cum_total"] = np.arange(1, len(df_cap)+1)
        
        # Calcular ganho
        total_pos = df_cap["y_true"].sum()
        if total_pos > 0:  # Evitar divisão por zero
            df_cap["gain"] = df_cap["cum_true"] / total_pos
            
            cap_fig = go.Figure()
            cap_fig.add_trace(go.Scatter(x=np.linspace(0, 1, len(df_cap)), y=df_cap["gain"], 
                                     mode='lines', name="CAP"))
            cap_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', 
                                     name="Baseline", line=dict(dash='dash')))
            cap_fig.update_layout(title="Cumulative Accuracy Profile (Multiclass)", 
                              xaxis_title="Percentile", yaxis_title="Gain")
            st.plotly_chart(cap_fig)
        else:
            st.warning("Cannot calculate CAP curve - no positive samples")
        
        # 5. Distribution of Predicted Scores (Histogram)
        st.subheader("Distribution of Predicted Scores")
        
        # Para multiclasse, plotamos um histograma por classe
        if n_classes <= 5:  # Limite para não ficar muito poluído
            dist_fig = go.Figure()
            
            for i in range(n_classes):
                if y_proba_all.shape[1] > i:
                    class_name = class_names.get(i, f"Class {i}")
                    dist_fig.add_trace(go.Histogram(
                        x=y_proba_all[:, i],
                        name=class_name,
                        opacity=0.7,
                        nbinsx=30
                    ))
            
            dist_fig.update_layout(
                title="Distribution of Predicted Probabilities by Class",
                xaxis_title="Predicted Probability",
                yaxis_title="Count",
                barmode='overlay'
            )
            st.plotly_chart(dist_fig)
        else:
            # Se houver muitas classes, mostramos apenas a distribuição da máxima probabilidade
            dist_fig = px.histogram(
                x=np.max(y_proba_all, axis=1), 
                nbins=50, 
                labels={'x': "Maximum Class Probability"}, 
                title="Histogram of Maximum Predicted Probability"
            )
            st.plotly_chart(dist_fig)
        
        # 6. Performance Dashboard Table
        st.subheader("Performance Dashboard")
        st.write("Model Metrics Comparison:")
        st.dataframe(st.session_state.get('result_df', pd.DataFrame()))
        
        # 7. SHAP Beeswarm Summary Plot (with sampling and bar plot option)
        st.subheader("SHAP Summary Plot (Beeswarm & Bar)")
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_test)
        
        # Sampling to avoid overcrowding
        max_points = 3000  # adjust as needed
        if len(X_test) > max_points:
            X_test_sample = X_test.sample(n=max_points, random_state=42)
            shap_values_sample = shap_values[[i for i in X_test_sample.index]]
        else:
            X_test_sample = X_test
            shap_values_sample = shap_values

        st.markdown("**Beeswarm Plot (sampled if needed)**")
        shap.summary_plot(shap_values_sample, X_test_sample, max_display=15, show=False)
        fig_beeswarm = plt.gcf()  # Get current figure
        st.pyplot(fig_beeswarm, bbox_inches="tight")
        plt.clf()

        st.markdown("---")
        st.markdown("**Bar Plot (average impact)**")
        shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15, show=False)
        fig_bar = plt.gcf()  # Get current figure
        st.pyplot(fig_bar, bbox_inches="tight")
        plt.clf()

##############################################
# Section 4: Advanced Feature Engineering & Selection
##############################################
if page == "Advanced Feature Engineering":
    st.title("Advanced Feature Engineering & Selection")
    if 'data' not in st.session_state:
        st.error("Please upload the data in the 'Data Upload & EDA' section first!")
    else:
        data = st.session_state['data']
        st.subheader("Select Features for Modeling")
        all_features = list(data.columns)
        # Remove required columns
        required_columns = ['match_id', 'match_outcome']
        selectable = [col for col in all_features if col not in required_columns]
        selected_features = st.multiselect("Select variables", selectable, default=selectable, label_visibility="visible")
        
        if len(selected_features) == 0:
            st.error("Please select at least one variable to proceed.")
        else:
            st.write("Selected Features:", selected_features)
            st.subheader("Correlation Heatmap")
            
            # Filtrando apenas características numéricas para a matriz de correlação
            numeric_features = []
            for feature in selected_features:
                if pd.api.types.is_numeric_dtype(data[feature]):
                    numeric_features.append(feature)
            
            if len(numeric_features) < 2:
                st.warning("Pelo menos duas características numéricas são necessárias para gerar a matriz de correlação. Por favor, selecione mais características numéricas.")
            else:
                fig, ax = plt.subplots(figsize=(10,8))
                sns.heatmap(data[numeric_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            
            st.subheader("PCA - Principal Component Analysis")
            
            # Também usamos apenas features numéricas para PCA
            if len(numeric_features) < 2:
                st.warning("Pelo menos duas características numéricas são necessárias para executar a PCA. Por favor, selecione mais características numéricas.")
            else:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(data[numeric_features].dropna())
                pca_df = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
                pca_fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA of Selected Data")
                st.plotly_chart(pca_fig)

##############################################
# Section 5: Dynamic Hyperparameter Tuning
##############################################
if page == "Dynamic Hyperparameter Tuning":
    st.title("Dynamic Hyperparameter Tuning")
    if 'data' not in st.session_state:
        st.error("Please upload the data in the 'Data Upload & EDA' section first!")
    else:
        data = st.session_state['data']
        try:
            # Adaptado para dados de apostas esportivas
            # Usamos match_id como ID único e match_outcome como target
            features_cols = [col for col in data.columns if col not in ['match_id', 'match_outcome']]
            
            # Verificar se existem colunas numéricas suficientes
            numeric_features = []
            for col in features_cols:
                if pd.api.types.is_numeric_dtype(data[col]):
                    numeric_features.append(col)
            
            if len(numeric_features) < 3:
                st.error("Seu arquivo CSV não possui features numéricas suficientes para treinamento do modelo.")
                st.stop()
                
            # Remover colunas não numéricas e colunas que não são úteis para predição
            exclude_cols = ['match_id', 'match_outcome', 'home_team', 'away_team', 'date', 'league', 'country']
            X_cols = [col for col in numeric_features if col not in exclude_cols]
            
            # Selecionar features e target
            X = data[X_cols]
            y = data['match_outcome']
            
            # Mostrar as features selecionadas
            st.write("Features selecionadas:", X.columns.tolist())
            st.write("Distribuição da variável alvo:")
            
            # Mostrar a distribuição da variável alvo
            target_counts = y.value_counts()
            fig_target, ax_target = plt.subplots(figsize=(8, 4))
            bars = ax_target.bar(['Empate (0)', 'Vitória Casa (1)', 'Vitória Fora (2)'], 
                       [target_counts.get(0, 0), target_counts.get(1, 0), target_counts.get(2, 0)],
                       color=['#5D8BF4', '#18A558', '#FF5252'])
            ax_target.set_title('Distribuição dos Resultados', color='white')
            ax_target.set_ylabel('Contagem', color='white')
            ax_target.tick_params(colors='white')
            
            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                ax_target.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', color='white')
            
            st.pyplot(fig_target)
            
        except KeyError as e:
            st.error(f"Erro ao processar as colunas do CSV: {e}")
            st.info("Seu arquivo CSV deve conter pelo menos as colunas 'match_id' e 'match_outcome', além de features numéricas para previsão.")
            st.stop()
        
        # Balanceamento de classe com SMOTE
        try:
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            X_res = pd.DataFrame(X_res, columns=X.columns)
            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
            
            # Informar ao usuário sobre o balanceamento
            st.success(f"Dados balanceados com SMOTE. Forma original: {X.shape}, Após SMOTE: {X_res.shape}")
        
            st.subheader("RandomForest Hyperparameter Tuning")
            n_estimators = st.slider("Number of Trees", 50, 500, 200, step=50, label_visibility="visible")
            max_depth = st.slider("Max Depth", 1, 20, 5, label_visibility="visible")
            
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Avaliação do modelo
            st.subheader("Avaliação do Modelo")
            
            # Calcular as probabilidades para cada classe
            y_proba_all = model.predict_proba(X_test)
            
            # Para versões multiclasse, precisamos calcular o AUC de maneira diferente
            if y_proba_all.shape[1] > 2:  # Multiclasse
                try:
                    auc = roc_auc_score(y_test, y_proba_all, multi_class='ovr', average='macro')
                except:
                    auc = accuracy_score(y_test, y_pred)  # Fallback para acurácia se AUC falhar
            else:  # Binário
                try:
                    auc = roc_auc_score(y_test, y_proba_all[:,1])
                except:
                    auc = accuracy_score(y_test, y_pred)
                    
            st.write(f"Model AUC: {auc:.3f}")
            
            # Matriz de confusão
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            labels = ['Empate', 'Vitória Casa', 'Vitória Fora']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                       xticklabels=labels, yticklabels=labels)
            ax_cm.set_xlabel("Predito")
            ax_cm.set_ylabel("Real")
            ax_cm.set_title("Matriz de Confusão")
            plt.tight_layout()
            st.pyplot(fig_cm)
            
            # Salvar o modelo e dados de teste/treino na sessão
            if 'models' not in st.session_state:
                st.session_state['models'] = {}
                
            model_name = f"RF_trees{n_estimators}_depth{max_depth}"
            st.session_state['models'][model_name] = model
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['best_model'] = model
            
            st.success(f"Modelo treinado com sucesso! Agora você pode explorar os resultados nas seções de Explainability.")
            
        except Exception as e:
            st.error(f"Erro durante o processamento: {e}")
            st.info("Verifique se seu arquivo CSV contém dados numéricos válidos e não tem valores faltantes.")

##############################################
# Section 6: Advanced Explainability
##############################################
if page == "Advanced Explainability":
    st.title("Advanced Model Explainability")
    if 'X_test' not in st.session_state:
        st.error("Please complete the modeling in the 'Modeling & Performance' section first!")
    else:
        X_test = st.session_state['X_test']
        best_model = st.session_state['best_model']
        X_train = st.session_state['X_train']
        
        st.subheader("SHAP Dependence Plot")
        
        # Use the appropriate explainer based on model type
        if hasattr(best_model, 'feature_importances_'):
            # For tree-based models, use TreeExplainer
            explainer = shap.TreeExplainer(best_model)
        else:
            # For other model types, use regular Explainer
            explainer = shap.Explainer(best_model, X_train)
            
        shap_values = explainer(X_test)
        feature = st.selectbox("Select a feature for the Dependence Plot", X_test.columns, label_visibility="visible")
        
        # Fix for multiclass - check if we're dealing with multiclass SHAP values
        try:
            plt.figure(figsize=(10, 6))
            
            # Detect multiclass by examining shape or structure of shap_values
            is_multiclass = False
            num_classes = 2  # Default for binary classification
            
            # Check if TreeExplainer output (list of arrays)
            if isinstance(shap_values, list):
                is_multiclass = len(shap_values) > 1
                num_classes = len(shap_values)
            # Check if Explainer output (has values attribute)
            elif hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) > 2:
                    is_multiclass = True
                    num_classes = shap_values.values.shape[1]
            
            if is_multiclass:  # Multiclass case
                st.info("Multiclass model detected. Select a class to view SHAP values for that class.")
                # Add class selection for multiclass models
                selected_class = st.selectbox(
                    "Select class for SHAP analysis:", 
                    range(num_classes), 
                    format_func=lambda i: f"Class {i} " + (["Draw", "Home Win", "Away Win"][i] if i < 3 else ""),
                    label_visibility="visible"
                )
                
                # Get feature index
                feature_idx = list(X_test.columns).index(feature)
                
                # Extract the appropriate SHAP values based on the explainer type
                if isinstance(shap_values, list):
                    # For TreeExplainer with list output
                    feature_values = X_test[feature].values
                    shap_feature_values = shap_values[selected_class][:, feature_idx]
                else:
                    # For Explainer with Explanation object
                    feature_values = X_test[feature].values
                    shap_feature_values = shap_values.values[:, selected_class, feature_idx]
                
                # Create a scatter plot
                plt.scatter(feature_values, shap_feature_values, alpha=0.6)
                
                # Add a trend line
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(feature_values, shap_feature_values)
                x_line = np.linspace(min(feature_values), max(feature_values), 100)
                y_line = slope * x_line + intercept
                plt.plot(x_line, y_line, color='red', linestyle='--')
                
                plt.xlabel(feature)
                plt.ylabel(f"SHAP value (impact on class {selected_class})")
                plt.title(f"SHAP Dependence Plot for {feature} (Class {selected_class})")
                plt.grid(True, linestyle='--', alpha=0.6)
            else:  # Binary case
                try:
                    # For regular shap.dependence_plot
                    if hasattr(shap_values, 'values'):
                        shap.dependence_plot(feature, shap_values.values, X_test, show=False)
                    else:
                        # For older SHAP versions or direct arrays
                        shap.dependence_plot(feature, shap_values, X_test, show=False)
                except Exception as e:
                    st.error(f"Error with built-in dependence plot: {e}")
                    # Fallback to custom scatter plot
                    feature_idx = list(X_test.columns).index(feature)
                    feature_values = X_test[feature].values
                    
                    if isinstance(shap_values, list):
                        # For TreeExplainer with list output
                        shap_feature_values = shap_values[0][:, feature_idx]
                    else:
                        # For Explainer with Explanation object
                        shap_feature_values = shap_values.values[:, feature_idx]
                    
                    plt.scatter(feature_values, shap_feature_values, alpha=0.6)
                    plt.xlabel(feature)
                    plt.ylabel(f"SHAP value (impact on prediction)")
                    plt.title(f"SHAP Dependence Plot for {feature}")
                    plt.grid(True, linestyle='--', alpha=0.6)
        except Exception as e:
            st.error(f"Error generating dependence plot: {e}")
            st.info("Try using SHAP summary plots instead which handle multiclass data better.")
            # Fallback to summary plot
            plt.figure(figsize=(10, 6))
            try:
                shap.summary_plot(shap_values, X_test, max_display=10, show=False)
            except Exception as summary_error:
                st.error(f"Error generating summary plot: {summary_error}")
                plt.text(0.5, 0.5, "Unable to generate SHAP plots", 
                        horizontalalignment='center', verticalalignment='center')
            
        fig_dep = plt.gcf()
        st.pyplot(fig_dep, bbox_inches="tight")
        plt.clf()
        
        st.subheader("SHAP Force Plot for a Single Instance")
        instance_index = st.slider("Select instance for Force Plot", 0, X_test.shape[0]-1, 0, label_visibility="visible")
        
        # Similar fix for force plot
        try:
            # Check if we're dealing with multiclass SHAP values for force plot
            if len(shap_values.shape) > 2:  # Multiclass case
                if 'selected_class' not in locals():
                    selected_class = st.selectbox("Select class for force plot:", 
                                                range(shap_values.shape[1]), 
                                                format_func=lambda i: f"Class {i} " + (["Draw", "Home Win", "Away Win"][i] if i < 3 else ""),
                                                label_visibility="visible")
                
                # Create a custom force plot for multiclass
                plt.figure(figsize=(12, 4))
                # Get the features and their SHAP values for the selected instance and class
                instance_features = X_test.iloc[instance_index]
                instance_shap_values = shap_values.values[instance_index, selected_class, :]
                
                # Sort features by absolute SHAP value
                indices = np.argsort(np.abs(instance_shap_values))[-10:]  # Top 10 features
                sorted_features = [X_test.columns[i] for i in indices]
                sorted_values = instance_shap_values[indices]
                sorted_feature_values = [f"{instance_features[feature]:.3f}" for feature in sorted_features]
                
                # Create a horizontal bar chart
                plt.barh(range(len(sorted_features)), sorted_values, color=['r' if x < 0 else 'g' for x in sorted_values])
                plt.yticks(range(len(sorted_features)), [f"{feature} = {value}" for feature, value in zip(sorted_features, sorted_feature_values)])
                plt.xlabel(f"SHAP value (impact on class {selected_class})")
                plt.title(f"Top features influencing prediction for instance {instance_index} (Class {selected_class})")
                plt.tight_layout()
            else:  # Binary case
                force_fig = shap.force_plot(explainer.expected_value, 
                                         shap_values.values[instance_index], 
                                         X_test.iloc[instance_index], 
                                         matplotlib=True, show=False)
                force_fig = plt.gcf()
        except Exception as e:
            st.error(f"Error generating force plot: {e}")
            st.info("This may be due to incompatible SHAP values or model type.")
            force_fig = plt.figure()
            plt.text(0.5, 0.5, "Unable to generate force plot", 
                    horizontalalignment='center', verticalalignment='center')
            
        if 'force_fig' in locals():
            st.pyplot(force_fig, bbox_inches="tight")
        else:
            st.pyplot(plt.gcf(), bbox_inches="tight")
        plt.clf()
        
        
        st.subheader("Local Explanation with LIME and History Logging")
        if 'lime' in globals():
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values, 
                feature_names=X_train.columns, 
                class_names=["No Default", "Defaulted"], 
                discretize_continuous=True
            )
            i = st.slider("Select instance for LIME explanation", 0, X_test.shape[0]-1, 0, label_visibility="visible")
            exp = lime_explainer.explain_instance(
                X_test.iloc[i].values, 
                best_model.predict_proba, 
                num_features=10
            )
            st.write("LIME Explanation:", exp.as_list())
            record_explanation(i, exp.as_list())
            st.subheader("Explanation History")
            st.write(st.session_state.get('explanation_history', {}))
        else:
            st.info("LIME is not installed. To use LIME, install with: pip install lime")

##############################################
# Section 7: Interactive Explainability & Model Comparison
##############################################
if page == "Interactive Explainability & Model Comparison":
    st.title("Interactive Explainability & Model Comparison")
    if 'models' not in st.session_state or len(st.session_state['models']) < 2:
        st.error("At least two models must be trained in the 'Modeling & Performance' section for comparison!")
    else:
        models = st.session_state['models']
        selected_models = st.multiselect("Select models for comparison", list(models.keys()), label_visibility="visible")
        if selected_models:
            cols = st.columns(len(selected_models))
            for idx, model_name in enumerate(selected_models):
                model = models[model_name]
                with cols[idx]:
                    st.subheader(f"SHAP Summary - {model_name}")
                    
                    try:
                        # Use appropriate explainer based on model type
                        if model.__class__.__name__ in ["RandomForestClassifier", "GradientBoostingClassifier", "AdaBoostClassifier", "XGBClassifier"]:
                            explainer = shap.TreeExplainer(model)
                        elif model.__class__.__name__ == "SVC":
                            # For SVC, use KernelExplainer with a sample of the training data as background
                            background = st.session_state['X_train'].iloc[:100]
                            explainer = shap.KernelExplainer(model.predict_proba, background)
                        else:
                            explainer = shap.Explainer(model, st.session_state['X_train'])
                        
                        # Get SHAP values
                        shap_values = explainer(st.session_state['X_test'])
                        
                        # Check if we have multiclass SHAP values
                        is_multiclass = False
                        if isinstance(shap_values, list):
                            is_multiclass = len(shap_values) > 1
                        elif hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
                            is_multiclass = True
                        
                        # For multiclass, let user select which class to visualize
                        if is_multiclass:
                            if isinstance(shap_values, list):
                                num_classes = len(shap_values)
                            else:
                                num_classes = shap_values.values.shape[1]
                            
                            # We'll show the aggregated values across all classes to avoid cluttering the UI
                            st.info(f"Showing aggregated feature importance across all classes for {model_name}")
                            
                            # Create a custom summary plot
                            plt.figure(figsize=(8, 6))
                            
                            if isinstance(shap_values, list):
                                # For TreeExplainer with list output
                                # Compute mean absolute SHAP values across all classes
                                mean_abs_shap = np.zeros(st.session_state['X_test'].shape[1])
                                for class_idx in range(len(shap_values)):
                                    mean_abs_shap += np.mean(np.abs(shap_values[class_idx]), axis=0)
                                mean_abs_shap /= len(shap_values)
                                
                                # Sort features by importance
                                sorted_idx = np.argsort(mean_abs_shap)
                                sorted_features = [st.session_state['X_test'].columns[i] for i in sorted_idx[-10:]]  # Top 10
                                sorted_values = mean_abs_shap[sorted_idx[-10:]]
                                
                                # Plot a horizontal bar chart
                                plt.barh(range(len(sorted_features)), sorted_values)
                                plt.yticks(range(len(sorted_features)), sorted_features)
                                plt.xlabel('Mean |SHAP value|')
                                plt.title('Feature Importance (from SHAP values)')
                            else:
                                # For Explainer with Explanation object
                                try:
                                    # Try standard summary plot (may work for some multiclass models)
                                    shap.summary_plot(shap_values, st.session_state['X_test'], max_display=10, show=False)
                                except Exception as e:
                                    st.warning(f"Standard summary plot failed: {e}. Showing custom plot instead.")
                                    
                                    # Fallback to custom plot
                                    # Compute mean absolute SHAP values across all classes
                                    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=(0, 1))
                                    
                                    # Sort features by importance
                                    sorted_idx = np.argsort(mean_abs_shap)
                                    sorted_features = [st.session_state['X_test'].columns[i] for i in sorted_idx[-10:]]  # Top 10
                                    sorted_values = mean_abs_shap[sorted_idx[-10:]]
                                    
                                    # Plot a horizontal bar chart
                                    plt.barh(range(len(sorted_features)), sorted_values)
                                    plt.yticks(range(len(sorted_features)), sorted_features)
                                    plt.xlabel('Mean |SHAP value|')
                                    plt.title('Feature Importance (from SHAP values)')
                        else:
                            # Binary case - use standard summary plot
                            try:
                                if isinstance(shap_values, list):
                                    shap.summary_plot(shap_values[0], st.session_state['X_test'], max_display=10, plot_type="bar", show=False)
                                else:
                                    shap.summary_plot(shap_values, st.session_state['X_test'], max_display=10, plot_type="bar", show=False)
                            except Exception as e:
                                st.warning(f"Standard summary plot failed: {e}. Showing custom plot instead.")
                                
                                # Fallback to custom plot
                                if isinstance(shap_values, list):
                                    mean_abs_shap = np.mean(np.abs(shap_values[0]), axis=0)
                                else:
                                    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
                                
                                # Sort features by importance
                                sorted_idx = np.argsort(mean_abs_shap)
                                sorted_features = [st.session_state['X_test'].columns[i] for i in sorted_idx[-10:]]  # Top 10
                                sorted_values = mean_abs_shap[sorted_idx[-10:]]
                                
                                # Plot a horizontal bar chart
                                plt.barh(range(len(sorted_features)), sorted_values)
                                plt.yticks(range(len(sorted_features)), sorted_features)
                                plt.xlabel('Mean |SHAP value|')
                                plt.title('Feature Importance (from SHAP values)')
                    except Exception as e:
                        st.error(f"Error generating SHAP values for {model_name}: {e}")
                        plt.figure(figsize=(8, 4))
                        plt.text(0.5, 0.5, f"Could not generate SHAP values:\n{str(e)}", 
                                horizontalalignment='center', verticalalignment='center')
                    
                    # Display the plot
                    st.pyplot(plt.gcf(), bbox_inches="tight")
                    plt.clf()

##############################################
# Section 8: Export Report
##############################################
if page == "Export Report":
    st.title("Export Report")
    if 'result_df' not in st.session_state or 'best_model' not in st.session_state:
        st.error("Please complete the modeling and analyses before generating the report!")
    else:
        metrics = {}
        try:
            # Obtenha as probabilidades previstas
            y_proba = st.session_state['best_model'].predict_proba(st.session_state['X_test'])
            
            # Verifique se é classificação binária ou multiclasse
            if y_proba.shape[1] == 2:  # Classificação binária
                metrics["AUC"] = roc_auc_score(st.session_state['y_test'], y_proba[:, 1])
                # KS statistic para binário
                metrics["KS"] = np.max(np.abs(np.linspace(0, 1, len(st.session_state['X_test'])) - 
                                        np.sort(y_proba[:, 1])))
            else:  # Classificação multiclasse
                # Use 'ovr' (one-vs-rest) para multiclasse
                metrics["AUC"] = roc_auc_score(
                    st.session_state['y_test'], 
                    y_proba, 
                    multi_class='ovr', 
                    average='macro'
                )
                # Para multiclasse, usamos a probabilidade máxima como métrica para KS
                metrics["KS"] = np.max(np.abs(np.linspace(0, 1, len(st.session_state['X_test'])) - 
                                        np.sort(np.max(y_proba, axis=1))))
        except Exception as e:
            st.warning(f"Não foi possível calcular algumas métricas: {e}")
            metrics = {"Nota": "Métricas não disponíveis devido a um erro"}
            
        report_text = "<p>This report was generated from the interactive Sports Betting Dashboard.</p>"
        figs_list = []  # Optionally, add figures from previous sections here
        st.markdown("Click the button below to download the HTML report.")
        report_buffer = generate_report(st.session_state.get('result_df', pd.DataFrame()), metrics, report_text, figs_list)
        st.download_button("Download Report", data=report_buffer, file_name="sports_betting_report.html", mime="text/html")

##############################################
# Section 9: Advanced What-if Simulation
##############################################
if page == "Advanced What-if Simulation":
    st.title("Advanced What-if Simulation")
    if 'X_test' not in st.session_state:
        st.error("Please complete the modeling in the 'Modeling & Performance' section first!")
    else:
        X_test = st.session_state['X_test']
        best_model = st.session_state['best_model']
        
        st.markdown("### Multiple Scenario Simulation")
        num_scenarios = st.number_input("Number of scenarios to simulate", min_value=1, max_value=5, value=1)
        scenarios = []
        
        # Identify the top 2 important numeric features for simulation
        if len(X_test.columns) >= 2:
            # Get first two numeric columns instead of hardcoding column names
            numeric_cols = [col for col in X_test.columns if pd.api.types.is_numeric_dtype(X_test[col])]
            if len(numeric_cols) >= 2:
                feature1 = numeric_cols[0]
                feature2 = numeric_cols[1]
                
                for i in range(num_scenarios):
                    st.markdown(f"#### Scenario {i+1}")
                    feat1_input = st.number_input(f"Scenario {i+1} - {feature1}", 
                                                value=float(X_test[feature1].mean()), key=f"feat1_{i}")
                    feat2_input = st.number_input(f"Scenario {i+1} - {feature2}", 
                                                 value=float(X_test[feature2].mean()), key=f"feat2_{i}")
                    
                    # Create a copy of a reference instance and update selected features for simulation
                    scenario_instance = X_test.iloc[0].copy()
                    scenario_instance[feature1] = feat1_input
                    scenario_instance[feature2] = feat2_input
                    scenarios.append(scenario_instance)
                
                st.markdown("### Predicted Probabilities for Each Scenario")
                for idx, scenario in enumerate(scenarios):
                    # For multiclass, show probability for each class
                    probas = best_model.predict_proba(scenario.values.reshape(1, -1))[0]
                    
                    st.write(f"Scenario {idx+1} Predictions:")
                    num_classes = len(probas)
                    if num_classes == 2:  # Binary (e.g., Win/Loss)
                        st.write(f"- Probability of outcome 0 (Draw): {probas[0]:.3f}")
                        st.write(f"- Probability of outcome 1 (Home Win): {probas[1]:.3f}")
                    else:  # Multiclass
                        for class_idx, prob in enumerate(probas):
                            outcome_name = ""
                            if class_idx == 0:
                                outcome_name = "Draw"
                            elif class_idx == 1:
                                outcome_name = "Home Win"
                            elif class_idx == 2:
                                outcome_name = "Away Win"
                            else:
                                outcome_name = f"Class {class_idx}"
                            st.write(f"- Probability of {outcome_name}: {prob:.3f}")
                
                st.markdown("### Sensitivity Analysis - Visual Feedback")
                feature = st.selectbox("Select feature for sensitivity analysis", numeric_cols, label_visibility="visible")
                instance_ref = X_test.iloc[0].copy()
                base_val = instance_ref[feature]
                # Avoid negative values for sensitivity analysis
                min_val = max(base_val * 0.5, 0.1) if base_val > 0 else 0.1
                values = np.linspace(min_val, base_val * 1.5, 20)
                
                # For multiclass, track probabilities for all classes
                probas_by_class = [[] for _ in range(best_model.predict_proba(X_test.iloc[0:1])[0].shape[0])]
                
                for val in values:
                    instance_ref[feature] = val
                    probs = best_model.predict_proba(instance_ref.values.reshape(1, -1))[0]
                    
                    # Save probabilities for each class
                    for i, prob in enumerate(probs):
                        probas_by_class[i].append(prob)
                
                # Create figure for sensitivity analysis
                sens_fig = go.Figure()
                
                # Add a trace for each class
                class_names = {0: "Draw", 1: "Home Win", 2: "Away Win"}
                for i, probs in enumerate(probas_by_class):
                    class_name = class_names.get(i, f"Class {i}")
                    sens_fig.add_trace(go.Scatter(
                        x=values, 
                        y=probs, 
                        mode='lines+markers', 
                        name=f'{class_name}'
                    ))
                
                sens_fig.update_layout(
                    title=f"Sensitivity of Prediction to {feature}",
                    xaxis_title=feature, 
                    yaxis_title="Predicted Probability"
                )
                st.plotly_chart(sens_fig)
            else:
                st.error("Not enough numeric features available for simulation. Please ensure your dataset contains at least 2 numeric columns.")
        else:
            st.error("Not enough features available for simulation. Please ensure your dataset has been properly loaded.")

##############################################
# Section 10: Advanced Interpretability Visualizations
##############################################
if page == "Advanced Interpretability Visualizations":
    st.title("Advanced Interpretability Visualizations")
    if 'X_test' not in st.session_state or 'best_model' not in st.session_state:
        st.error("Please complete the modeling in the 'Modeling & Performance' section first!")
    else:
        X_test = st.session_state['X_test']
        best_model = st.session_state['best_model']
        X_train = st.session_state['X_train']
        
        st.header("Partial Dependence Plot (PDP) and ICE")
        # Allow user to select one or more features for PDP/ICE
        features = st.multiselect("Select features for PDP/ICE", X_test.columns, default=[X_test.columns[0]], label_visibility="visible")
        if features:
            try:
                # Check if multiclass by determining the number of classes from model output
                is_multiclass = False
                try:
                    y_proba = best_model.predict_proba(X_test.iloc[:1])
                    n_classes = y_proba.shape[1]
                    is_multiclass = n_classes > 2
                except:
                    pass
                
                # For multiclass models, let user select which class to visualize
                target_class = None
                if is_multiclass:
                    target_class = st.selectbox("Select class to visualize:", 
                                             range(n_classes), 
                                             format_func=lambda i: f"Class {i} " + (["Draw", "Home Win", "Away Win"][i] if i < 3 else ""),
                                             label_visibility="visible")
                    st.info(f"Showing PDP/ICE plots for class {target_class}")
                    # Use the selected target class for multiclass
                    disp = PartialDependenceDisplay.from_estimator(
                        best_model, X_test, features, 
                        target=target_class,  # Specify which class to visualize
                        kind="both", 
                        subsample=50, 
                        grid_resolution=20
                    )
                else:
                    # For binary classification, no need to specify target
                    disp = PartialDependenceDisplay.from_estimator(
                        best_model, X_test, features, 
                        kind="both", 
                        subsample=50, 
                        grid_resolution=20
                    )
                
                st.pyplot(disp.figure_, bbox_inches="tight")
            except Exception as e:
                st.error(f"Error generating PDP/ICE: {e}")
                st.info("For multiclass models, you need to select which class to visualize.")

        st.header("Waterfall Chart for a Single Prediction")
        instance_idx = st.slider("Select instance for Waterfall Chart", 0, X_test.shape[0]-1, 0, label_visibility="visible")
        st.markdown("Waterfall Chart for the selected instance:")
        try:
            explainer = shap.Explainer(best_model, X_train)
            shap_values = explainer(X_test)
            
            # Check if we're dealing with multiclass SHAP values
            if len(shap_values.shape) > 2:  # Multiclass case
                # Let the user select which class to visualize
                n_classes = shap_values.shape[1]
                selected_class = st.selectbox(
                    "Select class for waterfall plot:", 
                    range(n_classes), 
                    format_func=lambda i: f"Class {i} " + (["Draw", "Home Win", "Away Win"][i] if i < 3 else ""),
                    label_visibility="visible"
                )
                
                # Plot the waterfall for the selected class
                st.info(f"Showing SHAP waterfall plot for class {selected_class}")
                
                # Create a custom waterfall plot for multiclass
                plt.figure(figsize=(10, 8))
                
                # Get the features and their SHAP values for the selected instance and class
                instance_features = X_test.iloc[instance_idx]
                instance_shap_values = shap_values.values[instance_idx, selected_class, :]
                
                # Get expected value for this class (base value)
                if isinstance(explainer.expected_value, list) or isinstance(explainer.expected_value, np.ndarray):
                    expected_value = explainer.expected_value[selected_class]
                else:
                    expected_value = explainer.expected_value
                
                # Sort features by SHAP value magnitude
                indices = np.argsort(np.abs(instance_shap_values))
                sorted_features = [X_test.columns[i] for i in indices[-10:]]  # Top 10 features
                sorted_values = instance_shap_values[indices[-10:]]
                sorted_feature_values = [f"{instance_features[feature]:.3f}" for feature in sorted_features]
                
                # Calculate cumulative effect
                cumulative = np.cumsum(sorted_values)
                
                # Create custom waterfall chart
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Base value
                ax.plot([0, len(sorted_features)+1], [expected_value, expected_value], 'k--', label='Expected value')
                
                # Plot the bars
                y = expected_value
                for i, (feature, value) in enumerate(zip(sorted_features, sorted_values)):
                    if value >= 0:
                        ax.bar(i, value, bottom=y, color='green', alpha=0.7)
                    else:
                        ax.bar(i, value, bottom=y, color='red', alpha=0.7)
                    y += value
                
                # Final prediction
                final_prediction = expected_value + np.sum(sorted_values)
                ax.bar(len(sorted_features), 0, bottom=final_prediction, color='blue')
                
                # Add feature names and values
                ax.set_xticks(range(len(sorted_features)+1))
                ax.set_xticklabels([f"{f}\n({v})" for f, v in zip(sorted_features, sorted_feature_values)] + ['Prediction'], rotation=45, ha='right')
                
                # Add title and labels
                ax.set_title(f'SHAP Waterfall Plot for Class {selected_class}')
                ax.set_ylabel('Prediction')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add annotation for the final prediction
                ax.annotate(f'Final: {final_prediction:.3f}', 
                           (len(sorted_features), final_prediction),
                           xytext=(5, 5), textcoords='offset points')
                
                plt.tight_layout()
            else:
                # Binary classification case - use the standard waterfall plot
                try:
                    shap.plots.waterfall(shap_values[instance_idx], show=False)
                except:
                    # Fallback for older SHAP versions
                    shap.waterfall_plot(shap_values[instance_idx], show=False)
            
            fig_waterfall = plt.gcf()  # Get the current figure
            st.pyplot(fig_waterfall, bbox_inches="tight")
            plt.clf()
        except Exception as e:
            st.error(f"Error generating Waterfall Chart: {e}")
            st.info("Try selecting a different instance or model type. For multiclass problems, ensure the correct SHAP values are being used.")

        st.header("Permutation Feature Importance")
        from sklearn.inspection import permutation_importance
        
        try:
            # Determine appropriate scoring based on number of classes
            y_proba = best_model.predict_proba(X_test.iloc[:1])
            n_classes = y_proba.shape[1]
            
            if n_classes > 2:  # Multiclass
                st.info("Using 'accuracy' as the scoring metric for multiclass permutation importance.")
                scoring = 'accuracy'  # Use accuracy for multiclass as it's more stable
            else:  # Binary
                st.info("Using 'roc_auc' as the scoring metric for binary permutation importance.")
                scoring = 'roc_auc'
                
            # Run permutation importance with appropriate scoring
            result = permutation_importance(
                best_model, 
                X_test, 
                st.session_state['y_test'], 
                n_repeats=10, 
                random_state=42, 
                scoring=scoring
            )
            
            # Sort features by importance
            sorted_idx = result.importances_mean.argsort()[-15:]  # Show top 15 features
            
            # Create and display boxplot
            fig_perm, ax_perm = plt.subplots(figsize=(10, 8))
            ax_perm.boxplot(result.importances[sorted_idx].T, 
                           vert=False, 
                           labels=X_test.columns[sorted_idx])
            ax_perm.set_title(f"Permutation Feature Importance (using {scoring})")
            ax_perm.set_xlabel("Decrease in Performance")
            ax_perm.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_perm, bbox_inches="tight")
            plt.clf()
            
            # Also show as a bar chart for simpler interpretation
            importance_df = pd.DataFrame({
                'Feature': X_test.columns[sorted_idx],
                'Importance': result.importances_mean[sorted_idx]
            }).sort_values(by='Importance', ascending=False)
            
            fig_bar = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Feature Importance (using {scoring})"
            )
            st.plotly_chart(fig_bar)
            
        except Exception as e:
            st.error(f"Error calculating permutation importance: {e}")
            st.info("Try using a different scoring metric for multiclass classification, such as 'accuracy', 'balanced_accuracy', or 'f1_macro'.")

        st.header("Interactive Dashboard with Dynamic Filters")
        st.markdown("Use the filters below to visualize a subset of the test data and model predictions.")
        
        # Get numeric features from X_test
        numeric_features = [col for col in X_test.columns if pd.api.types.is_numeric_dtype(X_test[col])]
        
        if len(numeric_features) >= 2:
            # Let user select which features to filter on
            filter_features = st.multiselect(
                "Select features to filter on:", 
                numeric_features,
                default=numeric_features[:2] if len(numeric_features) >= 2 else numeric_features,
                label_visibility="visible"
            )
            
            if len(filter_features) > 0:
                # Create filters for each selected feature
                filter_conditions = []
                
                for feature in filter_features:
                    feature_min = float(X_test[feature].min())
                    feature_max = float(X_test[feature].max())
                    
                    # Create slider for this feature
                    feature_range = st.slider(
                        f"{feature}", 
                        feature_min, 
                        feature_max, 
                        (feature_min, feature_max),
                        label_visibility="visible"
                    )
                    
                    # Add condition for this feature
                    filter_conditions.append(
                        (X_test[feature] >= feature_range[0]) & 
                        (X_test[feature] <= feature_range[1])
                    )
                
                # Combine all conditions with AND
                combined_condition = filter_conditions[0]
                for condition in filter_conditions[1:]:
                    combined_condition = combined_condition & condition
                
                # Apply filter
                filtered_data = X_test[combined_condition]
                
                if not filtered_data.empty:
                    st.write(f"Filtered Test Data ({len(filtered_data)} rows, showing first 5):", filtered_data.head())
                    
                    # Get predictions for filtered data
                    y_proba = best_model.predict_proba(filtered_data)
                    
                    # For multiclass, let user select which class probability to view
                    if y_proba.shape[1] > 2:  # Multiclass
                        selected_class = st.selectbox(
                            "Select class probability to view:", 
                            range(y_proba.shape[1]),
                            format_func=lambda i: f"Class {i} " + (["Draw", "Home Win", "Away Win"][i] if i < 3 else ""),
                            label_visibility="visible"
                        )
                        
                        proba_filtered = y_proba[:, selected_class]
                        class_name = ["Draw", "Home Win", "Away Win"][selected_class] if selected_class < 3 else f"Class {selected_class}"
                        title = f"Distribution of Predicted Probability for {class_name}"
                    else:  # Binary
                        proba_filtered = y_proba[:, 1]  # Probability of class 1
                        title = "Distribution of Predicted Probability"
                    
                    # Plot histogram of probabilities
                    fig_filter = px.histogram(
                        x=proba_filtered, 
                        nbins=30, 
                        labels={'x': "Predicted Probability"},
                        title=title
                    )
                    st.plotly_chart(fig_filter)
                else:
                    st.warning("No data matches the selected filters. Try adjusting the filter ranges.")
            else:
                st.warning("Please select at least one feature to filter on.")
        else:
            st.warning("Not enough numeric features available for filtering.")

        st.header("Multivariate Sensitivity Analysis (Heatmap)")
        st.markdown("Select two features to analyze their combined effect on the model prediction.")
        
        # Get numeric features
        numeric_features = [col for col in X_test.columns if pd.api.types.is_numeric_dtype(X_test[col])]
        
        if len(numeric_features) >= 2:
            # Let user select which features to analyze
            feat1 = st.selectbox("Select first feature", numeric_features, index=0, label_visibility="visible")
            feat2 = st.selectbox("Select second feature", numeric_features, index=min(1, len(numeric_features)-1), label_visibility="visible")
            
            # Determine if multiclass
            y_proba = best_model.predict_proba(X_test.iloc[:1])
            n_classes = y_proba.shape[1]
            
            # For multiclass, let user select which class to visualize
            selected_class = 1  # Default to class 1 (typically "positive" class)
            if n_classes > 2:  # Multiclass
                selected_class = st.selectbox(
                    "Select class for heatmap:", 
                    range(n_classes),
                    format_func=lambda i: f"Class {i} " + (["Draw", "Home Win", "Away Win"][i] if i < 3 else ""),
                    label_visibility="visible"
                )
                class_name = ["Draw", "Home Win", "Away Win"][selected_class] if selected_class < 3 else f"Class {selected_class}"
                st.info(f"Showing heatmap for probability of {class_name}")
            
            # Define range for each feature using quantiles
            feat1_vals = np.linspace(X_test[feat1].quantile(0.05), X_test[feat1].quantile(0.95), 20)
            feat2_vals = np.linspace(X_test[feat2].quantile(0.05), X_test[feat2].quantile(0.95), 20)
            
            # Create heatmap data
            heatmap_data = np.zeros((len(feat2_vals), len(feat1_vals)))
            instance_base = X_test.iloc[0].copy()
            
            # Generate predictions for each combination of feature values
            for i, val1 in enumerate(feat1_vals):
                for j, val2 in enumerate(feat2_vals):
                    instance_temp = instance_base.copy()
                    instance_temp[feat1] = val1
                    instance_temp[feat2] = val2
                    pred = best_model.predict_proba(instance_temp.values.reshape(1, -1))[0, selected_class]
                    heatmap_data[j, i] = pred
            
            # Create and display heatmap
            fig_heat = px.imshow(
                heatmap_data, 
                x=feat1_vals, 
                y=feat2_vals, 
                labels={'x': feat1, 'y': feat2, 'color': f"Probability of {class_name if n_classes > 2 else 'Class 1'}"},
                title=f"Multivariate Sensitivity Analysis - Effect on {class_name if n_classes > 2 else 'Class 1'} Probability",
                color_continuous_scale="viridis"
            )
            
            # Add annotations for clearer visualization
            fig_heat.update_layout(
                xaxis_title=feat1,
                yaxis_title=feat2,
                coloraxis_colorbar=dict(
                    title="Probability",
                )
            )
            
            st.plotly_chart(fig_heat)
        else:
            st.warning("Not enough numeric features available for multivariate analysis.")

##############################################
# Section 11: Additional Evaluation Metrics
##############################################
if page == "Additional Evaluation Metrics":
    st.title("Additional Evaluation Metrics")
    if 'X_test' not in st.session_state or 'y_test' not in st.session_state or 'best_model' not in st.session_state:
        st.error("Please complete the modeling in the 'Modeling & Performance' section first!")
    else:
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        best_model = st.session_state['best_model']
        
        st.header("Lift Curve")
        # Calculate lift curve data
        try:
            # Handle multiclass differently
            y_proba = best_model.predict_proba(X_test)
            
            # For multiclass, use the probability of the actual class
            if y_proba.shape[1] > 2:  # Multiclass case
                # Convert y_test to numpy array for easier indexing
                y_test_np = np.array(y_test)
                
                # Get probabilities corresponding to the actual class for each instance
                y_proba_for_true_class = np.array([y_proba[i, y_test_np[i]] for i in range(len(y_test_np))])
                
                # Sort by probability
                sorted_idx = np.argsort(-y_proba_for_true_class)
            else:  # Binary case - use probability of class 1
                sorted_idx = np.argsort(-y_proba[:, 1])
            
            # Calculate lift
            sorted_y = y_test.iloc[sorted_idx].reset_index(drop=True)
            cum_positives = np.cumsum(sorted_y == 1)  # Count positives for binary or count specific class for multiclass
            total_positives = sum(sorted_y == 1)
            
            if total_positives > 0:  # Avoid division by zero
                percentile = np.linspace(1/len(sorted_y), 1.0, len(sorted_y))
                gain = cum_positives / total_positives
                # Avoid division by zero for the first percentile
                lift = gain / percentile
                
                fig_lift, ax_lift = plt.subplots()
                ax_lift.plot(percentile, lift, label='Lift Curve')
                ax_lift.set_xlabel("Percentile")
                ax_lift.set_ylabel("Lift")
                ax_lift.set_title("Lift Curve")
                ax_lift.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig_lift, bbox_inches="tight")
                plt.clf()
            else:
                st.warning("Not enough positive class examples to calculate lift curve.")
        except Exception as e:
            st.error(f"Error generating lift curve: {e}")
            st.info("This may be due to incompatible probabilities or class labels.")

        st.header("Learning Curve with Bias/Variance Analysis")
        try:
            from sklearn.model_selection import learning_curve
            train_sizes, train_scores, test_scores = learning_curve(
                best_model, X_test, y_test, cv=5, 
                train_sizes=np.linspace(0.1, 1.0, 5),
                scoring='accuracy'  # Use accuracy for multiclass
            )
            
            train_std = np.std(train_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            fig_lv, ax_lv = plt.subplots()
            ax_lv.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training Score")
            ax_lv.fill_between(train_sizes, np.mean(train_scores, axis=1)-train_std, np.mean(train_scores, axis=1)+train_std, alpha=0.2, color="r")
            ax_lv.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="Validation Score")
            ax_lv.fill_between(train_sizes, np.mean(test_scores, axis=1)-test_std, np.mean(test_scores, axis=1)+test_std, alpha=0.2, color="g")
            ax_lv.set_xlabel("Training Size")
            ax_lv.set_ylabel("Score")
            ax_lv.set_title("Learning Curve with Bias-Variance Analysis")
            ax_lv.legend(loc="best")
            ax_lv.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_lv, bbox_inches="tight")
            plt.clf()
        except Exception as e:
            st.error(f"Error generating learning curve: {e}")

        st.header("Bootstrap Confidence Intervals for Predictions")
        try:
            # Use bootstrapping to estimate the confidence interval of predictions
            # For multiclass, we'll show confidence intervals for each class probability
            n_bootstraps = 100
            y_proba = best_model.predict_proba(X_test)
            num_classes = y_proba.shape[1]
            
            if num_classes <= 3:  # Only do this for 2-3 classes to keep it manageable
                class_names = ["Draw", "Home Win", "Away Win"][:num_classes]
                
                # Create columns for multiple plots side by side
                cols = st.columns(num_classes)
                
                # For each class, calculate bootstrap confidence intervals
                for class_idx in range(num_classes):
                    boot_means = []
                    rng = np.random.RandomState(42)
                    
                    for i in range(n_bootstraps):
                        indices = rng.choice(range(len(X_test)), size=len(X_test), replace=True)
                        boot_X = X_test.iloc[indices]
                        boot_pred = best_model.predict_proba(boot_X)[:, class_idx]
                        boot_means.append(np.mean(boot_pred))
                    
                    boot_means = np.array(boot_means)
                    ci_lower = np.percentile(boot_means, 2.5)
                    ci_upper = np.percentile(boot_means, 97.5)
                    
                    with cols[class_idx]:
                        st.write(f"**{class_names[class_idx]}**")
                        st.write(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                        fig_ci = px.histogram(
                            boot_means, 
                            nbins=20, 
                            title=f"Bootstrap Distribution - {class_names[class_idx]}",
                            labels={'x': 'Mean Probability'}
                        )
                        st.plotly_chart(fig_ci, use_container_width=True)
            else:
                st.info("Too many classes to display individual bootstrap distributions. Showing overall accuracy instead.")
                
                # Calculate bootstrap confidence intervals for accuracy
                boot_accs = []
                rng = np.random.RandomState(42)
                
                for i in range(n_bootstraps):
                    indices = rng.choice(range(len(X_test)), size=len(X_test), replace=True)
                    boot_X = X_test.iloc[indices]
                    boot_y = y_test.iloc[indices]
                    boot_pred = best_model.predict(boot_X)
                    boot_accs.append(np.mean(boot_pred == boot_y))
                
                boot_accs = np.array(boot_accs)
                ci_lower = np.percentile(boot_accs, 2.5)
                ci_upper = np.percentile(boot_accs, 97.5)
                
                st.write(f"Bootstrap Confidence Interval for Accuracy (95%): [{ci_lower:.3f}, {ci_upper:.3f}]")
                fig_ci = px.histogram(boot_accs, nbins=20, title="Bootstrap Distribution of Accuracy")
                st.plotly_chart(fig_ci)
        except Exception as e:
            st.error(f"Error generating bootstrap confidence intervals: {e}")

##############################################
# Section 12: Executive Summary with AI
##############################################
if page == "Relatório":
    st.title("Relatório Automático de Análise")
    if 'result_df' not in st.session_state or 'best_model' not in st.session_state:
        st.error("Por favor, complete o treinamento de modelos e análises antes de gerar o relatório!")
    else:
        # Crie um layout estruturado para o relatório
        st.markdown("""
        <div style="background-color: var(--card-background); padding: 10px; border-radius: 5px; margin-bottom: 20px;">
            <h3 style="color: var(--primary-color); margin-top: 0;">Análise Técnica Avançada de Dados e Modelos Preditivos</h3>
            <p>Este relatório contém uma análise detalhada do desempenho dos modelos, métricas avançadas, 
            interpretação dos resultados e recomendações técnicas baseadas em evidências.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuração do tipo de relatório
        report_config = st.expander("⚙️ Configurações do Relatório", expanded=False)
        with report_config:
            col1, col2 = st.columns(2)
            with col1:
                summary_type_option = st.radio("Selecione o Tipo de Relatório", 
                                           ("Relatório Técnico Avançado", 
                                            "Relatório Estratégico de Apostas", 
                                            "Análise Completa com Recomendações"),
                                           label_visibility="visible")
            with col2:
                detail_level = st.select_slider("Nível de Detalhe", 
                                              options=["Conciso", "Detalhado", "Muito Detalhado", "Extremamente Detalhado"],
                                              value="Muito Detalhado")
        
        # Aggregate summarized information from the results with more detail
        metrics_info = "Desempenho do Modelo (Métricas Detalhadas):\n"
        
        try:
            metrics_dict = st.session_state['result_df'].to_dict()
            metrics_info += f"Acurácia: {metrics_dict.get('accuracy', {}).get(0, 'N/A')}\n"
            metrics_info += f"Precisão: {metrics_dict.get('precision', {}).get(0, 'N/A')}\n"
            metrics_info += f"Recall: {metrics_dict.get('recall', {}).get(0, 'N/A')}\n"
            metrics_info += f"F1-Score: {metrics_dict.get('f1', {}).get(0, 'N/A')}\n"
            metrics_info += f"ROC AUC: {metrics_dict.get('roc_auc', {}).get(0, 'N/A')}\n"
            metrics_info += f"Log Loss: {metrics_dict.get('log_loss', {}).get(0, 'N/A')}\n"
            
            # Adicionar informações sobre o melhor modelo
            best_model = st.session_state['best_model']
            model_type = type(best_model).__name__
            metrics_info += f"\nMelhor Modelo: {model_type}\n"
            
            if hasattr(best_model, 'get_params'):
                params = best_model.get_params()
                metrics_info += "Parâmetros do Modelo:\n"
                for param, value in params.items():
                    metrics_info += f"- {param}: {value}\n"
        except Exception as e:
            metrics_info += f"Erro ao processar métricas detalhadas: {str(e)}\n"
            metrics_info += str(st.session_state['result_df'].to_dict())
        
        # Use sports betting terminology with more technical details
        shap_insights = (
            "Análise de Importância de Features (SHAP):\n"
            "As análises de SHAP indicam a contribuição de cada feature para as previsões do modelo. "
            "Valores positivos de SHAP sugerem uma maior probabilidade de um resultado específico, "
            "enquanto valores negativos reduzem essa probabilidade. "
            "Features com distribuições mais amplas de valores SHAP têm maior impacto nas previsões. "
            "Padrões de interação entre features podem ser identificados nas análises de dependência.\n"
        )
        
        # Adicionar informação sobre as features mais importantes, se disponível
        if 'feature_importance' in st.session_state:
            shap_insights += "\nImportância das Features (Top 10):\n"
            for feature, importance in st.session_state.get('feature_importance', {}).items()[:10]:
                shap_insights += f"- {feature}: {importance}\n"
                
        learning_curve_info = (
            "Análise de Curva de Aprendizado:\n"
            "A curva de aprendizado demonstra a evolução do desempenho do modelo em relação ao tamanho do conjunto de treinamento. "
            "A convergência entre o desempenho de treinamento e validação indica a capacidade de generalização do modelo. "
            "A estabilização das métricas com mais dados sugere que o modelo está capturando adequadamente os padrões subjacentes. "
            "Oscilações nas curvas podem indicar sensibilidade a outliers ou variabilidade nos dados.\n"
        )
        
        sensitivity_info = (
            "Análise de Sensibilidade Multivariada:\n"
            "A análise de sensibilidade demonstra como as previsões do modelo respondem a mudanças nas variáveis de entrada. "
            "Relações não-lineares entre features e outputs são visualizadas através de gráficos de dependência parcial. "
            "Interações complexas entre múltiplas features são quantificadas e visualizadas em heatmaps. "
            "A estabilidade das previsões do modelo é avaliada em diferentes cenários e combinações de features. "
            "Estas análises permitem identificar pontos de alavancagem onde pequenas mudanças nas features causam grandes impactos nos resultados.\n"
        )
        
        # Summarize match data with more statistical insights
        match_info = ""
        if 'data' in st.session_state:
            data = st.session_state['data']
            try:
                match_info = "Análise Estatística dos Dados:\n"
                
                # Informações básicas do dataset
                total_matches = len(data)
                match_info += f"Total de Partidas Analisadas: {total_matches}\n"
                
                # Distribuição de resultados
                if 'match_outcome' in data.columns:
                    home_wins = len(data[data['match_outcome'] == 1])
                    away_wins = len(data[data['match_outcome'] == 2])
                    draws = len(data[data['match_outcome'] == 0])
                    
                    match_info += f"Vitórias em Casa: {home_wins} ({home_wins/total_matches*100:.2f}%)\n"
                    match_info += f"Empates: {draws} ({draws/total_matches*100:.2f}%)\n"
                    match_info += f"Vitórias Fora: {away_wins} ({away_wins/total_matches*100:.2f}%)\n\n"
                
                # Análise de gols
                if 'home_goals' in data.columns and 'away_goals' in data.columns:
                    total_home_goals = data['home_goals'].sum()
                    total_away_goals = data['away_goals'].sum()
                    avg_home_goals = data['home_goals'].mean()
                    avg_away_goals = data['away_goals'].mean()
                    
                    # Usar numpy para garantir um valor escalar
                    import numpy as np
                    avg_home_goals_val = np.mean(avg_home_goals)
                    avg_away_goals_val = np.mean(avg_away_goals)
                    avg_total_goals = np.mean(avg_home_goals_val + avg_away_goals_val)
                    
                    match_info += f"Total de Gols em Casa: {total_home_goals} (Média: {avg_home_goals_val:.2f} por jogo)\n"
                    match_info += f"Total de Gols Fora: {total_away_goals} (Média: {avg_away_goals_val:.2f} por jogo)\n"
                    match_info += f"Total de Gols: {total_home_goals + total_away_goals} (Média: {avg_total_goals:.2f} por jogo)\n\n"
                    
                    # Análise de Under/Over
                    under_2_5 = len(data[(data['home_goals'] + data['away_goals']) < 2.5])
                    over_2_5 = len(data[(data['home_goals'] + data['away_goals']) >= 2.5])
                    
                    match_info += f"Jogos Under 2.5: {under_2_5} ({under_2_5/total_matches*100:.2f}%)\n"
                    match_info += f"Jogos Over 2.5: {over_2_5} ({over_2_5/total_matches*100:.2f}%)\n\n"
                
                # Análise por temporada, se disponível
                if 'year' in data.columns:
                    match_info += "Distribuição por Temporada:\n"
                    season_counts = data['year'].value_counts().sort_index()
                    for season, count in season_counts.items():
                        match_info += f"- {season}: {count} jogos\n"
                
                # Análise de odds, se disponível
                if 'ODD Fecho H' in data.columns:
                    match_info += "\nAnálise de Odds:\n"
                    avg_home_odds = data['ODD Fecho H'].mean()
                    avg_draw_odds = data['ODD Fecho D'].mean() if 'ODD Fecho D' in data.columns else 'N/A'
                    avg_away_odds = data['ODD Fecho A'].mean() if 'ODD Fecho A' in data.columns else 'N/A'
                    
                    # Usar numpy para garantir um valor escalar se for uma Series
                    import numpy as np
                    avg_home_odds_val = np.mean(avg_home_odds)
                    avg_draw_odds_val = np.mean(avg_draw_odds) if avg_draw_odds != 'N/A' else 'N/A'
                    avg_away_odds_val = np.mean(avg_away_odds) if avg_away_odds != 'N/A' else 'N/A'
                    
                    match_info += f"Odds Médias para Vitória em Casa: {avg_home_odds_val:.2f}\n"
                    match_info += f"Odds Médias para Empate: {avg_draw_odds_val if avg_draw_odds_val == 'N/A' else f'{avg_draw_odds_val:.2f}'}\n"
                    match_info += f"Odds Médias para Vitória Fora: {avg_away_odds_val if avg_away_odds_val == 'N/A' else f'{avg_away_odds_val:.2f}'}\n"
            except Exception as e:
                match_info += f"\nErro ao processar algumas estatísticas: {str(e)}\n"
        
        # Function to generate a detailed prompt with clear sections and chain-of-thought instructions
        def generate_detailed_prompt(summary_type, metrics_info, shap_insights, learning_curve_info, sensitivity_info, match_info, detail_level):
            # Definir o nível de detalhe solicitado
            detail_multiplier = {
                "Conciso": "condensado e objetivo, focando apenas nos pontos mais importantes",
                "Detalhado": "com boa profundidade técnica e análise detalhada dos principais insights",
                "Muito Detalhado": "com alta profundidade técnica, análises aprofundadas e múltiplas perspectivas sobre os dados",
                "Extremamente Detalhado": "extremamente técnico e aprofundado, com análises extensivas, múltiplas perspectivas e recomendações baseadas em evidências"
            }
            
            detail_instruction = detail_multiplier.get(detail_level, "detalhado")
            
            system_prompt = (
                "Você é um analista sênior especializado em ciência de dados para apostas esportivas, "
                "com vasto conhecimento em estatística, machine learning e análise preditiva. "
                f"Forneça uma análise {detail_instruction}. "
                "Use linguagem técnica e precisa, com terminologia específica da área. "
                "Identifique padrões e tendências significativas. "
                "Base suas conclusões diretamente nos dados e métricas fornecidos."
            )
            
            if summary_type == "Relatório Técnico Avançado":
                prompt = (
                    "Por favor, produza um relatório técnico avançado seguindo esta estrutura:\n\n"
                    "# 1. RESUMO EXECUTIVO\n"
                    "Sintetize os achados principais e conclusões de maior relevância estatística.\n\n"
                    "# 2. AVALIAÇÃO DO MODELO\n"
                    "## 2.1 Métricas de Desempenho\n"
                    "Analise detalhadamente cada métrica, explicando suas implicações técnicas e práticas.\n"
                    "## 2.2 Análise Comparativa\n"
                    "Compare os resultados com benchmarks relevantes e explique o significado das diferenças.\n\n"
                    "# 3. ANÁLISE DE FEATURES\n"
                    "## 3.1 Importância e Influência\n"
                    "Detalhe o impacto de cada feature nas previsões, usando as análises SHAP.\n"
                    "## 3.2 Padrões de Interação\n"
                    "Descreva como as features interagem e o impacto dessas interações nos resultados.\n\n"
                    "# 4. AVALIAÇÃO ESTATÍSTICA DOS DADOS\n"
                    "Analise as distribuições, correlações e tendências nos dados de partidas.\n\n"
                    "# 5. LIMITAÇÕES E CONSIDERAÇÕES\n"
                    "Discuta possíveis limitações estatísticas e técnicas da modelagem e análise.\n\n"
                    "# 6. RECOMENDAÇÕES\n"
                    "Forneça recomendações técnicas baseadas nas análises para melhorar a precisão preditiva.\n\n"
                    "=== DADOS PARA ANÁLISE ===\n"
                    "=== Métricas do Modelo ===\n" + metrics_info + "\n"
                    "=== Análise SHAP ===\n" + shap_insights + "\n"
                    "=== Curva de Aprendizado ===\n" + learning_curve_info + "\n"
                    "=== Análise de Sensibilidade ===\n" + sensitivity_info + "\n"
                    "=== Estatísticas das Partidas ===\n" + match_info + "\n"
                )
            elif summary_type == "Relatório Estratégico de Apostas":
                prompt = (
                    "Por favor, produza um relatório estratégico para apostas esportivas seguindo esta estrutura:\n\n"
                    "# 1. SUMÁRIO ESTRATÉGICO\n"
                    "Destaque as principais oportunidades de apostas identificadas pela análise de dados.\n\n"
                    "# 2. ANÁLISE DE VALOR ESPERADO\n"
                    "## 2.1 Identificação de Valor\n"
                    "Explique onde o modelo identifica discrepâncias entre probabilidades previstas e odds de mercado.\n"
                    "## 2.2 Quantificação de Vantagem\n"
                    "Quantifique a vantagem estatística identificada em diferentes cenários de apostas.\n\n"
                    "# 3. PADRÕES DE MERCADO\n"
                    "## 3.1 Tendências Identificadas\n"
                    "Detalhe as principais tendências e padrões identificados nos dados históricos.\n"
                    "## 3.2 Ineficiências de Mercado\n"
                    "Identifique possíveis ineficiências sistemáticas em mercados específicos.\n\n"
                    "# 4. FRAMEWORK DE APOSTAS\n"
                    "## A. Gerenciamento de Banca\n"
                    "Recomendações para alocação otimizada de capital baseada nas probabilidades do modelo.\n"
                    "## B. Limiares de Apostas\n"
                    "Sugira valores de corte para apostas baseados no desempenho do modelo.\n\n"
                    "# 5. CONSIDERAÇÕES DE RISCO\n"
                    "Discuta a volatilidade estatística e estratégias para minimizar riscos sistemáticos.\n\n"
                    "# 6. PLANO DE IMPLEMENTAÇÃO\n"
                    "Forneça um roadmap para implementação progressiva da estratégia baseada em dados.\n\n"
                    "=== DADOS PARA ANÁLISE ===\n"
                    "=== Métricas do Modelo ===\n" + metrics_info + "\n"
                    "=== Análise SHAP ===\n" + shap_insights + "\n"
                    "=== Curva de Aprendizado ===\n" + learning_curve_info + "\n"
                    "=== Análise de Sensibilidade ===\n" + sensitivity_info + "\n"
                    "=== Estatísticas das Partidas ===\n" + match_info + "\n"
                )
            else:  # "Análise Completa com Recomendações"
                prompt = (
                    "Por favor, produza uma análise completa com recomendações seguindo esta estrutura:\n\n"
                    "# 1. RESUMO TÉCNICO E ESTRATÉGICO\n"
                    "Sintetize tanto os aspectos técnicos quanto estratégicos das análises realizadas.\n\n"
                    "# 2. AVALIAÇÃO TÉCNICA DO MODELO\n"
                    "## 2.1 Performance Preditiva\n"
                    "Analise detalhadamente o desempenho do modelo, suas forças e limitações.\n"
                    "## 2.2 Robustez e Generalização\n"
                    "Avalie a capacidade do modelo de generalizar para dados não vistos.\n\n"
                    "# 3. INSIGHTS ESTRATÉGICOS\n"
                    "## 3.1 Padrões de Alto Valor\n"
                    "Identifique combinações de condições que consistentemente geram previsões de alta confiança.\n"
                    "## 3.2 Tendências Emergentes\n"
                    "Detecte tendências temporais ou situacionais nos dados que possam indicar oportunidades.\n\n"
                    "# 4. ANÁLISE SETORIAL\n"
                    "## 4.1 Dados por Liga/Competição\n"
                    "Detalhe variações de performance do modelo entre diferentes ligas ou competições.\n"
                    "## 4.2 Características Específicas\n"
                    "Identifique características únicas de certos mercados ou competições.\n\n"
                    "# 5. AVALIAÇÃO DE RISCO-RETORNO\n"
                    "Quantifique o trade-off entre confiança preditiva e retorno potencial em diferentes cenários.\n\n"
                    "# 6. RECOMENDAÇÕES ACIONÁVEIS\n"
                    "## 6.1 Melhorias Técnicas\n"
                    "Sugira aprimoramentos específicos para a coleta de dados e modelagem.\n"
                    "## 6.2 Implementação Estratégica\n"
                    "Forneça um framework de decisão para aplicação prática dos insights.\n\n"
                    "# 7. PRÓXIMOS PASSOS\n"
                    "Outline um plano de ação detalhado para maximizar o valor das análises realizadas.\n\n"
                    "=== DADOS PARA ANÁLISE ===\n"
                    "=== Métricas do Modelo ===\n" + metrics_info + "\n"
                    "=== Análise SHAP ===\n" + shap_insights + "\n"
                    "=== Curva de Aprendizado ===\n" + learning_curve_info + "\n"
                    "=== Análise de Sensibilidade ===\n" + sensitivity_info + "\n"
                    "=== Estatísticas das Partidas ===\n" + match_info + "\n"
                )
            return system_prompt, prompt
        
        system_prompt, final_prompt = generate_detailed_prompt(
            summary_type_option,
            metrics_info,
            shap_insights,
            learning_curve_info,
            sensitivity_info,
            match_info,
            detail_level
        )
        
        # Removendo a exibição do prompt
        # st.markdown("### Generated Prompt:")
        # st.code(final_prompt)
        
        with st.spinner("Gerando relatório completo com IA..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-2024-11-20",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000  # Aumentando o tamanho para gerar respostas mais detalhadas
                )
                executive_summary = response.choices[0].message.content.strip()
                
                # Convertendo o markdown para HTML para melhor formatação
                st.markdown(executive_summary, unsafe_allow_html=True)
                
                # Botão para download do relatório em formato markdown
                report_md = executive_summary
                st.download_button(
                    label="📥 Download do Relatório Completo (Markdown)",
                    data=report_md,
                    file_name=f"relatorio_apostas_{summary_type_option.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                )
            except Exception as e:
                st.error(f"Erro ao gerar o relatório: {e}")

##############################################
# Section 13: Virtual Assistant
##############################################
if page == "Virtual Assistant":
    # Importar as bibliotecas necessárias para o assistente
    import time
    import random
    
    # Dicionário de conceitos de apostas para cards
    BETTING_CONCEPTS = {
        "odds": {
            "title": "Odds",
            "definition": "Representam a probabilidade implícita de um evento acontecer e determinam o valor do pagamento em caso de vitória.",
            "example": "Odds de 2.0 significam 50% de probabilidade implícita e pagam R$200 para cada R$100 apostados.",
            "tip": "Compare odds de diferentes casas de apostas para encontrar o melhor valor."
        },
        "valor esperado": {
            "title": "Valor Esperado (EV)",
            "definition": "O retorno médio que você pode esperar de uma aposta ao longo do tempo.",
            "example": "Se você faz uma aposta de R$100 com 60% de chance de ganhar R$200, seu EV é: (0.6 × R$200) - (0.4 × R$100) = R$80",
            "tip": "Aposte apenas quando o valor esperado for positivo, indicando vantagem estatística."
        },
        "gestão de banca": {
            "title": "Gestão de Banca",
            "definition": "Estratégia para administrar o capital disponível para apostas, minimizando riscos e maximizando retornos.",
            "example": "Método de Kelly: apostar uma porcentagem do bankroll baseada na vantagem percebida.",
            "tip": "Nunca aposte mais de 5% do seu bankroll total em um único evento, independente da confiança."
        },
        "handicap": {
            "title": "Handicap",
            "definition": "Vantagem ou desvantagem atribuída a um time para equilibrar as probabilidades.",
            "example": "Handicap -1.5 para o favorito significa que ele precisa vencer por pelo menos 2 gols.",
            "tip": "Os handicaps asiáticos eliminam o empate e podem reduzir o risco em apostas com favoritos claros."
        },
        "over under": {
            "title": "Over/Under",
            "definition": "Aposta no total combinado de pontos/gols/etc. em um jogo, acima ou abaixo de um valor definido.",
            "example": "Over 2.5 gols significa apostar que haverá 3 ou mais gols na partida.",
            "tip": "Analise as médias de gols recentes, lesões de jogadores ofensivos/defensivos, e condições climáticas."
        }
    }
    
    # Aplicar CSS personalizado para melhorar o design
    st.markdown("""
    <style>
    /* Estilo para balões de chat do assistente */
    [data-testid="stChatMessageContent"] {
        border-radius: 15px;
        padding: 10px 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 5px;
    }
    
    /* Estilo para mensagens do assistente */
    [data-testid="stChatMessage"] [data-testid="AIMessageContainer"] [data-testid="stChatMessageContent"] {
        background-color: #1E3A8A;
        color: white;
        border-top-left-radius: 0;
        position: relative;
    }
    [data-testid="stChatMessage"] [data-testid="AIMessageContainer"] [data-testid="stChatMessageContent"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: -10px;
        width: 0;
        height: 0;
        border-top: 10px solid #1E3A8A;
        border-left: 10px solid transparent;
    }
    
    /* Estilo para mensagens do usuário */
    [data-testid="stChatMessage"] [data-testid="HumanMessageContainer"] [data-testid="stChatMessageContent"] {
        background-color: #F0F7FF;
        border-top-right-radius: 0;
        position: relative;
        border: 1px solid #E1E8ED;
    }
    [data-testid="stChatMessage"] [data-testid="HumanMessageContainer"] [data-testid="stChatMessageContent"]::before {
        content: "";
        position: absolute;
        top: 0;
        right: -10px;
        width: 0;
        height: 0;
        border-top: 10px solid #F0F7FF;
        border-right: 10px solid transparent;
    }
    
    /* Animação de fadeIn */
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Estilização de card informativo */
    .info-card {
        background-color: #1A1A1A !important; 
        border-left: 5px solid #18A558 !important; 
        padding: 15px !important; 
        margin: 15px 0 !important; 
        border-radius: 5px !important;
        animation: fadeIn 0.8s ease-in !important;
        color: #FFFFFF !important; /* Garantir que o texto seja branco */
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }
    
    .info-card h4 {
        color: #18A558 !important; 
        margin-top: 0 !important;
        font-weight: bold !important;
    }
    
    .info-card p {
        color: #FFFFFF !important; /* Garantir que o texto do parágrafo seja branco */
        margin-bottom: 10px !important;
    }
    
    .info-card strong {
        color: #18A558 !important; /* Destacar o texto em negrito com a cor da marca */
        font-weight: bold !important;
    }
    
    .info-card .tip-box {
        background-color: #1E3A8A; 
        padding: 12px !important; 
        color: white !important; 
        border-radius: 5px !important;
        margin-top: 10px !important;
    }
    
    .info-card .tip-box p {
        color: #FFFFFF !important; /* Garantir que o texto na caixa de dica seja branco */
        margin: 0 !important;
        font-weight: 500 !important;
    }
    
    .info-card .tip-box p span {
        color: #FFFFFF !important; /* TODOS os spans dentro da caixa de dica serão brancos */
    }
    
    .info-card .tip-box .dica-label,
    .info-card .tip-box span:first-child {
        color: #FFEB3B !important; /* Amarelo brilhante APENAS para o label da dica */
        font-weight: bold !important;
    }
    
    .assistant-avatar {
        background: #1E3A8A;
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        text-align: center;
        line-height: 40px;
        font-weight: bold;
        font-size: 20px;
    }
    
    /* Melhora nos espaçamentos */
    .stChatInputContainer {
        padding-top: 1rem;
        border-top: 1px solid #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Assistente Virtual Big Data Bet")
    
    st.markdown("""
    <div style='background-color:#1E3A8A; padding:20px; border-radius:10px; margin-bottom:20px;'>
        <h3 style='color:white; margin-top:0;'>🔥 E aí, galera da Big Data Bet! Vamos falar de apostas? 💰</h3>
        <p style='color:white;'>
        Fala, apostador! Sou o assistente virtual da <b>Big Data Bet</b>, seu parceiro para dominar o mercado de apostas esportivas com ciência de dados e estatísticas avançadas!
        </p>
        <p style='color:white;'>
        Pode me perguntar <b>qualquer coisa</b> sobre odds, probabilidades, modelos preditivos, estratégias de apostas ou como interpretar dados esportivos. 
        Estou aqui para turbinar suas apostas com insights baseados em dados! 📊🏆
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Função para criar cards informativos
    def create_info_card(concept):
        if concept.lower() not in BETTING_CONCEPTS:
            return ""
            
        info = BETTING_CONCEPTS[concept.lower()]
        card_html = f"""
        <div style="background-color: #1A1A1A; border-left: 5px solid #18A558; padding: 15px; margin: 15px 0; border-radius: 5px; color: #FFFFFF; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
            <h4 style="color: #18A558; margin-top: 0; font-weight: bold;">{info['title']}</h4>
            <p style="color: #FFFFFF; margin-bottom: 10px;"><strong style="color: #18A558;">Definição:</strong> {info['definition']}</p>
            <p style="color: #FFFFFF; margin-bottom: 10px;"><strong style="color: #18A558;">Exemplo:</strong> {info['example']}</p>
            <div style="background-color: #1E3A8A; padding: 12px; border-radius: 5px; margin-top: 10px;">
                <p style="color: #FFFFFF; margin: 0; font-weight: 500;">
                    <span style="color: #FFEB3B; font-weight: bold;">💡 Dica Big Data Bet:</span> 
                    <span style="color: #FFFFFF; font-weight: 500;">{info['tip']}</span>
                </p>
            </div>
        </div>
        """
        return card_html
    
    # Função para efeito de digitação
    def typing_animation():
        with st.chat_message("assistant", avatar="🎲"):
            placeholder = st.empty()
            for i in range(5):
                dots = "." * (i % 4)
                placeholder.markdown(f"Digitando{dots}", unsafe_allow_html=True)
                time.sleep(0.3)
            placeholder.empty()
    
    # Função para detectar conceitos de apostas na resposta
    def detect_concepts(text):
        detected = []
        for concept in BETTING_CONCEPTS.keys():
            if concept in text.lower():
                detected.append(concept)
        return detected[:2]  # Limitando a 2 conceitos para não sobrecarregar a UI
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="🎲"):
                st.markdown(message["content"], unsafe_allow_html=True)
                
                # Mostrar cards informativos associados à mensagem, se houver
                if "cards" in message and message["cards"]:
                    for card in message["cards"]:
                        st.markdown(card, unsafe_allow_html=True)
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
    
    # Get user query
    user_query = st.chat_input("E aí, o que você quer saber sobre apostas esportivas e modelos preditivos? 🎲")
    
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Process the question and generate a response
        with st.spinner(""):
            try:
                # Simular efeito de digitação 
                typing_animation()
                
                # Check if it's a simple greeting
                simple_greetings = ["oi", "olá", "ola", "e ai", "eai", "fala", "bom dia", "boa tarde", "boa noite", "hi", "hello"]
                
                if user_query.lower().strip() in simple_greetings or user_query.lower().strip() + "!" in simple_greetings:
                    # Provide a short greeting response
                    ai_response = "<div class='fade-in'>E aí! Aqui é o assistente da Big Data Bet Sports Market! 🔥 Como posso ajudar com suas apostas esportivas hoje? Tá precisando de alguma dica ou quer saber mais sobre odds, estratégias ou modelos preditivos? É só mandar! 📊🏆</div>"
                    
                    # Add message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                    # Display the response
                    with st.chat_message("assistant", avatar="🎲"):
                        st.markdown(ai_response, unsafe_allow_html=True)
                else:
                    # Create prompt for GPT for non-greeting queries
                    system_content = (
                        "Você é o assistente virtual da Big Data Bet, especialista em mercado esportivo, ciência de dados e modelos preditivos. "
                        "Seu tom é jovial, dinâmico e energético, como um verdadeiro entusiasta de apostas esportivas e análise de dados. "
                        "MUITO IMPORTANTE: Sempre inicie TODAS as suas respostas se apresentando como o 'Assistente da Big Data Bet' antes de responder qualquer pergunta. "
                        "Por exemplo: 'Fala, apostador! Aqui é o Assistente da Big Data Bet! Sobre sua pergunta...' "
                        "Você está sempre pronto para ajudar com curiosidades, explicações e insights sobre odds, probabilidades, modelos estatísticos, "
                        "estratégias de apostas e tendências do mercado esportivo. "
                        "Mesmo sem acesso a dados específicos, você pode oferecer conselhos gerais e explicações sobre conceitos "
                        "de apostas esportivas, interpretação de odds, value bets, modelos estatísticos e estratégias de gestão de banca. "
                        "Sempre mostre entusiasmo pelo potencial das análises de dados no mercado de apostas!"
                    )
                    
                    # Conseguir informações contextuais, se disponíveis
                    context_info = ""
                    if 'processed_data' in st.session_state:
                        processed_data = st.session_state['processed_data']
                        
                        # Adicione um pequeno resumo dos dados, se houver
                        if not processed_data.empty:
                            context_info = f"Há dados de {len(processed_data)} partidas disponíveis para análise. "
                            
                            # Adicione informação sobre modelos, se disponível
                            if 'models' in st.session_state:
                                context_info += f"Existem {len(st.session_state['models'])} modelos treinados. "
                    
                    user_content = f"Pergunta do usuário: {user_query}\n\n"
                    
                    # Se houver contexto, adicione-o, caso contrário informe que é uma resposta genérica
                    if context_info:
                        user_content += f"Informações contextuais: {context_info}\n\n"
                    else:
                        user_content += "Por favor, responda de forma genérica sobre conceitos de apostas esportivas e análise de dados.\n\n"
                    
                    # Generate AI response
                    response = client.chat.completions.create(
                        model="gpt-4o-2024-11-20",
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    ai_text_response = response.choices[0].message.content.strip()
                    
                    # Wrap response in fade-in div
                    ai_response = f"<div class='fade-in'>{ai_text_response}</div>"
                    
                    # Detectar se a resposta menciona conceitos que temos cards
                    concepts = detect_concepts(user_query + " " + ai_text_response)
                    cards = [create_info_card(concept) for concept in concepts]
                    cards = [card for card in cards if card]  # Remove cards vazios
                    
                    # Add message to chat history with cards
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": ai_response,
                        "cards": cards
                    })
                    
                    # Display the response
                    with st.chat_message("assistant", avatar="🎲"):
                        st.markdown(ai_response, unsafe_allow_html=True)
                        
                        # Mostrar cards informativos, se aplicável
                        for card in cards:
                            st.markdown(card, unsafe_allow_html=True)
                
            except Exception as e:
                error_message = f"<div class='fade-in'>Desculpe, não consegui processar essa pergunta: {str(e)}</div>"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                
                with st.chat_message("assistant", avatar="🎲"):
                    st.markdown(error_message, unsafe_allow_html=True)
    else:
        st.info("Digite sua pergunta acima para que eu possa ajudar com suas dúvidas sobre apostas esportivas! 🎯")

def create_styled_card(title, content, step_number=None, card_type="primary"):
    """
    Cria um cartão estilizado com título e conteúdo
    
    Parameters:
    -----------
    title : str
        Título do cartão
    content : str
        Conteúdo HTML do cartão
    step_number : int or None
        Número do passo (opcional)
    card_type : str
        Tipo de cartão: "primary", "secondary", "accent"
    
    Returns:
    --------
    str
        HTML do cartão estilizado
    """
    # Definir cores com base no tipo de cartão
    colors = {
        "primary": {"bg": "#18A558", "border": "#18A558"},
        "secondary": {"bg": "#1E2761", "border": "#1E2761"},
        "accent": {"bg": "#e84855", "border": "#e84855"}
    }
    
    color = colors.get(card_type, colors["primary"])
    
    # Criar o step_badge se o número do passo for fornecido
    step_badge = ""
    if step_number is not None:
        step_badge = f"""
        <div style="background-color: {color['bg']}; color: white; padding: 8px; 
             border-radius: 8px; margin-right: 15px; font-weight: bold; font-size: 1.2em;">
            {step_number:02d}
        </div>
        """
    
    # Montar o HTML do cartão
    card_html = f"""
    <div style="background-color: #1E1E1E; padding: 25px; border-radius: 10px; 
         border-left: 5px solid {color['border']}; margin-bottom: 25px; 
         box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
        
        <div style="display: flex; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #444; padding-bottom: 15px;">
            {step_badge}
            <h2 style="color: {color['bg']}; margin: 0; font-size: 1.6em; font-weight: 600;">
                {title}
            </h2>
        </div>
        
        <div style="color: #FFFFFF; margin: 15px 0; line-height: 1.6; font-size: 1.05em;">
            {content}
        </div>
    </div>
    """
    
    return card_html

##############################################
# Section 14: Sobre o Projeto
##############################################
if page == "Sobre":
    # Card de cabeçalho especial para a página Sobre
    st.markdown("""
    <div style="background: linear-gradient(135deg, #121212, #1E3A8A, #121212); border-radius: 15px; padding: 30px; 
         margin-bottom: 30px; position: relative; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
        
        <!-- Padrão de grid -->
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
             background-image: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px); 
             background-size: 20px 20px; z-index: 0; opacity: 0.3;"></div>
        
        <!-- Logo abstrato -->
        <div style="position: absolute; right: 30px; top: 30px; opacity: 0.1; z-index: 0;">
            <svg width="150" height="150" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="35" stroke="#18A558" stroke-width="2" fill="none" />
                <circle cx="50" cy="50" r="25" stroke="#18A558" stroke-width="2" fill="none" />
                <circle cx="50" cy="50" r="15" stroke="#18A558" stroke-width="2" fill="none" />
                <line x1="20" y1="50" x2="80" y2="50" stroke="#18A558" stroke-width="2" />
                <line x1="50" y1="20" x2="50" y2="80" stroke="#18A558" stroke-width="2" />
            </svg>
        </div>
        
        <h1 style="color: white; font-size: 2.5em; font-weight: 700; margin-bottom: 10px; position: relative; z-index: 1;">
            Big Data Bet<span style="font-weight: 300;"> | Sports Market</span>
        </h1>
        
        <p style="color: #CCC; font-size: 1.2em; max-width: 80%; margin-bottom: 25px; position: relative; z-index: 1;">
            Uma plataforma avançada que combina <strong style="color: #18A558;">ciência de dados</strong>, 
            <strong style="color: #18A558;">machine learning</strong> e <strong style="color: #18A558;">análise estatística</strong> 
            para aprimorar decisões em apostas esportivas.
        </p>
        
        <div style="display: flex; gap: 15px; position: relative; z-index: 1;">
            <div style="background-color: #18A558; color: white; padding: 8px 16px; border-radius: 30px; font-weight: 500; font-size: 0.9em;">
                Versão 1.0
            </div>
            <div style="background-color: rgba(255,255,255,0.1); color: white; padding: 8px 16px; border-radius: 30px; font-weight: 500; font-size: 0.9em;">
                &copy; 2025 Big Data Bet
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Visão Geral
    intro_content = """
    <div style="position: relative; background: linear-gradient(135deg, rgba(24, 165, 88, 0.03), rgba(24, 165, 88, 0.07));
         border-radius: 15px; padding: 25px; margin-bottom: 20px;">
         
        <div style="position: absolute; top: -15px; right: -15px; background-color: rgba(24, 165, 88, 0.1); 
             width: 100px; height: 100px; border-radius: 50%; filter: blur(25px);"></div>
        
        <h2 style="color: #18A558; margin-top: 0; border-bottom: 2px solid rgba(24, 165, 88, 0.2); padding-bottom: 15px; margin-bottom: 20px;">Introdução</h2>
             
        <p style="font-size: 1.1em; line-height: 1.7;">A <strong style="color: #18A558;">Big Data Bet</strong> é uma plataforma avançada que utiliza algoritmos de machine learning para analisar dados históricos de eventos esportivos e identificar padrões que possam auxiliar na tomada de decisões em apostas esportivas.</p>
        
        <p style="font-size: 1.1em; line-height: 1.7;">Desenvolvida para atender tanto apostadores iniciantes quanto profissionais, nossa plataforma transforma dados complexos em insights acionáveis e previsões estatisticamente fundamentadas, ajudando você a fazer apostas mais inteligentes.</p>
        
        <p style="font-size: 1.1em; line-height: 1.7;">Nossa missão é democratizar o acesso à análise avançada de dados esportivos, permitindo que qualquer pessoa possa tomar decisões baseadas em evidências estatísticas, não apenas em intuição ou viés emocional.</p>
        
        <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-top: 25px; justify-content: center;">
            <div style="background-color: rgba(24, 165, 88, 0.05); padding: 20px; border-radius: 12px; text-align: center; flex: 1; min-width: 150px; border: 1px solid rgba(24, 165, 88, 0.1);">
                <div style="font-size: 2em; color: #18A558; margin-bottom: 10px;">🔍</div>
                <h4 style="margin: 0 0 10px 0; color: #18A558;">Análise Preditiva</h4>
                <p style="margin: 0; color: #CCC; font-size: 0.9em;">Modelos probabilísticos para prever resultados esportivos</p>
            </div>
            
            <div style="background-color: rgba(24, 165, 88, 0.05); padding: 20px; border-radius: 12px; text-align: center; flex: 1; min-width: 150px; border: 1px solid rgba(24, 165, 88, 0.1);">
                <div style="font-size: 2em; color: #18A558; margin-bottom: 10px;">📊</div>
                <h4 style="margin: 0 0 10px 0; color: #18A558;">Visualização</h4>
                <p style="margin: 0; color: #CCC; font-size: 0.9em;">Gráficos interativos para explorar e compreender os dados</p>
            </div>
            
            <div style="background-color: rgba(24, 165, 88, 0.05); padding: 20px; border-radius: 12px; text-align: center; flex: 1; min-width: 150px; border: 1px solid rgba(24, 165, 88, 0.1);">
                <div style="font-size: 2em; color: #18A558; margin-bottom: 10px;">🤖</div>
                <h4 style="margin: 0 0 10px 0; color: #18A558;">IA Assistente</h4>
                <p style="margin: 0; color: #CCC; font-size: 0.9em;">Assistente virtual para explicar conceitos e análises</p>
            </div>
        </div>
    </div>
    """
    st.markdown(intro_content, unsafe_allow_html=True)
    
    # Funcionalidades com design moderno
    col1, col2 = st.columns([3, 2])
    
    with col1:
        features_content = """
        <div style="background-color: #1A1A1A; border-radius: 15px; padding: 25px; box-shadow: 0 10px 25px rgba(0,0,0,0.2);">
            <h2 style="color: #18A558; margin-top: 0; border-bottom: 2px solid rgba(24, 165, 88, 0.2); padding-bottom: 15px; margin-bottom: 20px;">Funcionalidades Principais</h2>
            
            <div style="display: flex; margin-bottom: 20px; align-items: flex-start;">
                <div style="background: linear-gradient(135deg, #18A558, #126e3b); color: white; width: 35px; height: 35px; display: flex; justify-content: center; align-items: center; border-radius: 50%; margin-right: 15px; flex-shrink: 0;">1</div>
                <div>
                    <h4 style="color: #18A558; margin: 0 0 8px 0;">Upload & Análise Exploratória de Dados</h4>
                    <p style="margin: 0; color: #CCC; line-height: 1.6;">Carregue seus dados em CSV/Excel e obtenha visualizações estatísticas automáticas para entender distribuições, correlações e tendências.</p>
                </div>
            </div>
            
            <div style="display: flex; margin-bottom: 20px; align-items: flex-start;">
                <div style="background: linear-gradient(135deg, #18A558, #126e3b); color: white; width: 35px; height: 35px; display: flex; justify-content: center; align-items: center; border-radius: 50%; margin-right: 15px; flex-shrink: 0;">2</div>
                <div>
                    <h4 style="color: #18A558; margin: 0 0 8px 0;">Modelagem Preditiva</h4>
                    <p style="margin: 0; color: #CCC; line-height: 1.6;">Diversos algoritmos como XGBoost, Random Forest, redes neurais e modelos bayesianos para previsões precisas de resultados esportivos.</p>
                </div>
            </div>
            
            <div style="display: flex; margin-bottom: 20px; align-items: flex-start;">
                <div style="background: linear-gradient(135deg, #18A558, #126e3b); color: white; width: 35px; height: 35px; display: flex; justify-content: center; align-items: center; border-radius: 50%; margin-right: 15px; flex-shrink: 0;">3</div>
                <div>
                    <h4 style="color: #18A558; margin: 0 0 8px 0;">Explicabilidade de Modelos</h4>
                    <p style="margin: 0; color: #CCC; line-height: 1.6;">Tecnologias SHAP e LIME para entender os fatores que influenciam cada previsão, tornando as "caixas-pretas" do ML transparentes.</p>
                </div>
            </div>
            
            <div style="display: flex; margin-bottom: 20px; align-items: flex-start;">
                <div style="background: linear-gradient(135deg, #18A558, #126e3b); color: white; width: 35px; height: 35px; display: flex; justify-content: center; align-items: center; border-radius: 50%; margin-right: 15px; flex-shrink: 0;">4</div>
                <div>
                    <h4 style="color: #18A558; margin: 0 0 8px 0;">Simulação de Cenários</h4>
                    <p style="margin: 0; color: #CCC; line-height: 1.6;">Análises "e se?" para avaliar impactos de diferentes variáveis nos resultados e testar estratégias de apostas em cenários controlados.</p>
                </div>
            </div>
            
            <div style="display: flex; align-items: flex-start;">
                <div style="background: linear-gradient(135deg, #18A558, #126e3b); color: white; width: 35px; height: 35px; display: flex; justify-content: center; align-items: center; border-radius: 50%; margin-right: 15px; flex-shrink: 0;">5</div>
                <div>
                    <h4 style="color: #18A558; margin: 0 0 8px 0;">Assistente Virtual</h4>
                    <p style="margin: 0; color: #CCC; line-height: 1.6;">Inteligência artificial para responder dúvidas sobre apostas, explicar conceitos estatísticos e auxiliar na interpretação dos resultados dos modelos.</p>
                </div>
            </div>
        </div>
        """
        st.markdown(features_content, unsafe_allow_html=True)
    
    with col2:
        tech_content = """
        <div style="background-color: #1A1A1A; border-radius: 15px; padding: 25px; box-shadow: 0 10px 25px rgba(0,0,0,0.2); height: 100%;">
            <h2 style="color: #1E3A8A; margin-top: 0; border-bottom: 2px solid rgba(30, 58, 138, 0.2); padding-bottom: 15px; margin-bottom: 20px;">Tecnologias</h2>
            
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 25px;">
                <span style="background-color: rgba(30, 58, 138, 0.1); color: white; padding: 10px 15px; border-radius: 30px; font-size: 0.9em; border: 1px solid rgba(30, 58, 138, 0.2);">Python</span>
                <span style="background-color: rgba(30, 58, 138, 0.1); color: white; padding: 10px 15px; border-radius: 30px; font-size: 0.9em; border: 1px solid rgba(30, 58, 138, 0.2);">Streamlit</span>
                <span style="background-color: rgba(30, 58, 138, 0.1); color: white; padding: 10px 15px; border-radius: 30px; font-size: 0.9em; border: 1px solid rgba(30, 58, 138, 0.2);">Pandas</span>
                <span style="background-color: rgba(30, 58, 138, 0.1); color: white; padding: 10px 15px; border-radius: 30px; font-size: 0.9em; border: 1px solid rgba(30, 58, 138, 0.2);">Scikit-learn</span>
                <span style="background-color: rgba(30, 58, 138, 0.1); color: white; padding: 10px 15px; border-radius: 30px; font-size: 0.9em; border: 1px solid rgba(30, 58, 138, 0.2);">XGBoost</span>
                <span style="background-color: rgba(30, 58, 138, 0.1); color: white; padding: 10px 15px; border-radius: 30px; font-size: 0.9em; border: 1px solid rgba(30, 58, 138, 0.2);">SHAP</span>
                <span style="background-color: rgba(30, 58, 138, 0.1); color: white; padding: 10px 15px; border-radius: 30px; font-size: 0.9em; border: 1px solid rgba(30, 58, 138, 0.2);">LIME</span>
                <span style="background-color: rgba(30, 58, 138, 0.1); color: white; padding: 10px 15px; border-radius: 30px; font-size: 0.9em; border: 1px solid rgba(30, 58, 138, 0.2);">Plotly</span>
                <span style="background-color: rgba(30, 58, 138, 0.1); color: white; padding: 10px 15px; border-radius: 30px; font-size: 0.9em; border: 1px solid rgba(30, 58, 138, 0.2);">OpenAI</span>
            </div>
            
            <div style="background: linear-gradient(135deg, rgba(30, 58, 138, 0.05), rgba(30, 58, 138, 0.1)); padding: 20px; border-radius: 10px; border-left: 3px solid #1E3A8A;">
                <h4 style="color: #1E3A8A; margin-top: 0; margin-bottom: 10px;">Arquitetura do Sistema</h4>
                <p style="color: #CCC; margin: 0 0 15px 0; line-height: 1.6; font-size: 0.95em;">
                    A plataforma utiliza uma arquitetura modular, com componentes especializados para:
                </p>
                <ul style="color: #CCC; margin: 0; padding-left: 20px; line-height: 1.6; font-size: 0.95em;">
                    <li>Processamento e limpeza de dados</li>
                    <li>Engenharia de features</li>
                    <li>Modelagem preditiva</li>
                    <li>Interpretação de modelos</li>
                    <li>Visualização interativa</li>
                    <li>Interface de usuário</li>
                    <li>IA conversacional</li>
                </ul>
            </div>
            
            <div style="background: linear-gradient(135deg, rgba(30, 58, 138, 0.05), rgba(30, 58, 138, 0.1)); padding: 20px; border-radius: 10px; border-left: 3px solid #1E3A8A; margin-top: 20px;">
                <h4 style="color: #1E3A8A; margin-top: 0; margin-bottom: 10px;">Stack de Desenvolvimento</h4>
                <ul style="color: #CCC; margin: 0; padding-left: 20px; line-height: 1.6; font-size: 0.95em;">
                    <li><strong style="color: #1E3A8A;">Frontend:</strong> Streamlit</li>
                    <li><strong style="color: #1E3A8A;">Backend:</strong> Python</li>
                    <li><strong style="color: #1E3A8A;">Análise de Dados:</strong> Pandas, NumPy</li>
                    <li><strong style="color: #1E3A8A;">Machine Learning:</strong> Scikit-learn, XGBoost, TensorFlow</li>
                    <li><strong style="color: #1E3A8A;">Visualização:</strong> Plotly, Matplotlib, Seaborn</li>
                    <li><strong style="color: #1E3A8A;">IA:</strong> OpenAI, SHAP, LIME</li>
                </ul>
            </div>
        </div>
        """
        st.markdown(tech_content, unsafe_allow_html=True)
    
    # Metodologia e Abordagem Científica
    st.markdown("""
    <div style="background-color: #1A1A1A; border-radius: 15px; padding: 25px; box-shadow: 0 10px 25px rgba(0,0,0,0.2); margin-top: 30px; margin-bottom: 30px;">
        <h2 style="color: #e84855; margin-top: 0; border-bottom: 2px solid rgba(232, 72, 85, 0.2); padding-bottom: 15px; margin-bottom: 20px;">Nossa Metodologia</h2>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
            <div style="background-color: rgba(232, 72, 85, 0.05); padding: 20px; border-radius: 10px; border-left: 3px solid #e84855;">
                <h4 style="color: #e84855; margin-top: 0; margin-bottom: 10px;">1. Coleta e Preparação de Dados</h4>
                <p style="color: #CCC; margin: 0; line-height: 1.6; font-size: 0.95em;">
                    Reunimos dados históricos de múltiplas fontes, validamos sua qualidade e realizamos limpeza e normalização para garantir consistência.
                </p>
            </div>
            
            <div style="background-color: rgba(232, 72, 85, 0.05); padding: 20px; border-radius: 10px; border-left: 3px solid #e84855;">
                <h4 style="color: #e84855; margin-top: 0; margin-bottom: 10px;">2. Engenharia de Features</h4>
                <p style="color: #CCC; margin: 0; line-height: 1.6; font-size: 0.95em;">
                    Criamos características preditivas poderosas a partir dos dados brutos, utilizando conhecimento especializado sobre fatores que influenciam resultados esportivos.
                </p>
            </div>
            
            <div style="background-color: rgba(232, 72, 85, 0.05); padding: 20px; border-radius: 10px; border-left: 3px solid #e84855;">
                <h4 style="color: #e84855; margin-top: 0; margin-bottom: 10px;">3. Modelagem Preditiva</h4>
                <p style="color: #CCC; margin: 0; line-height: 1.6; font-size: 0.95em;">
                    Aplicamos múltiplos algoritmos de ML e técnicas estatísticas avançadas, validando com metodologias rigorosas de teste e calibração.
                </p>
            </div>
            
            <div style="background-color: rgba(232, 72, 85, 0.05); padding: 20px; border-radius: 10px; border-left: 3px solid #e84855;">
                <h4 style="color: #e84855; margin-top: 0; margin-bottom: 10px;">4. Explicabilidade e Confiança</h4>
                <p style="color: #CCC; margin: 0; line-height: 1.6; font-size: 0.95em;">
                    Fornecemos insights detalhados sobre por que cada previsão foi feita, permitindo que você confie nos resultados e tome decisões informadas.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Nota final
    st.markdown("""
    <div style="background-color: rgba(24, 165, 88, 0.05); border-left: 4px solid #18A558; padding: 20px; 
         margin: 30px 0; border-radius: 0 15px 15px 0; position: relative; overflow: hidden;">
         
        <div style="position: absolute; bottom: -20px; right: -20px; background-color: rgba(24, 165, 88, 0.1); 
             width: 100px; height: 100px; border-radius: 50%; filter: blur(30px);"></div>
             
        <h3 style="color: #18A558; margin-top: 0; margin-bottom: 15px;">Nota Importante</h3>
        <p style="color: #EEE; margin: 0; line-height: 1.7;">
            O Big Data Bet Sport Market foi desenvolvido com fins educacionais e de pesquisa em ciência de dados aplicada.
            As previsões e análises devem ser utilizadas como complemento a uma estratégia de apostas responsável, 
            e não como única fonte de decisão. Sempre lembre-se de que apostas envolvem riscos e devem ser praticadas 
            com responsabilidade financeira.
        </p>
    </div>
    
    <div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin-top: 40px; padding: 20px; background-color: rgba(255,255,255,0.03); border-radius: 10px;">
        <div style="text-align: center;">
            <p style="color: #999; margin: 0; font-size: 0.9em;">© 2025 Big Data Bet | Todos os direitos reservados</p>
            <p style="color: #999; margin: 5px 0 0 0; font-size: 0.9em;">Versão 1.0.0 | Desenvolvido com 💚 pelo Time Big Data Bet</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

##############################################
# Section: Arbitrage Calculator
##############################################
elif page == "Arbitrage Calculator":
    # Section title with improved custom styling
    st.markdown("""
    <div class="section-header">
        <h1 style="margin: 0; padding: 0; font-size: 1.8em; font-weight: 600;">
            Arbitrage Calculator
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Informational card
    st.markdown("""
    <div class="info-card">
        <h4>Sports Betting Arbitrage Calculator</h4>
        <p>An arbitrage opportunity exists when bookmakers offer different odds on the same event, allowing you to place bets on all possible outcomes and guarantee a profit regardless of the result.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different types of arbitrage calculations
    arb_tab1, arb_tab2 = st.tabs(["2-Way Arbitrage", "3-Way Arbitrage"])
    
    with arb_tab1:
        st.markdown("### 2-Way Arbitrage Calculator (e.g., Tennis, Boxing)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            outcome1_name = st.text_input("Outcome 1 Name", "Player/Team A")
            outcome1_odds = st.number_input("Outcome 1 Odds", min_value=1.01, value=2.0, step=0.01, format="%.2f")
            bookmaker1 = st.text_input("Bookmaker 1", "Bookmaker A")
            
        with col2:
            outcome2_name = st.text_input("Outcome 2 Name", "Player/Team B")
            outcome2_odds = st.number_input("Outcome 2 Odds", min_value=1.01, value=2.0, step=0.01, format="%.2f")
            bookmaker2 = st.text_input("Bookmaker 2", "Bookmaker B")
        
        # Calculate arbitrage opportunity
        # The arbitrage formula: 1/odds1 + 1/odds2 < 1 means there's an arbitrage opportunity
        total_probability = (1 / outcome1_odds) + (1 / outcome2_odds)
        
        # Calculate profit margin
        arb_margin = (1 - total_probability) * 100
        
        # Calculate the optimal stake distribution for a given total stake
        total_stake = st.number_input("Total Investment ($)", min_value=10.0, value=1000.0, step=10.0)
        
        stake1 = (total_stake / outcome1_odds) / total_probability * outcome1_odds
        stake2 = (total_stake / outcome2_odds) / total_probability * outcome2_odds
        
        # Display results
        st.markdown("### Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if arb_margin > 0:
                st.metric(
                    "Arbitrage Opportunity", 
                    f"{arb_margin:.2f}% profit", 
                    "Yes, profitable arbitrage exists!"
                )
            else:
                st.metric(
                    "Arbitrage Opportunity", 
                    f"{arb_margin:.2f}% (loss)", 
                    "No arbitrage opportunity"
                )
        
        with col2:
            st.metric(
                "Total Market Percentage", 
                f"{total_probability * 100:.2f}%",
                f"{100 - (total_probability * 100):.2f}% margin" if arb_margin > 0 else None
            )
        
        with col3:
            guaranteed_profit = total_stake * (arb_margin / 100) if arb_margin > 0 else 0
            st.metric(
                "Guaranteed Profit", 
                f"${guaranteed_profit:.2f}",
                f"{(guaranteed_profit / total_stake) * 100:.2f}% ROI" if arb_margin > 0 else "No profit"
            )
        
        # Show stake distribution
        st.markdown("### Stake Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style='background-color: rgba(24, 165, 88, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #18A558;'>
                <h4 style='color: #18A558; margin-top: 0;'>{outcome1_name} ({bookmaker1})</h4>
                <p style='font-size: 1.2em; font-weight: bold;'>${stake1:.2f} ({stake1/total_stake*100:.2f}%)</p>
                <p>Potential Return: ${stake1 * outcome1_odds:.2f}</p>
                <p>Profit: ${stake1 * outcome1_odds - total_stake:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background-color: rgba(30, 58, 138, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #1E3A8A;'>
                <h4 style='color: #1E3A8A; margin-top: 0;'>{outcome2_name} ({bookmaker2})</h4>
                <p style='font-size: 1.2em; font-weight: bold;'>${stake2:.2f} ({stake2/total_stake*100:.2f}%)</p>
                <p>Potential Return: ${stake2 * outcome2_odds:.2f}</p>
                <p>Profit: ${stake2 * outcome2_odds - total_stake:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualize the stake distribution
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=[f"{outcome1_name} ({bookmaker1})", f"{outcome2_name} ({bookmaker2})"],
            values=[stake1, stake2],
            marker_colors=['#18A558', '#1E3A8A'],
            textinfo='percent+value',
            hole=0.4
        ))
        
        fig.update_layout(
            title='Stake Distribution',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig)
    
    with arb_tab2:
        st.markdown("### 3-Way Arbitrage Calculator (e.g., Soccer/Football)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            home_team = st.text_input("Home Team", "Team A")
            home_odds = st.number_input("Home Win Odds", min_value=1.01, value=2.5, step=0.01, format="%.2f")
            bookmaker_home = st.text_input("Bookmaker (Home)", "Bookmaker A")
            
        with col2:
            draw_odds = st.number_input("Draw Odds", min_value=1.01, value=3.4, step=0.01, format="%.2f")
            bookmaker_draw = st.text_input("Bookmaker (Draw)", "Bookmaker B")
            
        with col3:
            away_team = st.text_input("Away Team", "Team B")
            away_odds = st.number_input("Away Win Odds", min_value=1.01, value=2.9, step=0.01, format="%.2f")
            bookmaker_away = st.text_input("Bookmaker (Away)", "Bookmaker C")
        
        # Calculate 3-way arbitrage
        total_probability_3way = (1 / home_odds) + (1 / draw_odds) + (1 / away_odds)
        arb_margin_3way = (1 - total_probability_3way) * 100
        
        # Calculate optimal stake distribution
        total_stake_3way = st.number_input("Total Investment ($)", min_value=10.0, value=1000.0, step=10.0, key="3way_stake")
        
        home_stake = (total_stake_3way / home_odds) / total_probability_3way * home_odds
        draw_stake = (total_stake_3way / draw_odds) / total_probability_3way * draw_odds
        away_stake = (total_stake_3way / away_odds) / total_probability_3way * away_odds
        
        # Display results
        st.markdown("### Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if arb_margin_3way > 0:
                st.metric(
                    "Arbitrage Opportunity", 
                    f"{arb_margin_3way:.2f}% profit", 
                    "Yes, profitable arbitrage exists!"
                )
            else:
                st.metric(
                    "Arbitrage Opportunity", 
                    f"{arb_margin_3way:.2f}% (loss)", 
                    "No arbitrage opportunity"
                )
        
        with col2:
            st.metric(
                "Total Market Percentage", 
                f"{total_probability_3way * 100:.2f}%",
                f"{100 - (total_probability_3way * 100):.2f}% margin" if arb_margin_3way > 0 else None
            )
        
        with col3:
            guaranteed_profit_3way = total_stake_3way * (arb_margin_3way / 100) if arb_margin_3way > 0 else 0
            st.metric(
                "Guaranteed Profit", 
                f"${guaranteed_profit_3way:.2f}",
                f"{(guaranteed_profit_3way / total_stake_3way) * 100:.2f}% ROI" if arb_margin_3way > 0 else "No profit"
            )
        
        # Show stake distribution
        st.markdown("### Stake Distribution")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style='background-color: rgba(24, 165, 88, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #18A558;'>
                <h4 style='color: #18A558; margin-top: 0;'>{home_team} Win ({bookmaker_home})</h4>
                <p style='font-size: 1.2em; font-weight: bold;'>${home_stake:.2f} ({home_stake/total_stake_3way*100:.2f}%)</p>
                <p>Potential Return: ${home_stake * home_odds:.2f}</p>
                <p>Profit: ${home_stake * home_odds - total_stake_3way:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background-color: rgba(232, 72, 85, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #e84855;'>
                <h4 style='color: #e84855; margin-top: 0;'>Draw ({bookmaker_draw})</h4>
                <p style='font-size: 1.2em; font-weight: bold;'>${draw_stake:.2f} ({draw_stake/total_stake_3way*100:.2f}%)</p>
                <p>Potential Return: ${draw_stake * draw_odds:.2f}</p>
                <p>Profit: ${draw_stake * draw_odds - total_stake_3way:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='background-color: rgba(30, 58, 138, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #1E3A8A;'>
                <h4 style='color: #1E3A8A; margin-top: 0;'>{away_team} Win ({bookmaker_away})</h4>
                <p style='font-size: 1.2em; font-weight: bold;'>${away_stake:.2f} ({away_stake/total_stake_3way*100:.2f}%)</p>
                <p>Potential Return: ${away_stake * away_odds:.2f}</p>
                <p>Profit: ${away_stake * away_odds - total_stake_3way:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualize the stake distribution
        fig2 = go.Figure()
        
        fig2.add_trace(go.Pie(
            labels=[f"{home_team} Win ({bookmaker_home})", f"Draw ({bookmaker_draw})", f"{away_team} Win ({bookmaker_away})"],
            values=[home_stake, draw_stake, away_stake],
            marker_colors=['#18A558', '#e84855', '#1E3A8A'],
            textinfo='percent+value',
            hole=0.4
        ))
        
        fig2.update_layout(
            title='Stake Distribution',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig2)

##############################################
# Section: Value Bet Detector
##############################################
elif page == "Value Bet Detector":
    # Section title with improved custom styling
    st.markdown("""
    <div class="section-header">
        <h1 style="margin: 0; padding: 0; font-size: 1.8em; font-weight: 600;">
            Value Bet Detector
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Informational card
    st.markdown("""
    <div class="info-card">
        <h4>Find Value Bets with Positive Expected Value</h4>
        <p>A value bet exists when the probability implied by bookmaker odds is lower than the true probability of the outcome occurring. This tool helps you identify and analyze value betting opportunities.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Bookmaker Odds")
        bookmaker_odds = st.number_input("Bookmaker Odds", min_value=1.01, value=2.0, step=0.01, format="%.2f")
        bookmaker_name = st.text_input("Bookmaker Name", "Bookmaker X")
        st.markdown("### Your Assessment")
        true_probability = st.slider("Your Estimated Win Probability (%)", min_value=1.0, max_value=99.0, value=50.0, step=0.1) / 100
        confidence_level = st.slider("Confidence Level in Your Assessment", min_value=1, max_value=10, value=7)
        
    with col2:
        st.markdown("### Event Details")
        event_name = st.text_input("Event Name", "Team A vs Team B")
        outcome_name = st.text_input("Outcome to Analyze", "Team A to Win")
        market_type = st.selectbox("Market Type", ["1X2 (Win-Draw-Win)", "Over/Under", "Asian Handicap", "Both Teams to Score", "Other"])
        st.markdown("### Betting Parameters")
        bankroll = st.number_input("Current Bankroll ($)", min_value=10.0, value=1000.0, step=100.0)
        stake_method = st.selectbox("Stake Calculation Method", ["Fixed Percentage", "Kelly Criterion", "Fixed Amount"])
    
    # Calculate implied probability from bookmaker odds
    implied_probability = 1 / bookmaker_odds
    
    # Calculate edge
    edge_percentage = (true_probability - implied_probability) / implied_probability * 100
    
    # Calculate expected value
    expected_value = (true_probability * (bookmaker_odds - 1)) - (1 - true_probability)
    
    # Calculate recommended stake
    if stake_method == "Fixed Percentage":
        recommended_percentage = st.slider("Percentage of Bankroll", min_value=0.5, max_value=5.0, value=2.0, step=0.5) / 100
        recommended_stake = bankroll * recommended_percentage
    elif stake_method == "Kelly Criterion":
        kelly_fraction = st.slider("Kelly Fraction", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        # Kelly formula: f = (bp - q) / b where b = odds-1, p = probability, q = 1-p
        b = bookmaker_odds - 1
        p = true_probability
        q = 1 - p
        f = (b * p - q) / b if b > 0 else 0
        f = max(0, min(1, f)) * kelly_fraction  # Limit to [0,1] and apply fraction
        recommended_stake = bankroll * f
    else:  # Fixed Amount
        recommended_stake = st.number_input("Fixed Stake Amount ($)", min_value=1.0, value=50.0, step=5.0)
    
    # Button to analyze value
    if st.button("Analyze Value Bet"):
        # Display results in cards
        st.markdown("## Value Bet Analysis")
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Implied Probability", 
                f"{implied_probability*100:.2f}%",
                f"{(true_probability - implied_probability) * 100:.2f}%" if true_probability > implied_probability else f"{(true_probability - implied_probability) * 100:.2f}%"
            )
        
        with col2:
            value_label = "Value Bet" if edge_percentage > 0 else "No Value"
            value_delta = f"+{edge_percentage:.2f}%" if edge_percentage > 0 else f"{edge_percentage:.2f}%"
            st.metric(
                "Edge", 
                value_label,
                value_delta
            )
        
        with col3:
            ev_label = "Positive EV" if expected_value > 0 else "Negative EV"
            st.metric(
                "Expected Value", 
                ev_label,
                f"{expected_value:.4f}"
            )
        
        with col4:
            st.metric(
                "Recommended Stake", 
                f"${recommended_stake:.2f}",
                f"{recommended_stake/bankroll*100:.2f}% of bankroll"
            )
        
        # Value bet visualization
        st.markdown("### Value Visualization")
        
        # Create a more detailed analysis card
        value_color = "#18A558" if edge_percentage > 0 else "#e84855"
        confidence_adjusted_ev = expected_value * (confidence_level / 10)
        
        st.markdown(f"""
        <div style='background-color: rgba({value_color.lstrip('#')[:2]}, {value_color.lstrip('#')[2:4]}, {value_color.lstrip('#')[4:]}, 0.1); 
             padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid {value_color};'>
            
            <h3 style='color: {value_color}; margin-top: 0;'>
                {outcome_name} - {'Value Bet Detected! ✓' if edge_percentage > 0 else 'No Value Found ✗'}
            </h3>
            
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;'>
                <div>
                    <p style='font-weight: bold; margin-bottom: 5px;'>Event</p>
                    <p style='margin-top: 0;'>{event_name}</p>
                </div>
                
                <div>
                    <p style='font-weight: bold; margin-bottom: 5px;'>Market</p>
                    <p style='margin-top: 0;'>{market_type}</p>
                </div>
                
                <div>
                    <p style='font-weight: bold; margin-bottom: 5px;'>Bookmaker</p>
                    <p style='margin-top: 0;'>{bookmaker_name} @ {bookmaker_odds:.2f}</p>
                </div>
                
                <div>
                    <p style='font-weight: bold; margin-bottom: 5px;'>Your Assessment</p>
                    <p style='margin-top: 0;'>{true_probability*100:.2f}% (Confidence: {confidence_level}/10)</p>
                </div>
            </div>
            
            <div style='margin-top: 20px;'>
                <p style='font-weight: bold;'>Analysis</p>
                <ul style='margin-top: 5px;'>
                    <li>Edge: {edge_percentage:.2f}%</li>
                    <li>Raw Expected Value: {expected_value:.4f}</li>
                    <li>Confidence-Adjusted EV: {confidence_adjusted_ev:.4f}</li>
                    <li>Recommended Stake: ${recommended_stake:.2f}</li>
                    <li>Potential Return: ${recommended_stake * bookmaker_odds:.2f}</li>
                    <li>Potential Profit: ${recommended_stake * (bookmaker_odds - 1):.2f}</li>
                </ul>
            </div>
            
            <div style='margin-top: 20px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);'>
                <p style='font-weight: bold;'>Value Rating</p>
                <div style='background-color: #333; height: 10px; border-radius: 5px; overflow: hidden; margin-top: 10px;'>
                    <div style='background-color: {value_color}; height: 100%; width: {min(max(edge_percentage * 3, 5), 100) if edge_percentage > 0 else 5}%;'></div>
                </div>
                <div style='display: flex; justify-content: space-between; margin-top: 5px;'>
                    <span>No Value</span>
                    <span>Strong Value</span>
                </div>
            </div>
            
            <div style='margin-top: 20px;'>
                <p style='font-weight: bold;'>Recommendation</p>
                <p style='margin-top: 5px;'>
                    {
                        "This appears to be a strong value bet. Consider placing a stake according to your recommended amount." 
                        if edge_percentage > 5 else 
                        "This appears to be a slight value bet. Consider a reduced stake." 
                        if edge_percentage > 0 else
                        "This does not appear to be a value bet. Avoid placing this bet."
                    }
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualize the probabilities
        fig = go.Figure()
        
        # Create a bar chart comparing probabilities
        labels = ["Your Estimate", "Bookmaker"]
        values = [true_probability * 100, implied_probability * 100]
        colors = ["#18A558", "#1E3A8A"]
        
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}%" for v in values],
            textposition="auto"
        ))
        
        fig.update_layout(
            title="Probability Comparison",
            yaxis_title="Probability (%)",
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig)
        
        # Show long-term expected results
        st.markdown("### Long-term Simulation")
        
        # Simulate results
        num_simulations = 1000
        results = []
        bankroll_evolution = [bankroll]
        current_bankroll = bankroll
        
        for i in range(num_simulations):
            # Simulate bet outcome based on true probability
            outcome = np.random.random() < true_probability
            
            # Calculate stake based on current bankroll
            if stake_method == "Fixed Percentage":
                stake = current_bankroll * recommended_percentage
            elif stake_method == "Kelly Criterion":
                stake = current_bankroll * f
            else:  # Fixed Amount
                stake = min(recommended_stake, current_bankroll)  # Don't bet more than current bankroll
            
            # Update bankroll
            if outcome:  # Win
                current_bankroll += stake * (bookmaker_odds - 1)
                results.append(stake * (bookmaker_odds - 1))
            else:  # Loss
                current_bankroll -= stake
                results.append(-stake)
            
            # Record current bankroll
            bankroll_evolution.append(current_bankroll)
            
            # Break if bankrupt
            if current_bankroll <= 0:
                break
        
        # Plot bankroll evolution
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=list(range(len(bankroll_evolution))),
            y=bankroll_evolution,
            mode='lines',
            name='Bankroll',
            line=dict(color='#18A558', width=2)
        ))
        
        # Add initial bankroll reference line
        fig2.add_shape(
            type="line",
            x0=0,
            y0=bankroll,
            x1=len(bankroll_evolution) - 1,
            y1=bankroll,
            line=dict(color="white", width=1, dash="dash")
        )
        
        fig2.update_layout(
            title='Simulated Bankroll Evolution (1000 bets)',
            xaxis_title='Number of Bets',
            yaxis_title='Bankroll ($)',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig2)
        
        # Calculate summary statistics
        if len(results) > 0:
            win_count = sum(1 for r in results if r > 0)
            win_rate = win_count / len(results) * 100
            avg_win = sum(r for r in results if r > 0) / win_count if win_count > 0 else 0
            avg_loss = sum(abs(r) for r in results if r < 0) / (len(results) - win_count) if (len(results) - win_count) > 0 else 0
            
            # Show summary stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Final Bankroll", f"${current_bankroll:.2f}", f"{(current_bankroll - bankroll):.2f}")
            
            with col2:
                st.metric("Actual Win Rate", f"{win_rate:.2f}%", f"{win_rate - (true_probability * 100):.2f}%")
            
            with col3:
                st.metric("Average Win", f"${avg_win:.2f}")
            
            with col4:
                st.metric("Average Loss", f"${avg_loss:.2f}")
    
    # Educational content about value betting
    with st.expander("Understanding Value Betting"):
        st.markdown("""
        ## What is Value Betting?
        
        Value betting is a strategy where you place bets only when you believe the probability of an outcome is higher than what the bookmaker's odds suggest. Unlike arbitrage betting, which relies on discrepancies between different bookmakers, value betting requires you to develop your own probabilistic model or assessment.
        
        ### The Math Behind Value Betting
        
        #### Implied Probability
        The probability implied by bookmaker odds is calculated as:
        
        `Implied Probability = 1 / Decimal Odds`
        
        For example, odds of 2.50 imply a 40% probability (1 / 2.50 = 0.40 or 40%).
        
        #### Edge Calculation
        Your edge (or value) is calculated by comparing your assessed probability with the implied probability:
        
        `Edge = (Your Probability - Implied Probability) / Implied Probability`
        
        For example, if you think an outcome has a 50% chance of happening, but the odds imply 40%:
        
        `Edge = (0.50 - 0.40) / 0.40 = 0.25 or 25%`
        
        #### Expected Value (EV)
        The expected value of a bet is:
        
        `EV = (Probability of Winning × Profit if Win) - (Probability of Losing × Stake)`
        
        Or more simply:
        
        `EV = (Your Probability × (Odds - 1)) - (1 - Your Probability)`
        
        A positive EV indicates a value bet.
        
        ### Key Principles
        
        1. **Long-term Approach**: Value betting is profitable over the long run, but short-term variance can be high.
        
        2. **Bankroll Management**: Even with positive expected value, proper bankroll management is crucial.
        
        3. **Edge Size Matters**: The larger your edge, the more you should bet (hence Kelly Criterion).
        
        4. **Market Efficiency**: Markets for major events tend to be more efficient (less value), while niche markets often contain more value.
        
        5. **Record Keeping**: Track all your bets to evaluate your actual edge over time.
        
        ### Common Mistakes
        
        - **Overconfidence**: Many bettors overestimate their ability to predict outcomes accurately.
        
        - **Ignoring Vig**: Bookmakers build a margin into their odds. Account for this in your calculations.
        
        - **Chasing Losses**: Stick to your strategy even after a series of losses.
        
        - **Confirmation Bias**: Don't just look for information that confirms your pre-existing beliefs.
        """)
