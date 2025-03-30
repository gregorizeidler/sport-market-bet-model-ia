# Big Data Bet Sports Market Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
  <img src="https://img.shields.io/badge/Streamlit-1.0+-red.svg" alt="Streamlit Version"/>
</p>

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
  - [Data Upload & EDA](#-data-upload--exploratory-data-analysis-eda)
  - [Predictive Modeling & Evaluation](#-predictive-modeling--evaluation)
  - [Model Explainability](#-explainability--model-interpretation)
  - [Simulations & Analysis](#️-simulations--sensitivity-analysis)
  - [Reporting](#-report-generation)
  - [AI-Powered Insights](#-executive-summary-with-ai)
  - [Virtual Assistant](#️-virtual-assistant-for-project-insights)
- [Technical Architecture](#-technical-architecture)
- [Installation Guide](#️-installation-guide)
- [Usage Instructions](#-running-the-application)
- [Demonstration](#-demonstration-video)
- [Contributing](#-how-to-contribute)
- [License](#-license)
- [Support](#-support)

## 📌 Overview

This project delivers an enterprise-grade machine learning platform for **sports betting market analysis**, integrating advanced data science technologies including **Streamlit**, **scikit-learn**, **XGBoost**, **SHAP**, and **LIME**. The system provides end-to-end capabilities including automated data ingestion, sophisticated feature engineering, high-performance predictive modeling, and comprehensive model explainability with interactive visualizations and scenario-based simulations.

## 🚀 Key Features

### 📂 Data Upload & Exploratory Data Analysis (EDA)
- Enterprise-grade **XLSX/CSV file upload** with intelligent preprocessing
- Comprehensive **descriptive statistics** and **interactive visualizations** for betting variables
- Advanced **data quality assessment** including missing value detection, outlier identification, and correlation analysis

### 💡 Predictive Modeling & Evaluation
- Production-ready **supervised learning pipeline** with cross-validation
- Multi-algorithm support including:
  - **XGBoost**
  - **RandomForest**
  - **Gradient Boosting**
  - **AdaBoost**
  - **Support Vector Machines**
  - **Multi-layer Perceptron**
  - **Naïve Bayes**
  - **Logistic Regression**
- **SMOTE implementation** for class imbalance correction
- Comprehensive performance evaluation suite:
  - **Accuracy, AUC-ROC, Precision, Recall, F1-score**
  - **Confusion Matrix visualization**
  - **Lift Curve analysis**
- **Automated hyperparameter optimization** with configurable search spaces

### 🔍 Explainability & Model Interpretation
- Enterprise-grade interpretability with **SHAP** and **LIME** integration
- Multi-level visualization toolkit:
  - **SHAP Summary Plot**
  - **Dependence Plot**
  - **Waterfall Plot**
  - **Force Plot**
- Feature importance analysis for decision-making transparency

### ⚙️ Simulations & Sensitivity Analysis
- Advanced **"What-if" scenario modeling** for variable impact assessment
- Parameter adjustment interface for prediction robustness evaluation

### 📊 Report Generation
- Production-quality **HTML reports** featuring:
  - **Statistical summaries**
  - **Model comparison dashboards**
  - **Visual explanation components**

### 🧠 Executive Summary with AI
- **GPT-4 powered insights** delivering executive-level summaries
- Dual reporting system:
  - **Technical analysis** for data science teams
  - **Operational insights** for business stakeholders
- Automated extraction of key insights on feature importance, match trends, and model performance

### 🛡️ Virtual Assistant for Project Insights
- Enterprise-grade **AI assistant** for real-time project consultation
- Contextual understanding of:
  - **Model predictions**
  - **Feature importance rankings**
  - **Historical decision patterns**
- Interactive query system for match analysis, prediction criteria, and model evaluation

## 🏗 Technical Architecture

```
├── app.py               # Main application file with Streamlit interface
├── models/              # Pre-trained models and model utilities
│   ├── xgboost/         # XGBoost model implementations
│   └── ensemble/        # Ensemble model implementations
├── utils/               # Utility functions
│   ├── preprocessing.py # Data preprocessing functions
│   ├── evaluation.py    # Model evaluation utilities
│   └── visualization.py # Visualization components
├── data/                # Sample datasets and data handling
├── reports/             # Generated reports directory
└── config/              # Configuration files
```

## 🛠️ Installation Guide

### Prerequisites
- Python 3.8+
- pip package manager
- 8GB+ RAM recommended for large datasets

### Step 1: Clone the Repository
```sh
git clone https://github.com/your-username/big-data-bet-sports-market.git
cd big-data-bet-sports-market
```

### Step 2: Create a Virtual Environment (Recommended)
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```sh
pip install -r requirements.txt
```

### Troubleshooting
- If you encounter CUDA-related errors with XGBoost, install the CPU-only version:
  ```sh
  pip uninstall xgboost
  pip install xgboost==1.5.0
  ```
- For memory issues with large datasets, adjust the `PYTHONMEM` environment variable

## ▶️ Running the Application

Launch the interactive dashboard:
```sh
streamlit run app.py
```

Access the application via the Streamlit-provided URL (typically http://localhost:8501)

### Configuration Options
- `--server.port=XXXX`: Run on a specific port
- `--server.address=X.X.X.X`: Bind to a specific address
- `--server.headless`: Run without opening a browser

## 🎥 Demonstration Video
Para ver a plataforma em ação, incluindo todos os modelos e recursos de IA:
- [📺 Demonstração Completa](https://drive.google.com/file/d/1sqYFQHfOaASupKKBar_LGKHPjiL5NSsk/view?usp=sharing)

## 🤝 How to Contribute

We welcome enterprise and community contributions. To contribute:

1. **Fork** this repository
2. **Create** a feature branch: `git checkout -b feature/your-feature-name`
3. **Implement** your changes following our coding standards
4. **Add tests** for new functionality
5. **Submit** a pull request with comprehensive description of changes

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Include docstrings for all functions and classes
- Maintain test coverage above 80%
- Update documentation for any new features

### Priority Areas for Contribution
- Advanced ML algorithm implementations
- Enhanced explainability techniques
- Performance optimization for large datasets
- UI/UX improvements for the dashboard

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: [Project Wiki](#)
- **Issues**: Submit via [GitHub Issues](https://github.com/your-username/big-data-bet-sports-market/issues)
- **Commercial Support**: Contact us at support@example.com

---

<p align="center">
  <b>Big Data Bet Sports Market Analysis</b><br>
  Enterprise-grade AI platform for sports betting analytics and decision support
</p>
