# Sentiment Analysis Application

This application extracts comments from an online content source, filters those written in English, and analyzes their sentiment using a modern NLP model. It provides a simple and interactive interface built with Streamlit.

## Features
- Extract comments from a provided link  
- Detect and keep only English comments  
- Perform sentiment analysis (positive, neutral, negative)  
- Display results in an interactive table  
- Visualize sentiment distribution with a bar chart  

## Technologies Used
Python 3.9+, Streamlit, HuggingFace Transformers, langdetect, pandas

## Installation
```bash
git clone <project_url>
cd sentiment-analysis-app
pip install -r requirements.txt
streamlit run app.py
