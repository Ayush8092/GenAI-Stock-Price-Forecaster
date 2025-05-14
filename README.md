# GenAI-Stock-Price-Forecaster

# Stock Market Prediction with Deep Learning

## Project Description

This project compares two deep learning approaches for stock market prediction. First, we implement Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRU), which excel at learning temporal dependencies in sequential data. Second, we enhance predictions using Generative Adversarial Networks (GANs) with a unique architecture - using LSTM/GRU as the generator and CNN as the discriminator. We also incorporate Natural Language Processing (NLP) to analyze financial news sentiment and its impact on stock prices.

## Problem Statement

This research compares and enhances stock prediction algorithms through deep learning innovations. We first benchmark traditional LSTM and GRU networks, then develop an improved GAN architecture using recurrent networks as generators. The key contributions include:
- Comparing GRU vs LSTM performance
- Proposing a novel RNN-CNN GAN structure
- Implementing WGAN-GP for training stability
- Integrating NLP-derived sentiment scores as predictive features

## Technical Approaches

- **LSTM**: Specialized recurrent networks with memory cells that maintain long-term dependencies, ideal for sequential financial data.
- **GRU**: Similar to LSTM but with simplified gating mechanisms, offering faster training while preserving temporal learning capabilities.
- **GAN**: Adversarial framework where our generator (LSTM/GRU) learns to produce realistic forecasts by competing against a CNN discriminator.

## Dataset

The dataset contains Apple Inc. stock data (2497 observations, 36 features) including:
- **Price metrics**: Open, High, Low, Close, Volume
- **Market indices**: NASDAQ, S&P500, FTSE100, etc.
- **Commodities**: Crude Oil, Gold
- **Technical indicators**: Moving Averages, Bollinger Bands, MACD
- **News sentiment scores**: (-1 to 1) from FinBERT analysis

## Feature Engineering

We enhanced raw data with:
- **Technical indicators**: 7/21-day SMAs, EMA, momentum, Bollinger Bands
- **Fourier transforms**: For trend decomposition
- **News sentiment analysis**: Via FinBERT
- **Multi-order reconstructions**: 3/6/9-period absolute/angle components

## Implementation

### Code Structure

1. `Load_data.py`: Processes raw data and adds technical/Fourier features
2. `data_preprocessing.py`: Handles normalization and creates train/test splits (70/30)
3. `Baseline_LSTM.py`: LSTM implementation
4. `Basic_GRU.py`: GRU implementation
5. `Basic_GAN.py`: Standard GAN model
6. `WGAN_GP.py`: Improved Wasserstein GAN with gradient penalty
7. `Test_prediction.py`: Generates predictions from trained models

### Training Protocol

- **Input**: 30 days of historical data (all 36 features)
- **Output**: 3-day ahead predictions
- **Special Testing**: COVID-19 period (unexpected event analysis)

## Requirements

- Python 3.x
- TensorFlow/Keras
- FinBERT (for NLP)
- Standard data science stack (NumPy, Pandas, etc.)
