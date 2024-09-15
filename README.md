# Graph-based Imputation and Smoothing for Forecasting with Missing Data: A Deep Learning Approach

## Overview

This repository contains the code and resources for my Master's thesis titled **"Graph-based Imputation and Smoothing for Forecasting with Missing Data: A Deep Learning Approach"**, completed as part of the Statistical Science program during my double degree with USI University. The main focus of this work is to address the challenge of forecasting in Multivariate Time Series (MTS) with missing data using a novel two-stage approach combining graph-based imputation and forecasting models.

## Abstract

In many real-world time series applications, missing data can significantly hinder the performance of forecasting models. My thesis presents a method to overcome this by leveraging graph-based imputation techniques and ensemble learning. The key idea is to impute missing data based on the spatial and temporal relationships between different variables, represented as a graph. By generating a smoothed representation of the original dataset and feeding this into a forecasting model, we enhance the accuracy and robustness of predictions. The repository contains implementations of the graph-based imputation models, forecasting methods, and a comparison of different techniques on benchmark datasets.

## Key Components

1. **Graph-based Imputation**: This approach models the MTS as a graph to estimate missing data points. Nodes represent individual time series, and edges capture the relationships between them. We use soft-smoothing to improve the quality of the imputed data.

2. **Two-Stage Forecasting Approach**: In the first stage, we impute missing values using the graph-based method. In the second stage, the imputed data is passed into forecasting models. This includes traditional time-series models and deep learning approaches like RNNs and GNNs.

3. **Benchmarking and Evaluation**: The thesis benchmarks several imputation and forecasting models on various datasets, comparing their performance in the presence of missing data.


## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/graph-imputation-forecasting.git

