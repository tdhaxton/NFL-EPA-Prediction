# NFL-EPA-Prediction
Comparison of five neural network architectures to predict Expected Points Added (EPA) per play using 10 years of NFL play-by-play data (2009-2018, ~100 features).

# NFL Expected Points Added Prediction Models

A machine learning study using NFL play-by-play data from 2009–2018 to predict 
Expected Points Added (EPA) on a per-play basis using only pre-snap information.

Completed as the final project for CMPS 5323: Machine Learning, 
Department of Computer Science, Midwestern State University, December 2025.

## Project Overview

Expected Points Added (EPA) measures the change in expected points resulting from 
a given play, conditioned on game situation features such as down, distance, yard 
line, score differential, and time remaining. This project builds and compares five 
neural network architectures trained on a dataset of approximately 100 features to 
predict EPA on a per-play and per-game basis.

## Dataset

- **Source:** Max Horowitz NFL play-by-play dataset (2009–2018), as referenced in 
  Yurko, Ventura, and Horowitz (2019)
- **Features:** 95 features after preprocessing (from original 101)
- **Preprocessing:** Dropped post-snap features, converted categorical values to 
  numerical, applied Z-score normalization, separated and ordered data by game
- **Split:** 70% training / 15% validation / 15% testing

## Models

| Model | Hidden Layers | Activation | Neurons |
|-------|--------------|------------|---------|
| 1 | 2 | ReLU | 16, 8 |
| 2 | 3 | ReLU | 128, 64, 32 |
| 3 | 2 | ReLU | 4, 6 |
| 4 | 2 | TanH | 4, 6 (2-year data subset) |
| 5 | 2 | TanH | 4, 6 (single-team data subset) |

All models used Adam optimization, MSE loss, and a single linear output neuron.

## Results

| Model | Mean MAE/Play | Mean MAE/Game | Std Dev/Play |
|-------|--------------|---------------|--------------|
| 1 | 0.9613 | 16.55 | 0.001185 |
| 2 | 0.9632 | 16.87 | 0.004500 |
| 3 | 0.9583 | 15.89 | 0.005543 |
| 4 | 0.9581 | 16.15 | 0.009597 |
| 5 | 1.013  | 24.46 | 0.009292 |

Model #3 achieved the best per-game accuracy with the lowest standard deviation, 
making it the most consistent and best overall performer. Models #1–4 generalized 
well; Model #5 (single-team subset) performed significantly worse, suggesting 
insufficient training data.

## Repository Structure
├── notebooks/
│   ├── NFL_EPA_ML_Study_Model1_Run1.ipynb
│   ├── NFL_EPA_ML_Study_Model1_Run2.ipynb
│   ├── NFL_EPA_ML_Study_Model1_Run3.ipynb
│   ├── NFL_EPA_ML_Study_Model2_Run1.ipynb
│   ├── ... (16 notebooks total)
│   └── Error_Calculations.ipynb
├── Final_Project_Haxton.pdf
└── README.md

## Key Findings

- Down, yards to gain, quarter, yard line, and timeout features were consistently 
  the most important predictors across models
- Noise introduced by redundant team features likely limited model performance
- Future work could consolidate correlated features and incorporate team strength metrics

## Technologies

Python · TensorFlow/Keras · Google Colab · NumPy · Pandas · Scikit-learn

## Author

**Timothy Haxton**  
MS Computer Science, Midwestern State University  
[LinkedIn](https://linkedin.com/in/tim-haxton) | [GitHub](https://github.com/tdhaxton)