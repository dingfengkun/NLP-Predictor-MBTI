# NLP-Predictor-MBTI

This project explores the use of natural language processing (NLP) techniques to predict Myers-Briggs Type Indicator (MBTI) personality types based on users’ written text. It aims to combine psychology and machine learning by analyzing linguistic patterns to infer personality traits.

## Motivation

The MBTI is a widely-used personality classification system consisting of 16 types, each represented by a combination of four dichotomies (Introversion/Extraversion, Sensing/Intuition, Thinking/Feeling, Judging/Perceiving). With the increasing availability of user-generated content online, there is growing interest in inferring personality types from text data.

This project is motivated by the desire to understand how language reflects personality and how modern NLP tools can be applied to personality classification tasks. Potential applications include content personalization, career counseling, social media analytics, and digital mental health tools.

## Question

Can we predict a person’s MBTI personality type purely from their text using NLP techniques?

## Dataset

* **Source**: [MBTI Personality Dataset on Kaggle](https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset])
* **Description**: 1.7M records of posts collected from Reddit using Google big query. Each record contains:

  * `type`: The MBTI type of the user (e.g., INFP, ESTJ)
  * `posts`: A concatenation of 50 posts written by the user on an online personality forum

## EDA Highlight

Top 3 most frequent types make up ~40–50% of data (INTP, INTJ, INFJ)
Several types (e.g., ESFJ, ESFP, ESTJ) are rare, leading to biased learning
Posts vary greatly in length and language complexity

## Model

This project includes several models for MBTI type classification:

* **Baseline**: Logistic Regression with TF-IDF vectorized input
* **Model 1**: An LSTM-based neural network using pre-trained embeddings
* **Model 2**: A Random Forest classifier trained on engineered features (e.g., word counts, TF-IDF, or other textual features)

Models are evaluated using metrics such as accuracy and Weighted F1 Score(Evaluates each class equally and adjusts for class frequency— good for imbalance).

## Project Structure

```
NLP-Predictor-MBTI/
├── .gitattributes
├── .gitignore
├── README.md
├── .DS_Store
├── requirements.txt                  # Python package dependencies
├── Report.pdf
├── LR_RF                             # Classic ML models: Logistic Regression & Random Forest
│   ├── rf_lf_opt.ipynb
│   ├── rf_lf.ipynb
├── LSTM/                             # LSTM-based text classification
│   ├── benchmark.py
│   ├── LSTM_optimized.ipynb
│   ├── MBTI 500.csv
│   └── LSTM_parameter/               # Trained model assets and evaluation results
│       ├── best_model.pth
│       ├── label_encoder.pkl
│       ├── tokenizer.pkl
│       └── evaluation/               # Evaluation results
│           ├── confusion_matrix.png
│           └── metrics.txt
│
└── Preprocessing/                    # Data cleaning and exploratory analysis
    ├── EDA.ipynb
    ├── preprocessing.py
    └── preprocessing_optimized.py

```

## Usage

Install the required dependencies:

```bash
pip install -r requirements.txt
```

You can run the models and scripts in two ways:

* **From the command line**:
  Use `python` to execute `.py` files such as:

  ```bash
  python LSTM/benchmark.py
  ```

* **From Jupyter Notebook**:
  Open any `.ipynb` file (e.g., `rf_opt_MBTI_500.ipynb`) using Jupyter to explore results and modify the pipeline interactively:



All outputs (e.g., metrics, confusion matrices) will be printed to the console or visualized within the notebook, depending on the file you execute.


## Future work

* Improve hyperparameter tuning and model regularization
* Explore additional model architectures or ensemble methods
* Deploy model using a web interface or API

## References

* MBTI Dataset: [https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset](https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset)
* Dylan Storey. (2017). MBTI Reddit Personality Dataset.

