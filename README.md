# Text Classification for Topic Modeling

This repository contains the complete Python script and resources for the "Text Classification for Topic Modeling" project, developed as part of the DLBAIPNLP01 course at IU International University of Applied Sciences.

The project focuses on building a supervised machine learning pipeline to classify documents from the 20 Newsgroups dataset into their respective topics, providing a realistic baseline for content-only classification.

---

## 🚀 Features

- **Data Preprocessing:** A robust pipeline that cleans text by lowercasing, removing punctuation and numbers, filtering stop words, and performing lemmatization.
- **Feature Engineering:** Utilizes TF-IDF vectorization with bigrams and trigrams to capture meaningful phrases.
- **Model Training:** Implements a Multinomial Naive Bayes (MNB) classifier, a strong baseline for text classification tasks.
- **Comprehensive Evaluation:** Provides a detailed classification report with metrics like precision, recall, and F1-score.
- **Progressive Scaling Analysis:** Includes a routine to evaluate and report how model accuracy scales as the training dataset size increases (from 25% to 100%).
- **Visualization:** Generates a professional confusion matrix and bar charts showing the most predictive n-grams for each category, offering deep insights into the model's performance.

---

## 🛠️ Setup and Installation

To run this project, you will need Python 3 installed, along with the libraries listed in `requirements.txt`.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PentaPenguin237/Text-Classification-for-Topic-Modeling.git
    cd Text-Classification-for-Topic-Modeling
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ How to Run

Simply execute the main Python script from your terminal:

```bash
python main.py
```

The script will automatically:
1.  Download the necessary NLTK data (stopwords, wordnet).
2.  Fetch and preprocess the 20 Newsgroups dataset.
3.  Train the classifier on the full training set.
4.  Print the final classification report to the console.
5.  Display the confusion matrix and top n-gram plots.
6.  Run the progressive evaluation and print the results table.

---

## 📊 Results

The model achieves a realistic baseline accuracy of **52%** on the test set after removing all metadata (headers, footers, quotes) to ensure the model learns from content alone. The progressive evaluation shows a clear trend of improving accuracy as the training data size increases.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
