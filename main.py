# =============================================================================
# GUGLIELMO LURASCHI SICCA
# Matriculation: 92125339
# Project NLP
# =============================================================================


# =============================================================================
# 1. IMPORTS & INITIAL SETUP
# =============================================================================
import nltk
import string
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# =============================================================================
# 2. DOWNLOAD NLTK DATA (ONLY IF NEEDED)
# =============================================================================
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    print("Downloading NLTK WordNet...")
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4.zip')
except LookupError:
    print("Downloading NLTK OMW-1.4...")
    nltk.download('omw-1.4')


# =============================================================================
# 3. LOAD AND SPLIT THE DATASET
# =============================================================================
print("Loading the 20 Newsgroups dataset...")
newsgroups_data = fetch_20newsgroups(
    subset='all',
    shuffle=True,
    random_state=42,
    remove=('headers', 'footers', 'quotes')
)

print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    newsgroups_data.data,
    newsgroups_data.target,
    test_size=0.2,
    random_state=42,
    stratify=newsgroups_data.target
)

print(f"Total training set size: {len(X_train)} documents")
print(f"Total test set size: {len(X_test)} documents")

# =============================================================================
# 4. DEFINE THE PREPROCESSING FUNCTION
# =============================================================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')).union({
    'paper', 'study', 'research', 'abstract', 'conclusion', 'method',
    'result', 'use', 'using', 'used', 'model', 'system', 'graph',
    'algorithm', 'problem', 'data', 'performance', 'task', 'state',
    'group','space','function','algebra','prove','theory','set',
    'operator','number','approach', 'control', 'learning', 'training',
    'generation', 'feature', 'knowledge', 'prompt', 'architecture',
    'prediction', 'analysis', 'temperature', 'information', 'proposed',
    'challenge', 'point', 'series', 'equation', 'al', 'et', 'ca', 'wa',
    'ha', 'doe', 'did', 'subject', 'organization', 'article', 'line', 'host'
})

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[\d' + string.punctuation + ']', ' ', text)
    tokens = [word for word in text.split() if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# =============================================================================
# 5. APPLY PREPROCESSING AND VECTORIZE
# =============================================================================
print("\nInitializing TfidfVectorizer...")
vectorizer = TfidfVectorizer(
    preprocessor=preprocess_text,
    min_df=5,
    max_df=0.8,
    ngram_range=(2, 3)
)

print("Fitting vectorizer on the full training data and transforming it...")
X_train_tfidf = vectorizer.fit_transform(X_train)

print("Transforming test data...")
X_test_tfidf = vectorizer.transform(X_test)

print("\nPreprocessing and vectorization complete!")
print(f"Shape of the training TF-IDF matrix: {X_train_tfidf.shape}")

# =============================================================================
# 6. TRAIN A CLASSIFICATION MODEL (ON FULL DATASET)
# =============================================================================
print("\nTraining the Multinomial Naive Bayes classifier on the full training set...")
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train_tfidf, y_train)
print("Training complete.")

# =============================================================================
# 7. EVALUATE THE MODEL (ON FULL DATASET)
# =============================================================================
print("\nMaking predictions on the test set...")
y_pred = classifier.predict(X_test_tfidf)

print("\n--- Final Model Performance (Trained on 100% of Data) ---")
report = classification_report(
    y_test, y_pred, target_names=newsgroups_data.target_names
)
print(report)

# =============================================================================
# 8. VISUALIZE THE CONFUSION MATRIX
# =============================================================================
pretty_target_names = [
    'Atheism', 'Computer Graphics', 'MS-Windows (Misc)', 'PC Hardware',
    'Mac Hardware', 'X-Windows', 'For Sale', 'Automobiles', 'Motorcycles',
    'Baseball', 'Hockey', 'Cryptography', 'Electronics', 'Medicine', 'Space',
    'Christianity', 'Guns', 'Middle East Politics', 'General Politics', 'Misc Religion'
]

print("\nGenerating confusion matrix for the final model...")
conf_matrix = confusion_matrix(y_test, y_pred)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(16, 14))
cmap = "plasma"
sns.heatmap(
    conf_matrix, annot=True, fmt='d', cmap=cmap,
    xticklabels=pretty_target_names, yticklabels=pretty_target_names,
    linewidths=.5, ax=ax, cbar_kws={"shrink": .8}
)
ax.set_title('Confusion Matrix: 20 Newsgroups Classification', fontsize=22, pad=20, fontfamily='serif')
ax.set_xlabel('Predicted Label', fontsize=16, labelpad=20, fontfamily='serif')
ax.set_ylabel('True Label', fontsize=16, labelpad=20, fontfamily='serif')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout(pad=3.0)
plt.show()

# =============================================================================
# 9. VISUALIZE TOP N-GRAMS PER CATEGORY
# =============================================================================
print("\nGenerating plots for top n-grams per category for the final model...")

def plot_top_ngrams(classifier, vectorizer, categories, top_n=10):
    feature_names = np.array(vectorizer.get_feature_names_out())
    category_map = {original: pretty for original, pretty in zip(newsgroups_data.target_names, pretty_target_names)}
    n_categories = len(categories)
    n_cols = 2
    n_rows = (n_categories + n_cols - 1) // n_cols
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 5 * n_rows))
    axes = axes.flatten()
    colors = plt.cm.get_cmap('plasma', top_n + 3)

    for i, category in enumerate(categories):
        ax = axes[i]
        category_idx = list(newsgroups_data.target_names).index(category)
        top_indices = np.argsort(classifier.feature_log_prob_[category_idx])[-top_n:]
        top_features = feature_names[top_indices]
        top_scores = np.exp(classifier.feature_log_prob_[category_idx][top_indices])
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_scores, align='center', color=colors(np.linspace(0.2, 0.9, top_n)))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel('Predictive Score (Probability)', fontsize=12)
        pretty_title = category_map.get(category, category)
        ax.set_title(f'Top N-grams for: {pretty_title}', fontsize=16, pad=15, fontfamily='serif')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Most Predictive N-grams per Newsgroup Category', fontsize=24, y=1.03, fontfamily='serif')
    plt.tight_layout(pad=3.0)
    plt.show()

selected_categories = [
    'rec.sport.baseball', 'sci.space', 'comp.sys.mac.hardware',
    'talk.politics.guns', 'rec.autos', 'sci.med'
]
plot_top_ngrams(classifier, vectorizer, selected_categories)


# =============================================================================
# 10. PROGRESSIVE EVALUATION (NEW SECTION)
# =============================================================================
print("\n--- Starting Progressive Evaluation ---")
print("Evaluating model performance with increasing training data size...")

# Define the subsets of training data to use
training_fractions = [0.25, 0.50, 0.75, 1.0]
results = []

# Get the total number of training samples
n_train_total = X_train_tfidf.shape[0]

for fraction in training_fractions:
    # Calculate the number of samples for this fraction
    n_samples = int(n_train_total * fraction)
    
    # Create a subset of the training data
    X_train_subset = X_train_tfidf[:n_samples]
    y_train_subset = y_train[:n_samples]
    
    print(f"\nTraining model on {fraction*100:.0f}% of the data ({n_samples} documents)...")
    
    # Initialize and train a new classifier on the subset
    subset_classifier = MultinomialNB(alpha=0.1)
    subset_classifier.fit(X_train_subset, y_train_subset)
    
    # Evaluate on the SAME full test set
    y_pred_subset = subset_classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred_subset)
    
    # Store the results
    results.append({
        'Percentage': f"{fraction*100:.0f}%",
        'Documents': n_samples,
        'Accuracy': f"{accuracy:.1%}"
    })
    print(f"Test Set Accuracy: {accuracy:.1%}")

# Print a formatted table of the final results for the report
print("\n--- Progressive Evaluation Results Table ---")
print("This table can be copied into your project report.\n")
print("| Percentage of Training Data Used | Number of Training Documents | Test Set Accuracy |")
print("| :--- | :--- | :--- |")
for result in results:
    print(f"| {result['Percentage']:<30} | {result['Documents']:<30} | {result['Accuracy']:<17} |")