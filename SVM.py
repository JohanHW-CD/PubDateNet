import random
import warnings
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from scipy.sparse import vstack
import numpy as np

# -----------------------
# CONFIGURATION TOGGLES
# -----------------------
USE_TFIDF = False
USE_NGRAMS = True
NGRAM_RANGE = (1, 3)
MAX_FEATURES = 10000
SVM_VERBOSE = 0
NUM_LOO_TESTS = 40
MDF = 3

# -----------------------
# HELPERS
# -----------------------
def round_to_nearest_decade(year):
    return int(round(year / 10.0) * 10)

# -----------------------
# PATHS
# -----------------------
base_dir = Path(r"C:\Users\johan\python_projects\pubdatenet")
books_dir = base_dir / "Books"

# -----------------------
# COLLECT & READ ALL BOOKS
# -----------------------
all_books = []
all_texts = []
all_labels = []

for decade_dir in sorted(books_dir.iterdir()):
    if not decade_dir.is_dir():
        continue
    for book_file in decade_dir.glob("*.txt"):
        try:
            year = int(book_file.stem)
            with open(book_file, encoding="utf-8") as f:
                text = f.read()
            all_books.append((book_file, year))
            all_texts.append(text)
            all_labels.append(round_to_nearest_decade(year))
        except ValueError:
            print(f"Skipping file: {book_file.name}")

print(f"Total books available: {len(all_books)}")

# -----------------------
# Fit vectorizer once on all texts
# -----------------------
vectorizer_cls = TfidfVectorizer if USE_TFIDF else CountVectorizer
vectorizer = vectorizer_cls(
    max_features=MAX_FEATURES,
    stop_words='english',
    ngram_range=NGRAM_RANGE if USE_NGRAMS else (1, 1),
    min_df=MDF,
    strip_accents='unicode'
)
X_all = vectorizer.fit_transform(all_texts)

# -----------------------
# Encode all labels once
# -----------------------
le = LabelEncoder()
y_all = le.fit_transform(all_labels)
np_labels = np.array(all_labels)  # for numeric comparison
unique_decades = sorted(set(all_labels))
num_classes = len(unique_decades)

# -----------------------
# Randomly pick test indices
# -----------------------
random.seed(42)
test_indices = random.sample(range(len(all_books)), NUM_LOO_TESTS)
accuracies = []
near_miss_count = 0

# -----------------------
# Leave-One-Out rounds
# -----------------------
for i, test_idx in enumerate(test_indices):
    test_path, test_year = all_books[test_idx]
    actual_decade = round_to_nearest_decade(test_year)

    print(f"\n=== LEAVE-ONE-OUT {i + 1}/{NUM_LOO_TESTS} ===")

    # Split into train/test
    train_indices = [j for j in range(len(all_books)) if j != test_idx]
    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_test = X_all[test_idx]

    # Train classifier
    clf = LinearSVC(max_iter=10000, verbose=SVM_VERBOSE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X_train, y_train)

    # Predict
    pred_decade = le.inverse_transform(clf.predict(X_test))[0]
    print(f"{test_path.name}: predicted {pred_decade}, actual {actual_decade}")

    correct = pred_decade == actual_decade
    if not correct and abs(pred_decade - actual_decade) == 10:
        print("  ↪ Missed by one decade.")
        near_miss_count += 1

    accuracies.append(correct)

# -----------------------
# SUMMARY
# -----------------------
correct = sum(accuracies)
within_one = correct + near_miss_count

print("\n=== SUMMARY ===")
print(f"Correct predictions: {correct}/{NUM_LOO_TESTS}")
print(f"One-decade near predictions (correct or ±10y): {within_one}/{NUM_LOO_TESTS}")
print(f"Exact accuracy: {correct / NUM_LOO_TESTS:.2f}")
print(f"Within-one-decade accuracy: {within_one / NUM_LOO_TESTS:.2f}")

# -----------------------
# BASELINE RANDOM GUESSING (expected values)
# -----------------------
expected_correct_random = NUM_LOO_TESTS / num_classes
expected_within_one_random = expected_correct_random * 3  # correct + one decade below + one above

print("\n=== BASELINE (Uniform Random Guessing) ===")
print(f"Expected correct predictions: {expected_correct_random:.2f}/{NUM_LOO_TESTS}")
print(f"Expected within-one-decade predictions: {expected_within_one_random:.2f}/{NUM_LOO_TESTS}")
