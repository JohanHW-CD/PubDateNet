import os
import random
import numpy as np
from collections import defaultdict, Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
import os, glob
import re, pyarrow.parquet as pq, pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

def load_texts(base_dir):
    """
    Load text files and their corresponding decade labels from a directory
    structure.

    Returns:
        texts (list of str): List of all book texts loaded from files.
        labels (list of str): Corresponding decade labels for each text.
    """
    texts = []
    labels = []
    book_ids = []
    decades = sorted(os.listdir(base_dir))  # e.g. ['1810', '1820', ...]
    for decade in decades:
        decade_path = os.path.join(base_dir, decade)
        if not os.path.isdir(decade_path):
            continue
        for fname in os.listdir(decade_path):
            if fname.endswith('.txt'):
                with open(os.path.join(decade_path, fname), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(decade)
                    book_ids.append(f"{decade}_{fname}")
    return texts, labels, book_ids

def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", "", text)

def load_nyt_texts_streaming(parquet_file: str,
                             max_per_decade: int | None = 20_000,
                             min_year: int = 1920,
                             max_year: int = 2020,
                             clean_fn = remove_numbers):
    """
    Memory-light loader for the NYT Parquet dump.

    Parameters
    ----------
    parquet_file : str
        Path to the single *.parquet* file from Kaggle.
    max_per_decade : int | None
        Keep at most this many articles per decade (≈ class balancing).
        `None` ⇒ keep everything available.
    min_year, max_year : int
        Year range filter.
    clean_fn : callable
        Function applied to the excerpt (and title) for light pre-processing.

    Returns
    -------
    texts, labels, article_ids  (all plain Python lists)
    """
    pf = pq.ParquetFile(parquet_file)

    kept = defaultdict(int)
    texts, labels, ids = [], [], []

    # iterate row-group by row-group
    for rg in range(pf.num_row_groups):
        # pick only the needed columns to keep it tiny
        table = pf.read_row_group(rg, columns=["year", "title", "excerpt"])
        df = table.to_pandas()

        # filter rows by year in-place (fast, vectorised)
        mask = df["year"].between(min_year, max_year)
        df = df[mask]

        for idx, (year, title, excerpt) in df[["year", "title", "excerpt"]].iterrows():
            decade = str((year // 10) * 10)

            # honor per-decade cap, if requested
            if max_per_decade is not None and kept[decade] >= max_per_decade:
                continue

            kept[decade] += 1
            text = f"{title} {excerpt}"
            text = clean_fn(text)

            texts.append(text)
            labels.append(decade)
            ids.append(str(idx))   # row index is already unique

    return texts, labels, ids


def split_text_into_chunks(text, max_len, step):
    """
    Split text into overlapping chunks of tokens.

    - First the whole book is split into individual tokens.
    - Each chunk is max_len tokens long.
    - The window steps forward by step tokens.
    - This gives overlapping chunks.
    """
    tokens = text.split() 
    chunks = []
    for start in range(0, max(1, len(tokens) - max_len + 1), step):
        chunk = ' '.join(tokens[start:start + max_len])
        chunks.append(chunk)
    return chunks

def preprocess_and_chunk(texts, labels, book_ids, max_vocab, max_len, step):
    all_chunks = []
    all_labels = []
    all_groups = []

    # Each book is split into multiple chunks.
    # Each chunk is assigned the same label (decade) and book_id.
    # Now you have a long list of short text chunks and their corresponding labels and book identifiers.
    for text, label, book_id in zip(texts, labels, book_ids):
        chunks = split_text_into_chunks(text, max_len=max_len, step=step)
        all_chunks.extend(chunks)
        all_labels.extend([label] * len(chunks))
        all_groups.extend([book_id] * len(chunks))
    
    # Converts each chunk into a list of integers.
    # Words not in the max_vocab are treated as out-of-vocabulary.
    tokenizer = Tokenizer(num_words=max_vocab, oov_token='<OOV>')
    tokenizer.fit_on_texts(all_chunks)
    sequences = tokenizer.texts_to_sequences(all_chunks)

    # Ensures all sequences have the same max_len by padding or truncating.
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    le = LabelEncoder()
    y = le.fit_transform(all_labels)
    
    return padded, y, all_groups, tokenizer, le

def build_lstm(max_vocab, num_classes, embedding_dim, lstm_layers, lstm_units, dropout, learning_rate):
    """
    Build a configurable LSTM model for multi-class text classification.

    Args:
        max_vocab (int): Size of the vocabulary (input dimension to Embedding layer).
        max_len (int): Length of input sequences.
        num_classes (int): Number of output classes (decades).
        embedding_dim (int): Dimensionality of word embeddings.
        lstm_layers (int): Number of stacked LSTM layers (>=1).
        lstm_units (int): Number of units in each LSTM layer.
        dropout (float): Dropout rate applied after each LSTM layer except the last.
    """
    model = Sequential()
    model.add(Embedding(max_vocab, embedding_dim))
    for _ in range(lstm_layers - 1):
        model.add(LSTM(lstm_units, return_sequences=True))
        model.add(Dropout(dropout))
    # Last LSTM layer without return_sequences
    model.add(LSTM(lstm_units, kernel_regularizer=l2(1e-4)))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def train(model, X_train, y_train, X_val, y_val, epochs, batch_size, callbacks = None):
    """
    Train the LSTM model on the training data and validate on validation data.
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks or [] 
    )
    return history

def evaluate(model,
             X, y,
             label_encoder,
             tol,
             batch_size):
    """
    Report two metrics on (X, y):

    • strict accuracy  – prediction must match the decade exactly  
    • tolerant accuracy – prediction is counted correct when the
      predicted decade differs by ≤ tol *decades* from the target

    Returns
    -------
    loss, strict_acc, tolerant_acc
    """
    import numpy as np

    # ── Keras built-in metric (strict) ─────────────────────────────
    loss, strict_acc = model.evaluate(X, y,
                                      verbose=0,
                                      batch_size=batch_size)

    # ── tolerant metric ────────────────────────────────────────────
    # 1. raw probabilities → integer class ids
    y_pred_int = np.argmax(model.predict(X, batch_size=batch_size,
                                         verbose=0),
                           axis=1)

    # 2. map integer ids → decade numbers (e.g. 1820, 1830, …)
    decoded_pred = label_encoder.inverse_transform(y_pred_int).astype(int)
    decoded_true = label_encoder.inverse_transform(y).astype(int)

    # 3. decades are multiples of 10, so “±1 decade” ⇒ ≤ 10 years
    tolerant = np.abs(decoded_pred - decoded_true) <= (tol * 10)
    tolerant_acc = tolerant.mean()

    print(f"Loss: {loss:.4f}  |  "
          f"Exact acc: {strict_acc:.4f}  |  "
          f"±{tol}-decade acc: {tolerant_acc:.4f}")

    return loss, strict_acc, tolerant_acc

def stratified_book_split(
        book_ids: list[str],
        labels: list[str],
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42):
    """
    Split books into train/val/test so that

    • the split happens **per book**, never per text-chunk  
    • no book_id appears in more than one set  
    • the label (decade) distribution is as close as possible in all sets  

    Returns
    -------
    train_books, val_books, test_books : set[str]
        Three *disjoint* sets of book_ids.
    """
    rng = np.random.RandomState(seed)
    book_ids = np.asarray(book_ids)
    labels   = np.asarray(labels)

    # ───── first cut off the test set ────────────────────────────────────────────
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(
        X=labels, y=labels, groups=book_ids))

    # ───── then carve validation out of the remaining data ─────────────────────
    # note: val_ratio is w.r.t. the *original* data, so we rescale it
    val_frac_of_remaining = val_ratio / (1.0 - test_ratio)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_frac_of_remaining, random_state=seed)
    train_idx, val_idx = next(gss_val.split(
        X=labels[train_val_idx],
        y=labels[train_val_idx],
        groups=book_ids[train_val_idx]))

    train_books = set(book_ids[train_val_idx][train_idx])
    val_books   = set(book_ids[train_val_idx][val_idx])
    test_books  = set(book_ids[test_idx])

    # ───── sanity-check: no overlap ─────────────────────────────────────────────
    assert train_books.isdisjoint(val_books)
    assert train_books.isdisjoint(test_books)
    assert val_books.isdisjoint(test_books)

    return train_books, val_books, test_books

def chunk_with_tokenizer(texts, labels, book_ids,
                         tokenizer, label_encoder,
                         max_len, step):
    """Chunk texts and convert to padded integer sequences
       using *already-fitted* tokenizer & label_encoder."""
    all_chunks, all_labels, all_groups = [], [], []

    for text, lab, bid in zip(texts, labels, book_ids):
        for chunk in split_text_into_chunks(text, max_len=max_len, step=step):
            all_chunks.append(chunk)
            all_labels.append(lab)
            all_groups.append(bid)

    seqs = tokenizer.texts_to_sequences(all_chunks)
    X = pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')
    y = label_encoder.transform(all_labels)
    return X, y, all_groups
    
def train_sequences():
    base_dir = 'Books/'
    texts, labels, book_ids = load_texts(base_dir)
    MAX_LEN = 500
    MAX_VOCAB = 10000
    STEP    = MAX_LEN          # one-window-per-chunk as before
    BATCH_SIZE = 32

    # ── 1. book-level split ────────────────────────────────────────
    train_books, val_books, test_books = stratified_book_split(book_ids, labels)

    # helper to filter the raw lists
    def subset(bset):
        idx = [i for i, bid in enumerate(book_ids) if bid in bset]
        return [texts[i] for i in idx], [labels[i] for i in idx], [book_ids[i] for i in idx]

    train_texts, train_labs, train_ids = subset(train_books)
    val_texts,   val_labs,   val_ids   = subset(val_books)
    test_texts,  test_labs,  test_ids  = subset(test_books)

    # ── 2. fit tokenizer + label encoder ONLY on the training chunks ─
    X_tr_chunks, y_tr, groups_tr, tokenizer, label_enc = preprocess_and_chunk(
        train_texts, train_labs, train_ids,
        max_vocab=MAX_VOCAB, max_len=MAX_LEN, step=STEP)

    # ── 3. use that tokenizer/encoder for val & test ───────────────
    X_val, y_val, groups_val = chunk_with_tokenizer(
        val_texts, val_labs, val_ids, tokenizer, label_enc,
        max_len=MAX_LEN, step=STEP)

    X_test, y_test, groups_test = chunk_with_tokenizer(
        test_texts, test_labs, test_ids, tokenizer, label_enc,
        max_len=MAX_LEN, step=STEP)

    # ── 4. sanity check ────────────────────────────────────────────
    print("Train label distribution:", Counter(y_tr))
    print("Val   label distribution:", Counter(y_val))
    print("Test  label distribution:", Counter(y_test))

    # ── 5. build & train the model ─────────────────────────────────
    model = build_lstm(
        max_vocab=MAX_VOCAB,
        num_classes=len(label_enc.classes_),
        embedding_dim=128,
        lstm_layers=1,
        lstm_units=32,
        dropout=0.2,
        learning_rate=0.001
    )

    train(model, X_tr_chunks, y_tr, X_val, y_val,
          epochs=2, batch_size=BATCH_SIZE)

    evaluate(model, X_val, y_val, label_enc, 1, BATCH_SIZE) 
    evaluate(model, X_test, y_test, label_enc, 1, BATCH_SIZE)   # ← use y_test, not y_val

    # Loss: 2.5559  |  Exact acc: 0.1167  |  ±1-decade acc: 0.5087

def train_sequences_nyt():
    CSV_PATH = "nyt-articles-data/nyt_data.parquet"   # adjust if your file is named differently

    # --- 0. load ----------------------------------------------------------------
    texts, labels, article_ids = load_nyt_texts_streaming(
            parquet_file=CSV_PATH,
            max_per_decade=30000)

    # ── 1. stratified 80 / 10 / 10 split (keep IDs aligned) ────────────
    (train_texts, temp_texts,
     train_labs,  temp_labs,
     train_ids,   temp_ids) = train_test_split(
        texts, labels, article_ids,
        test_size=0.20,
        stratify=labels,
        random_state=42)

    (val_texts, test_texts,
     val_labs,  test_labs,
     val_ids,   test_ids) = train_test_split(
        temp_texts, temp_labs, temp_ids,
        test_size=0.50,          # half of 20 %  ⇒ 10 %
        stratify=temp_labs,
        random_state=42)

    # ── 2. tokenise & pad ───────────────────────────────────────────────
    MAX_LEN   = 128
    MAX_VOCAB = 50_000
    STEP      = MAX_LEN          # one chunk per article
    BATCH_SIZE = 256

    X_train, y_train, _, tokenizer, label_enc = preprocess_and_chunk(
        train_texts, train_labs, train_ids,
        max_vocab=MAX_VOCAB, max_len=MAX_LEN, step=STEP)

    X_val, y_val, _ = chunk_with_tokenizer(
        val_texts, val_labs, val_ids,
        tokenizer, label_enc,
        max_len=MAX_LEN, step=STEP)

    X_test, y_test, _ = chunk_with_tokenizer(
        test_texts, test_labs, test_ids,
        tokenizer, label_enc,
        max_len=MAX_LEN, step=STEP)

    # ── 3. build the model ──────────────────────────────────────────────
    model = build_lstm(
        max_vocab     = MAX_VOCAB,
        num_classes   = len(label_enc.classes_),
        embedding_dim = 128,
        lstm_layers   = 1,
        lstm_units    = 64,
        dropout       = 0.0,
        learning_rate = 1e-3
    )
    early   = EarlyStopping("val_loss", patience=3, restore_best_weights=True)

    # ── 4. train ────────────────────────────────────────────────────────
    train(model, X_train, y_train, X_val, y_val,
          epochs=25, batch_size=BATCH_SIZE, callbacks=early)

    # ── 5. evaluate ─────────────────────────────────────────────────────
    print("\nValidation:")
    evaluate(model, X_val,  y_val,  label_enc, tol=1, batch_size=BATCH_SIZE)
    print("\nTest:")
    evaluate(model, X_test, y_test, label_enc, tol=1, batch_size=BATCH_SIZE)
    '''
    Epoch 12/25
1037/1037 ━━━━━━━━━━━━━━━━━━━━ 225s 217ms/step - accuracy: 0.9337 - loss: 0.2167 - val_accuracy: 0.8115 - val_loss: 0.7597

    Validation:
    Loss: 0.7143  |  Exact acc: 0.8010  |  ±1-decade acc: 0.9122

    Test:
    Loss: 0.7190  |  Exact acc: 0.8000  |  ±1-decade acc: 0.9108
    '''

    
if __name__ == '__main__':
    '''
    Number of articles : 17370913
    Mean tokens        : 37.922644825864936
    Median             : 27
    90th percentile    : 77
    95th percentile    : 104
    99th percentile    : 176
    '''
    '''
    PARQUET = "nyt-articles-data/nyt_data.parquet"   # adjust the path
    COLS    = ["title", "excerpt"]                   # change if you add more fields

    # ── read only the columns you need ──────────────────────────────────
    table = pq.read_table(PARQUET, columns=COLS)
    df    = table.to_pandas()

    # ── build the exact text you will later feed the model ──────────────
    df["text"] = df["title"].fillna("") + " " + df["excerpt"].fillna("")

    # ── tokenise with plain .split() just for a rough length estimate ───
    lengths = df["text"].str.split().str.len()

    print("Number of articles :", len(lengths))
    print("Mean tokens        :", lengths.mean())
    print("Median             :", int(lengths.median()))
    for p in (90, 95, 99):
        print(f"{p:2d}th percentile    :", int(np.percentile(lengths, p)))
    '''
    train_sequences_nyt()
