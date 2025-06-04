# DD2417 Project  
**Predicting Year of Publication Using Machine Learning Models**

This project explores the use of various machine learning models to predict the decade a text was published in, based on its content.

---

## Models

**IMPORTANT NOTE:** The BERT.ipynb cannot be viewed from Github directly, please download it and open it with any IDE or Google Colab.

| Model | File | Notes |
|-------|------|-------|
| **SVM** | `SVM.py` | Support Vector Machine implementation |
| **BERT** | `BERT.ipynb` | Run on Google Colab – make sure to upload the dataset to your Google Drive |
| **LSTM** | `LSTM.py` | Long Short-Term Memory model |

---

## Datasets

1. **Project Gutenberg**  
   Public domain books retrieved from [Project Gutenberg](https://www.gutenberg.org/).  
   > *Note: We manually selected 63 books with balanced representation across 14 decades.*

2. **New York Times Corpus**  
   Excerpts from [Kaggle: NYT Articles Data](https://www.kaggle.com/datasets/tumanovalexander/nyt-articles-data).  
   > *Due to its large size (~3GB), the dataset is not included here. Please download it from the link above.*

---

## Miscellaneous

- `preprocess.py`:  
   Used to preprocess the Project Gutenberg texts by:
   - Removing any date-related content
   - Stripping the standard Project Gutenberg prefix and suffix

---
