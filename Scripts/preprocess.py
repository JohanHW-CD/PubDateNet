"""
1.  For each .txt file in C:\Users and subfolders:
2.   Read file content.
3.   Remove month names and 4-digit years.
4.   Remove Gutenberg START/END markers if present.
5.   Convert text to lowercase.
6.   Detect language from first 5000 chars (edit in #Detect Language)
7.   If not English, delete file (os.remove).
8.   If English, overwrite file with cleaned text.

ALWAYS fix BOOKS_DIR (because I always fuck it up)
"""

import os
import re
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # Make detection deterministic

BOOKS_DIR = r"C:\Users\johan\Downloads\PubDate\PubDateNet-main\Books"

month_pattern = re.compile(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b', re.IGNORECASE)
year_pattern = re.compile(r'\b\d{4}\b')

for root, dirs, files in os.walk(BOOKS_DIR):
    for filename in files:
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(root, filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Locate Gutenberg markers
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if '*** START OF THE PROJECT GUTENBERG' in line:
                start_idx = i + 1
            if '*** END OF THE PROJECT GUTENBERG' in line and end_idx is None:
                end_idx = i

        if start_idx is not None and end_idx is not None:
            content = ''.join(lines[start_idx:end_idx])
        elif start_idx is not None:
            content = ''.join(lines[start_idx:])
        elif end_idx is not None:
            content = ''.join(lines[:end_idx])
        else:
            content = ''.join(lines)

        # Clean content
        content_clean = month_pattern.sub('', content)
        content_clean = year_pattern.sub('', content_clean)
        content_clean = content_clean.lower()

        # Detect language
        sample = content_clean[:5000]
        try:
            lang = detect(sample)
        except:
            print(f"Language detection failed for {filename}, skipping.")
            continue

        if lang != 'en':
            print(f"Removing non-English file: {filename} (detected: {lang})")
            os.remove(file_path)
            continue

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content_clean)

        print(f"Processed {file_path}")
