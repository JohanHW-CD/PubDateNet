import os
import re

BOOKS_DIR = "../Books"

# Define regex patterns
month_pattern = re.compile(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b', re.IGNORECASE)
year_pattern = re.compile(r'\b\d{4}\b')

# Walk through all subdirectories and files
for root, dirs, files in os.walk(BOOKS_DIR):
    for filename in files:
        if filename.endswith(".txt"):
            file_path = os.path.join(root, filename)

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            start_idx = None
            for i, line in enumerate(lines):
                if '*** START OF THE PROJECT GUTENBERG' in line:
                    start_idx = i + 1  # Start from the line *after* the marker
                    break

            if start_idx is None:
                print(f"Marker not found in {filename}, skipping.")
                continue

            # Join the lines after the start marker
            content = ''.join(lines[start_idx:])
            # Remove month names and 4-digit years
            content = month_pattern.sub('', content)
            content = year_pattern.sub('', content)

            # Overwrite the original file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"Processed {file_path}")
