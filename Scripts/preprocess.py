import os
import re

BOOKS_DIR = r"C:"  # Raw string for Windows path

# Define regex patterns
month_pattern = re.compile(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b', re.IGNORECASE)
year_pattern = re.compile(r'\b\d{4}\b')

# Walk through all subdirectories and files
for root, dirs, files in os.walk(BOOKS_DIR):
    print(files)
    for filename in files:
        if filename.endswith(".txt"):
            file_path = os.path.join(root, filename)

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # Find start and end markers
            start_idx = None
            end_idx = None
            for i, line in enumerate(lines):
                if '*** START OF THE PROJECT GUTENBERG' in line:
                    start_idx = i + 1
                if '*** END OF THE PROJECT GUTENBERG' in line and end_idx is None:
                    end_idx = i

            if start_idx is None:
                print(f"Start marker not found in {filename}, skipping.")
                continue

            if end_idx is None:
                end_idx = len(lines)

            content = ''.join(lines[start_idx:end_idx])
            content = month_pattern.sub('', content)
            content = year_pattern.sub('', content)
            content = content.lower()

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"Processed {file_path}")
