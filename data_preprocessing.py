from khmerspell import khnormal
import re

"""
NORMALIZE AND CLEAN TEXT
"""

def normalize_khmer(text):
    """
    Correct order of characters and unicode normalization
    """
    return khnormal(text)

def clean_text(text, is_khmer=False):
    """
    Remove special characters and normalize text
    """
    text = str(text).strip()
    if is_khmer:
        text = re.sub(r'[^\u1780-\u17FF]', '', text)
        text = normalize_khmer(text)
    else:
        text = re.sub(r'[^a-z]', '', text.lower())
    return text
