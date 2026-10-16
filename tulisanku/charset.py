"""The characters TULISANKU learns to generate.

We use the 62 alphanumeric characters because that is exactly the set in
NIST Special Database 19 (our real-handwriting dataset), so fonts and
handwriting line up one-to-one. Punctuation can be added later.
"""
import string

DIGITS = string.digits
UPPER = string.ascii_uppercase
LOWER = string.ascii_lowercase

CHARSET = DIGITS + UPPER + LOWER  # "0..9A..Za..z", 62 characters
CHAR_TO_INDEX = {c: i for i, c in enumerate(CHARSET)}
