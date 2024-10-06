"""
Prepare the enwik8 dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import PyPDF2
import numpy as np

# Function to extract text from PDF files
def extract_text_from_pdfs(pdf_dir):
    text_data = ""
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            with open(os.path.join(pdf_dir, filename), 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text_data += page.extract_text()
    return text_data

# Directory containing PDF files
pdf_dir = os.path.join(os.path.dirname(__file__), 'pdf_reports')
data = extract_text_from_pdfs(pdf_dir)
print(f"length of dataset in characters: {len(data):,}")

# Placeholder for ESG-related content extraction logic
def extract_esg_content(text):
    # Implement logic to extract ESG-related content from text
    # This could involve keyword matching, NLP techniques, etc.
    return text

# Extract ESG-related content
esg_data = extract_esg_content(data)

# get all the unique characters that occur in this text
chars = sorted(list(set(esg_data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train, validation, and test splits
n = len(esg_data)
num_test_chars = 5000000
train_data = esg_data[: -2 * num_test_chars]
val_data = esg_data[-2 * num_test_chars: -num_test_chars]
test_data = esg_data[-num_test_chars:]

# encode all splits to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
