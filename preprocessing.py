import pandas as pd
from sklearn.model_selection import train_test_split
import json
from indicnlp.tokenize import indic_tokenize

hindi_vocab = {'<sos>': 0, '<eos>': 1, '<pad>': 2, '<unk>': 3}
bhili_vocab = {'<sos>': 0, '<eos>': 1, '<pad>': 2, '<unk>': 3}


def build_vocab(sentences, vocab):
        for word in indic_tokenize.trivial_tokenize(sentences):
            if word not in vocab:
                vocab[word] = len(vocab)
                
csv_file_path= r"speech1.csv"

df= pd.read_csv(csv_file_path)
# print(df['Hindi'])

import pandas as pd

# Assuming df is your DataFrame
def convert_columns_to_string(df):
    df = df.map(str)
    return df

df= convert_columns_to_string(df)

for sentence in df['Hindi']:
    text= sentence
    build_vocab(text, hindi_vocab)

for sentence in df['Bhili']:
    text= sentence
    # print(text)
#     print(text)
    build_vocab(text, bhili_vocab)

# print(repr(bhili_vocab))

# print("Hindi Vocabulary:", hindi_vocab)
# print("Bhili Vocabulary:", bhili_vocab)

# Define file paths to save vocabularies
import json
import os
vocab_hi_path = 'vocab/vocab_hindi.json'
vocab_bhi_path = 'vocab/vocab_bhili.json'

def create_vocab_file_if_not_exists(vocab_path):

    directory = os.path.dirname(vocab_path)
    
    # Debugging: Print the directory to ensure it's correct
    print(f"Directory to be created (if not exists): {directory}")
    
    # Ensure the directory exists
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory {directory} created or already exists.")
    except Exception as e:
        print(f"Failed to create directory {directory}. Error: {str(e)}")

    if not os.path.exists(vocab_path):
        # Create an empty vocabulary dictionary
        vocab_dict = {}
        # Save it to the specified path
        with open(vocab_path, 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)
        print(f"Created an empty vocabulary file at {vocab_path}")
    else:
        print(f"Vocabulary file already exists at {vocab_path}")

create_vocab_file_if_not_exists(vocab_hi_path)
create_vocab_file_if_not_exists(vocab_bhi_path)

# Save vocabularies as JSON files
with open(vocab_hi_path, 'w') as f_hi:
    json.dump(hindi_vocab, f_hi, indent=4)

with open(vocab_bhi_path, 'w') as f_bhi:
    json.dump(bhili_vocab, f_bhi, indent=4)

class Tokenizer:
    def __init__(self,vocab_path,lang):

        self.vocab=self.load_vocab(vocab_path)
        self.lang=lang


    def load_vocab(self,vocab_path):

        with open(vocab_path,'r') as file:
            vocab= json.load(file)
        return vocab
    
    def encode(self,text_series):
        encoded_series= text_series.apply(self._encode_single)
        return encoded_series.tolist()
    
    def _encode_single(self,text):
        # def tokenize_text(self, indic_string):
        tokens = indic_tokenize.trivial_tokenize(text)
        token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        # print(token_ids)
        return token_ids
    
    def get_special_token_id(self,token_name):

        return self.vocab.get(token_name,None)
    
    def get_vocab_size(self):

        return len(self.vocab)
    
    def save_tokenized_data(self,tokenized_data,file_path):
        with open(file_path,'w') as f:
            json.dump(tokenized_data,f)
    
    @staticmethod
    def load_tokenized_data(file_path):
        with open(file_path,'r') as f:
            return json.load(f)
        
    def decode(self,token_ids):

        inv_vocab={v: k for k,v in self.vocab.items()}
        tokens=[inv_vocab.get(token_id,'<unk>') for token_id in token_ids]
        # print(tokens)
        return ' '.join(tokens)
    






