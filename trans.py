import torch
import pandas as pd
from preprocessing import Tokenizer
from model import build_transformer
from config import get_config, latest_weights_file_path
from train import get_model
from bleu_Score import calculate_bleu

import pandas as pd



def translate_all_sentences(config, model, tokenizer_src, tokenizer_tgt, device):
    # Load the CSV file
    filepath = 'mankibaat.csv'
    df = pd.read_csv(filepath)
    translations = []
    # Loop through all source sentences in the CSV
    avg=0
    for index, row in df.iterrows():
        src_text = row[config['lang_src']]  # Assuming 'src_lang' is 'Hindi' in config
        print(f"Translating sentence {index+1}: {src_text}")
        
        translation = translate_single_sentence(model, src_text, tokenizer_src, tokenizer_tgt, config, device)
        bleu=calculate_bleu(src_text,translation)
        # print(f'bleu_Score {bleu}')
        translations.append({'source': src_text, 'translation': translation})
        print(f"Translation: {translation}\n")
        avg+=bleu
    
    
    output_df = pd.DataFrame(translations)
    output_df['bleu']=avg
    output_df.to_csv(config['translated_csv_file'], index=False)
    print("Translations saved to translated_sentences.csv")
    
    
    


def translate_single_sentence(model, src_sentence, tokenizer_src, tokenizer_tgt, config, device):
    # Tokenize the source sentence
    source = tokenizer_src._encode_single(src_sentence)
    max_pos_encoding_len = 30  
    if len(source) > max_pos_encoding_len - 2:  # Accounting for <sos> and <eos> tokens
        source = source[:max_pos_encoding_len - 2]
    source = torch.cat([
        torch.tensor([tokenizer_src.get_special_token_id('<sos>')], dtype=torch.int64),
        torch.tensor(source, dtype=torch.int64),
        torch.tensor([tokenizer_src.get_special_token_id('<eos>')], dtype=torch.int64)
    ]).unsqueeze(0).to(device)
    
    # Generate the source mask
    source_mask = (source != tokenizer_src.get_special_token_id('<pad>')).unsqueeze(1).unsqueeze(1).to(device)
    
    # Get the model's encoder output
    encoder_output = model.encode(source, source_mask)
    
    # Prepare decoder input with the SOS token
    decoder_input = torch.tensor([[tokenizer_tgt.get_special_token_id('<sos>')]], dtype=torch.int64).to(device)
    
    # Translation loop
    for _ in range(config['seq_len']):
        decoder_mask = torch.triu(torch.ones((decoder_input.size(1), decoder_input.size(1))), diagonal=1).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)
        
        if next_word == tokenizer_tgt.get_special_token_id('<eos>'):
            break
    
    # Decode and return the predicted target sentence
    # Decode and return the predicted target sentence
    decoded_sentence = tokenizer_tgt.decode(decoder_input[0].tolist())
    
    # Remove <sos> and <eos> tokens
    tokens = decoded_sentence.split()  # Split by whitespace or any other delimiter used
    if tokens and tokens[0] == '<sos>':
        tokens.pop(0)  # Remove <sos> if it's at the start
    if tokens and tokens[-1] == '<eos>':
        tokens.pop()  # Remove <eos> if it's at the end
    
    return ' '.join(tokens)
    # return tokenizer_tgt.decode(decoder_input[0].tolist())

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()

    tokenizer_src = Tokenizer(config["vocab_hi_path"], 'Hindi')
    tokenizer_tgt = Tokenizer(config["vocab_bhi_path"], 'Bhili')

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    
    # Translate all sentences in the CSV
    translate_all_sentences(config, model, tokenizer_src, tokenizer_tgt, device)
