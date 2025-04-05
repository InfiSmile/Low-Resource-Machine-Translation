from sklearn.model_selection import train_test_split
import torchmetrics
from pathlib import Path
import pandas as pd
import wandb 
from bleu_Score import calculate_bleu
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
from dataset import causal_mask,BilingualDataset,build_vocab,collate_fn
from preprocessing import Tokenizer
from typing import Any
from model import build_transformer
from config import *
import wandb

import warnings
from tqdm import tqdm
import os

from torch.utils.tensorboard import SummaryWriter

# Check if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU instead.")

# Now, you can move your model and tensors to the selected device
# model = model.to(device)
# tensor = tensor.to(device)


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, batch):
    # loss_fn = nn.CrossEntropyLoss(ignore_index= tokenizer_src.get_special_token_id('<pad>')).to(device)
    sos_idx = tokenizer_tgt.get_special_token_id('<sos>')
    eos_idx = tokenizer_tgt.get_special_token_id('<eos>')

    label = batch['label'].to(device)  # Shape: (1, seq_len)
    batch_size = label.size(0)

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)  # Shape: (1, 1)
    predicted_tokens = []
    # total_loss = 0  # Initialize total loss
    for i in range(max_len):
        if decoder_input.size(1) > max_len:
            break

        # Build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        out_proj = model.project(out)  # Shape: (1, seq_len, vocab_size)
        vocab_size = tokenizer_tgt.get_vocab_size()
        if torch.isnan(label).any() or torch.isinf(label).any() or (label < 0).any() or (label >= vocab_size  ).any():
              print("Invalid label value detected.")
        if torch.isnan(out_proj).any():
            print(f'invalid out proj {out_proj}')
        # Compute loss word by word
         # Ensure predicted sequence length matches target length
        # predicted_len = decoder_input.size(1)
        # target_len = label.size(1)
        prob = out_proj[:, -1, :]  # Shape: (1, vocab_size)
        _, next_word = torch.max(prob, dim=1)
        
        # Teacher forcing: Use the ground truth with a certain probability
        # use_teacher_forcing = random.random() < teacher_forcing_ratio

        # if use_teacher_forcing and i < label.size(1):
        #     next_word = label[:, i]  # Use the ground truth word
        

        
        # if i < label.size(1):  # Ensure we don't go out of bounds
            
                
        #     loss = loss_fn(out_proj[:, -1, :], label[:, i])
            
        #     print(f'loss : {loss}')
        #     total_loss += loss.item()

          


        # Get next token
        # prob = out_proj[:, -1, :]  # Shape: (1, vocab_size)
        # _, next_word = torch.max(prob, dim=1)

        # Prepare the input for the next step
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim=1
        )

        if next_word.item() == eos_idx:
            break

        if next_word.item() != sos_idx and next_word.item() != eos_idx:
                predicted_tokens.append(next_word.item())


        
    predicted_len = len(predicted_tokens)
    target_len = (label != tokenizer_src.get_special_token_id('<pad>')).sum().item() 
    if predicted_len > target_len:
        print(f"Warning: Predicted sequence is longer than target sequence.")
        return decoder_input.squeeze(0)

    return decoder_input.squeeze(0)  # Return decoded sequence and average loss



def get_ds(config):
  df_raw=pd.read_csv('speech1.csv')
  

# Assuming df is your DataFrame
  def convert_columns_to_string(df):
    df = df.applymap(str)
    return df
  df_raw= convert_columns_to_string(df_raw)
  duplicates_col1 = df_raw.duplicated('Hindi', keep=False)
  duplicates_col2 = df_raw.duplicated('Bhili', keep=False)
  duplicates_any_col = duplicates_col1 | duplicates_col2
  df_raw = df_raw[~duplicates_any_col]
  
  train_df_raw, val_df_raw = train_test_split(df_raw, test_size=0.2, random_state=42)
  train_df_raw.reset_index(drop=True, inplace=True)
  val_df_raw.reset_index(drop=True, inplace=True)

  # Define paths for saving/loading tokenized data

  tokenized_train_path = Path(config['tokenizer_file'].format('train'))
  tokenized_val_path = Path(config['tokenizer_file'].format('val'))
    

  # Build tokenizers
  tokenizer_src= Tokenizer(config["vocab_hi_path"], 'Hindi')
  tokenizer_tgt= Tokenizer(config["vocab_bhi_path"], 'Bhili')
 


  # Tokenize or load tokenized data
  if not tokenized_train_path.exists():
      print(f"Tokenizing training data...")
      train_src_ids = tokenizer_src.encode(train_df_raw['Hindi'])
      train_tgt_ids = tokenizer_tgt.encode(train_df_raw['Bhili'])
      tokenized_train = {
          'src': train_src_ids,
          'tgt': train_tgt_ids
      }
      tokenizer_src.save_tokenized_data(tokenized_train, tokenized_train_path)
  else:
      print(f"Loading tokenized training data from {tokenized_train_path}...")
      tokenized_train = Tokenizer.load_tokenized_data(tokenized_train_path)

  if not tokenized_val_path.exists():
      print(f"Tokenizing validation data...")
      val_src_ids = tokenizer_src.encode(val_df_raw['Hindi'])
      val_tgt_ids = tokenizer_tgt.encode(val_df_raw['Bhili'])
      tokenized_val = {
          'src': val_src_ids,
          'tgt': val_tgt_ids
      }
      tokenizer_src.save_tokenized_data(tokenized_val, tokenized_val_path)
  else:
      print(f"Loading tokenized validation data from {tokenized_val_path}...")
      tokenized_val = Tokenizer.load_tokenized_data(tokenized_val_path)

  train_ds= BilingualDataset(train_df_raw, tokenizer_src, tokenizer_tgt, 'Hindi', 'Bhili', config['seq_len'])
  val_ds= BilingualDataset(val_df_raw,tokenizer_src,tokenizer_tgt,'Hindi','Bhili',config['seq_len'])

  max_len_src=0
  max_len_tgt=0

  src_ids=tokenizer_src.encode(df_raw['Hindi'])
  tgt_ids=tokenizer_tgt.encode(df_raw['Bhili'])
  
  max_len_src= max(max_len_src, len(src_ids))
  max_len_tgt= max(max_len_tgt, len(tgt_ids))

  print(f'Max length of source sentence: {max_len_src}')
  print(f'Max length of target sentence: {max_len_tgt}')

  train_dataloader= DataLoader(train_ds, batch_size=config['batch_size'],shuffle= True,collate_fn=collate_fn)

  val_dataloader= DataLoader(val_ds,batch_size=1, shuffle= True,collate_fn=collate_fn)

  return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config,vocab_src_len,vocab_tgt_len):
    model= build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],config['d_model'])
    return model

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=1):
    model.eval()
    count = 0
    global_step=0
    source_texts = []
    expected = []
    predicted = []
    sos_idx = tokenizer_tgt.get_special_token_id('<sos>')
    eos_idx = tokenizer_tgt.get_special_token_id('<eos>')


    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80
    # batch_iterator= tqdm(validation_ds,desc=f'Preprocessing epoch {epoch:02d}')
    with torch.no_grad():
        for batch in validation_ds:

            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)


            print(f' in validation {encoder_input.shape}')
            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device,batch)

            source_text = batch['src_txt'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            label=batch['label'].to(device) #seq_len


            
            # target_len = label.size(1)  # Length of the target sequence
            # predicted_len = model_out.size(0)  # Length of the predicted sequence

            # print(f"Target length: {target_len}")
            # print(f"Predicted length: {predicted_len}")

            # if predicted_len > target_len:
            #     print(f"Warning: Predicted sequence is longer than target sequence.")
            #     predicted = predicted[:target_len]
            # if torch.isnan(losss):
            #     print(f"NaN loss detected")


            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg('-'*console_width)
            # print_msg(f"Validation Loss: {losss:.4f}")
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
          # Evaluate the character error rate
    # Compute the char error rate 
    # Compute the Character Error Rate (CER)
    cer_metric = torchmetrics.text.CharErrorRate()
    cer = cer_metric(predicted, expected)
    print(f"Character Error Rate (CER): {cer}")

    # Compute the Word Error Rate (WER)
    wer_metric = torchmetrics.text.WordErrorRate()
    wer = wer_metric(predicted, expected)
    print(f"Word Error Rate (WER): {wer}")

    # Compute the BLEU Score
    
    # print(f"predicted : {predicted}")
    # print(f"expected {expected}")
    bleu=calculate_bleu(expected, predicted)
    
    print(f"BLEU Score: {bleu}")
    return bleu

    # chrf = calculate_chrf2(expected, predicted)
    # print(f"chrf_nltk {chrf}")

def train_model(config):
  device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Using device {device}')
  # Make sure the weights folder exists
  
  Path(config['model_folder']).mkdir(parents= True, exist_ok= True)

  train_dataloader,val_dataloader,tokenizer_src, tokenizer_tgt= get_ds(config)
  model= get_model(config, tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
  #Tensorboard
  writer= SummaryWriter(config['experiment_name'])
  print(tokenizer_src.get_vocab_size())
  optimizer= torch.optim.Adam(model.parameters(), lr= config['lr'], eps=1e-9)

  initial_epoch=0
  global_step=0
  preload = config['preload']
  model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None

  if config['preload']:
    model_filename= get_weights_file_path(config,config['preload'])
    print(f'Preloading model {model_filename}')
    state= torch.load(model_filename)
    initial_epoch= state['epoch']+1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step= state['global_step']

  loss_fn= nn.CrossEntropyLoss(ignore_index= tokenizer_src.get_special_token_id('<pad>'), label_smoothing=.1).to(device)# for every highest probability token it will take 0.1% of score and give it to the others.


  for epoch in range(initial_epoch, config['num_epochs']):
    model.train()
    print('hey')
    batch_iterator= tqdm(train_dataloader,desc=f'Preprocessing epoch {epoch:02d}')
    # print('hello',type(batch_iterator))

    for batch in batch_iterator:

      # print("type of batch",type(batch))
      # print("keys ",batch.keys())
      encoder_input= batch['encoder_input'].to(device) #(B,Seq_len)
      decoder_input= batch['decoder_input'].to(device) #(B,Seq_len)
      encoder_mask=batch['encoder_mask'].to(device) # (B,1,1,seq_len)
      decoder_mask= batch['decoder_mask'].to(device) #(B,1,Seq_len,Seq_len)

      # Run the tensors throgh the transformer
      encoder_output= model.encode(encoder_input,encoder_mask)
    #   print(f'encoder_output:::::::::::::::::::::::::::::::: {encoder_output.shape}')
      decoder_output=model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)#(B,Seq_len,d_model)
    #   print(f'decoder output**********************: {decoder_output.shape}')
      proj_output= model.project(decoder_output) #(B,seq_len,tgt_vocab_size)
    #   print(f'proj_output>>>>>>>>>>>>>{proj_output.shape}')
      # print('*******************************************************************************************************')
      # print(f'shape of proj_output: {proj_output.shape}')
      label= batch['label'].to(device) #(B,seq_len)

      #(B,seq_len,tgt_vocab_size)--> (B*seq_len,tgt_vocab_size)
      loss= loss_fn(proj_output.reshape(-1,tokenizer_tgt.get_vocab_size()),label.reshape(-1))
      batch_iterator.set_postfix({f'loss': f"{loss.item():6.3f}"})

      #log the loss
      writer.add_scalar('train loss', loss.item(), global_step)
      writer.flush()

      # Backpropagate the loss
      loss.backward()

    #   torch.nn.utils.clip_grad_norm_(model.parameters(), )  # Gradient clipping

      #Update the weights
      optimizer.step()
      optimizer.zero_grad()

      global_step += 1
    run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

    model_filename= get_weights_file_path(config,f'{epoch:02d}')
    torch.save({

        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,

    },model_filename)



