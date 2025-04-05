from torch.utils.data import Dataset
from typing import Any
import torch
from preprocessing import *


class BilingualDataset(Dataset):
    def __init__(self, df, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = df
        # print(self.ds.head())
        # self.ds = df.drop(columns=['Unnamed: 0'])
        self.tokenizer_src = tokenizer_src  # Use the provided tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt  # Use the provided tokenizer_tgt
        self.src_lang = src_lang  # e.g., 'english'
        self.tgt_lang = tgt_lang  # e.g., 'spanish'
        self.seq_len = seq_len

        self.sos_token = torch.tensor(self.tokenizer_src.get_special_token_id('<sos>'), dtype=torch.int64)
        self.eos_token = torch.tensor(self.tokenizer_src.get_special_token_id('<eos>'), dtype=torch.int64)
        self.pad_token = torch.tensor(self.tokenizer_src.get_special_token_id('<pad>'), dtype=torch.int64)
        self.unk_token = torch.tensor(self.tokenizer_src.get_special_token_id('<unk>'), dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index) -> Any:
        # print(f"Accessing index: {index}")
        src_text = self.ds[self.src_lang][index]
        tgt_text = self.ds[self.tgt_lang][index]
        
        enc_input_tokens = self.tokenizer_src._encode_single(src_text)
        dec_input_tokens = self.tokenizer_tgt._encode_single(tgt_text)

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
        #     raise ValueError('Sentence is too long')
        if enc_num_padding_tokens < 0:
            enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
            enc_num_padding_tokens = 0

        if dec_num_padding_tokens < 0:
            dec_input_tokens = dec_input_tokens[:self.seq_len - 1]
            dec_num_padding_tokens = 0

        encoder_input = torch.cat([
            self.sos_token.unsqueeze(0),
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token.unsqueeze(0),
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ])

        decoder_input = torch.cat([
            self.sos_token.unsqueeze(0),
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token.unsqueeze(0),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (Seq_Len)
            "decoder_input": decoder_input,  # (Seq_Len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1,1,Seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1,Seq_Len) &(1,seq_len,seq_len)
            "label": label, # (seq_len)
            "src_txt": src_text,
            "tgt_text": tgt_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

def collate_fn(batch):
    # Stack encoder inputs, decoder inputs, and labels directly
    # Calculate the lengths of the encoder inputs
    lengths = [len(item['encoder_input']) for item in batch]
    
    # Sort the batch by the length of the source sequences in descending order
    batch = [x for _, x in sorted(zip(lengths, batch), key=lambda x: x[0], reverse=True)]
    encoder_inputs = torch.stack([item['encoder_input'] for item in batch])
    decoder_inputs = torch.stack([item['decoder_input'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    # Stack masks
    encoder_masks = torch.stack([item['encoder_mask'] for item in batch])
    decoder_masks = torch.stack([item['decoder_mask'] for item in batch])

    # Gather the source and target texts for reference
    src_texts = [item['src_txt'] for item in batch]
    tgt_texts = [item['tgt_text'] for item in batch]

    return {
        "encoder_input": encoder_inputs,
        "decoder_input": decoder_inputs,
        "encoder_mask": encoder_masks,
        "decoder_mask": decoder_masks,
        "label": labels,
        "src_txt": src_texts,
        "tgt_text": tgt_texts,
    }

