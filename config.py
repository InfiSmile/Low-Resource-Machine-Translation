from pathlib import Path
def get_config():
  return{
      "batch_size":20,
      "num_epochs":30,
      "lr": 10**-3,
      "seq_len": 30,
      "d_model":256,
      "lang_src":"Hindi",
      "lang_tgt":"Bhili",
      "model_folder": "weights",
      "model_basename": "tmodel_",
      # "preload":None,
      "preload": "latest",
      "tokenizer_file":"tokenizer_{}.json",
      "experiment_name": "runs/tmodel",
      "vocab_hi_path" : 'vocab/vocab_hindi.json',
      "vocab_bhi_path" : 'vocab/vocab_bhili.json',
      "translated_csv_file" :'translated_sentences.csv'
      
  }

def get_weights_file_path(config,_epoch:str):
  model_folder= config['model_folder']
  model_basename= config['model_basename']
  model_filename= f"{model_basename}{_epoch}.pt"
  return str(Path('.')/ model_folder/model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])