from preprocessing import Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt')
import config




from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(reference_texts, candidate_texts):
    smoothie = SmoothingFunction().method4
    total_bleu = 0
    count = len(reference_texts)

    for ref, pred in zip(reference_texts, candidate_texts):
        reference_tokens = [list(ref)]
        candidate_tokens = list(pred)

        
        bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
        total_bleu += bleu_score

    avg_bleu = total_bleu / count if count > 0 else 0
    return avg_bleu


