import torch
from bert_score import score
import numpy as np
from tqdm import tqdm
from bleurt import score as BleurtScore
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt, SelfCheckNLI # pyright: ignore[reportMissingImports]
import spacy # pyright: ignore[reportMissingImports]
import statistics
#from PandaLM.pandalm import EvaluationPipeline
from Config import *
from transformers import DebertaV2Tokenizer
def custom_batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
    # SelfCheckGPT passes a list of [premise, hypothesis] lists/tuples
    # We must separate them into 'text' and 'text_pair' lists for the tokenizer
    if isinstance(batch_text_or_text_pairs, list) and len(batch_text_or_text_pairs) > 0:
        # Check if the items are pairs (lists or tuples)
        first_item = batch_text_or_text_pairs[0]
        if isinstance(first_item, (list, tuple)) and len(first_item) == 2:
            text = [item[0] for item in batch_text_or_text_pairs]
            text_pair = [item[1] for item in batch_text_or_text_pairs]
            return self(text=text, text_pair=text_pair, **kwargs)
            
    # Fallback for standard usage
    return self(text=batch_text_or_text_pairs, **kwargs)

if not hasattr(DebertaV2Tokenizer, "batch_encode_plus"):
    DebertaV2Tokenizer.batch_encode_plus = custom_batch_encode_plus
"""
Evaluation Metrics
"""

def BERTScore(generated_responses, references):
    print("\nCalculating BERTScore")
    P, R, F1 = score(
        generated_responses,
        references,
        lang="en",
        verbose=True,
        device=DEVICE,
        rescale_with_baseline=True,
        return_hash = False
    )
    # Convert to numpy arrays
    P = P.numpy()
    R = R.numpy()
    F1 = F1.numpy()

    # Calculate statistics
    metrics = {
        "precision": {
            "mean": float(np.mean(P)),
            "std": float(np.std(P))
        },
        "recall": {
            "mean": float(np.mean(R)),
            "std": float(np.std(R))
        },
        "f1": {
            "mean": float(np.mean(F1)),
            "std": float(np.std(F1))
        },
        "total_examples": len(references)
    }
    return metrics, P, R, F1
    
def BLEURT_Score(generated_responses, references):
    checkpoint = "bleurt/BLEURT-20"
    scorer = BleurtScore.BleurtScorer(checkpoint)
    scores = scorer.score(references=references, candidates=generated_responses)
    mean = statistics.fmean(scores)
    return mean

def SelfCheckGPT(generator, N):
    prompt = """
    Give me the professional journey of Louis de Broglie in detail.
    Answer: 
    """
    Response = generator(prompt, do_sample=False, max_new_tokens=1024, return_full_text=False)
    Samples = generator(
    [prompt] * N,
    temperature=1.0,
    do_sample=True,
    max_new_tokens=1024,
    return_full_text=False,
    )
    Response = Response[0]["generated_text"]
    # print(f"Response:{Response}")
    Samples = [sample[0]["generated_text"] for sample in Samples]
    # print(f"Sample:{Samples}")
    selfcheck_nli = SelfCheckNLI(device=DEVICE)
    nlp = spacy.load("en_core_web_sm")
    sentences = [
        sent.text.strip() for sent in nlp(Response).sents
    ] 
    sent_scores_nli = selfcheck_nli.predict(
        sentences=sentences,  # list of sentences
        sampled_passages=Samples,  # list of sampled passages
    )
    # print(sent_scores_nli)
    return np.mean(sent_scores_nli)

#def PandaLMEvaluate():
#    pipeline = EvaluationPipeline(candidate_paths = ["Llama-3.2-3B-Instruct", "FFT_Model", "HFT_Model", "QLoRA_Model"], input_data_path = "Dataset/test.json")
#    print(pipeline.evaluate())

def PerplexityScore(input_texts, tokenizer, model, device):
    max_length = 2048
    stride = 1024
    encodings = tokenizer("\n\n".join(input_texts), return_tensors = "pt")
    seq_len = encodings.input_ids.size(1)
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        num_loss_tokens = trg_len - 1
        if num_loss_tokens <= 0:
            continue# subtract batch_size due to internal label shift
        
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    if n_tokens == 0:
        return float('inf')

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    return ppl
