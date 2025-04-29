from bert_score import score
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define the ground truth based on trusted sources (e.g., NIH, medical textbooks)


ground_truth = [
    """
    """
]

# Responses from the base model (provided earlier)
base_model_output = [
    """
    """
]

# Responses from the RAG-based model (custom model with the database)
rag_model_output = [
    """
    """
    ]

# Compute BERTScore
def compute_bertscore(candidate_texts, reference_texts):
    """
    Compute BERT score for a candidate sentence against a reference sentence.
    """
    P, R, F1 = score(candidate_texts, reference_texts, lang="en")
    return F1.mean().item(), P.mean().item(), R.mean().item()

# Evaluate BERTScore for the Base Model
f1_base, p_base, r_base = compute_bertscore(base_model_output, ground_truth)
print(f"Base Model BERTScore F1: {f1_base}, Precision: {p_base}, Recall: {r_base}")

# Evaluate BERTScore for the RAG-based Model
f1_rag, p_rag, r_rag = compute_bertscore(rag_model_output, ground_truth)
print(f"RAG Model BERTScore F1: {f1_rag}, Precision: {p_rag}, Recall: {r_rag}")

# Initialize GPT-2 for Perplexity calculation
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model.eval()

def compute_perplexity(text, model, tokenizer):
    """
    Compute the perplexity of a given text using GPT-2.
    """
    inputs = tokenizer.encode(text, return_tensors="pt")
    input_ids = inputs

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        log_likelihood = outputs.loss

    perplexity = torch.exp(log_likelihood)
    return perplexity.item()

# Example for Perplexity evaluation on the Base Model output
perplexity_base = compute_perplexity(base_model_output[0], gpt2_model, gpt2_tokenizer)
print(f"Perplexity of the Base Model output: {perplexity_base}")

# Example for Perplexity evaluation on the RAG Model output
perplexity_rag = compute_perplexity(rag_model_output[0], gpt2_model, gpt2_tokenizer)
print(f"Perplexity of the RAG Model output: {perplexity_rag}")
