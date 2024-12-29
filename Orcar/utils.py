import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def get_bert_embedding(text, model, tokenizer, device="cuda"):
    """Get BERT embeddings for a given text."""
    # Tokenize and move to device
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state

        # Use mean pooling to get sentence embedding
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        sentence_embedding = summed / torch.clamp(mask.sum(1), min=1e-9)

        # Normalize embeddings
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)

    return sentence_embedding[0]


def check_observation_similarity(
    text1, text2, threshold=0.97, model_name="bert-base-uncased"
):
    """
    Check similarity between two paragraphs using BERT embeddings.
    Returns similarity score and boolean indicating if paragraphs are similar.

    Parameters:
    - text1, text2: Input paragraphs to compare
    - threshold: Optional float between 0 and 1. If None, returns only similarity score
    - model_name: Name of the BERT model to use
    """
    # Initialize model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # Get embeddings
    emb1 = get_bert_embedding(text1, model, tokenizer, device)
    emb2 = get_bert_embedding(text2, model, tokenizer, device)

    # Calculate dot product as similarity score
    similarity = torch.dot(emb1, emb2).item()

    if threshold is None:
        return similarity

    return similarity, similarity >= threshold
