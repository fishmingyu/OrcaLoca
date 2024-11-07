import numpy as np
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
    text1, text2, threshold=0.95, model_name="bert-base-uncased"
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


def suggest_threshold(example_pairs, model_name="bert-base-uncased"):
    """
    Suggest a threshold based on example similar and dissimilar pairs.

    Parameters:
    - example_pairs: List of tuples (text1, text2, is_similar)
    Returns suggested threshold value
    """
    similarities = []
    labels = []

    for text1, text2, is_similar in example_pairs:
        sim = check_observation_similarity(text1, text2, model_name=model_name)
        similarities.append(sim)
        labels.append(is_similar)

    # Convert to numpy arrays
    similarities = np.array(similarities)
    labels = np.array(labels)

    # Find the middle point between the max similarity of dissimilar pairs
    # and min similarity of similar pairs
    similar_mins = similarities[labels]
    dissimilar_maxs = similarities[~labels]

    suggested_threshold = (np.min(similar_mins) + np.max(dissimilar_maxs)) / 2

    return suggested_threshold


if __name__ == "__main__":
    # Example usage
    text1 = r"""
    "observation_feedback": "Based on the search results, the bug occurs in proxy model metadata handling. The key issues are in RelatedPopulator.__init__ where it uses self.model_cls._meta.pk.attname without considering proxy models, and in Query.deferred_to_data where the metadata handling needs to use concrete_model._meta for proxy models. The error occurs because init_list is constructed using proxy model fields but pk lookup uses direct model metadata.",
    """
    text2 = r"""
    "observation_feedback": "Based on the search results, the bug occurs in multiple locations where proxy model metadata handling is incorrect. The RelatedPopulator.__init__ method uses self.model_cls._meta.pk.attname without considering proxy models, and Query.deferred_to_data uses cur_model._meta directly instead of concrete_model._meta. The error 'id not in list' occurs because init_list is constructed using proxy model fields but pk lookup uses direct model metadata.",
    """
    text3 = r"""
    "observation_feedback": "Based on the search results, the bug appears in multiple locations where proxy model metadata handling is incorrect. The main issues are in RelatedPopulator.__init__ where it uses self.model_cls._meta.pk.attname without considering proxy models, and in Query.deferred_to_data where the metadata handling needs to use concrete_model._meta for proxy models. The error occurs because the init_list is constructed using proxy model fields but pk lookup uses direct model metadata.",
    """
    text4 = r"""
    "observation_feedback": "Based on the search results, the bug appears in multiple locations where proxy model metadata handling is incorrect. The main issues are in RelatedPopulator.__init__ where it uses self.model_cls._meta.pk.attname without considering proxy models, and in Query.deferred_to_data where the metadata handling needs to use concrete_model._meta for proxy models. The error 'id not in list' occurs because the init_list is constructed using proxy model fields but pk lookup uses direct model metadata.",
    """
    similarity_score = check_observation_similarity(text1, text2)
    print(f"Similarity score: {similarity_score}")
    similarity_score = check_observation_similarity(text2, text3)
    print(f"Similarity score: {similarity_score}")
    similarity_score = check_observation_similarity(text3, text4)
    print(f"Similarity score: {similarity_score}")

    example_pairs = [
        ("The cat sat on the mat.", "The dog lay on the rug.", False),
        ("A quick brown fox jumps over the lazy dog.", "The dog is sleeping.", False),
        (text1, text2, True),
        (text2, text3, True),
        (text3, text4, True),
        (text1, text4, True),
    ]
    suggested_threshold = suggest_threshold(example_pairs)
    print(f"Suggested threshold: {suggested_threshold}")
