import anthropic
import torch
import torch.nn.functional as F
from llama_index.llms.anthropic import Anthropic
from transformers import AutoModel, AutoTokenizer


class VertexAnthropicWithCredentials(Anthropic):
    def __init__(self, credentials, **kwargs):
        """
        In addition to all parameters accepted by Anthropic, this class accepts a
        new parameter `credentials` that will be passed to the underlying clients.
        """
        # Pop parameters that determine client type so we can reuse them in our branch.
        region = kwargs.get("region")
        project_id = kwargs.get("project_id")
        aws_region = kwargs.get("aws_region")

        # Call the parent initializer; this sets up a default _client and _aclient.
        super().__init__(**kwargs)

        # If using AnthropicVertex (i.e., region and project_id are provided and aws_region is None),
        # override the _client and _aclient with the additional credentials parameter.
        if region and project_id and not aws_region:
            self._client = anthropic.AnthropicVertex(
                region=region,
                project_id=project_id,
                credentials=credentials,  # extra argument
                timeout=self.timeout,
                max_retries=self.max_retries,
                default_headers=kwargs.get("default_headers"),
            )
            self._aclient = anthropic.AsyncAnthropicVertex(
                region=region,
                project_id=project_id,
                credentials=credentials,  # extra argument
                timeout=self.timeout,
                max_retries=self.max_retries,
                default_headers=kwargs.get("default_headers"),
            )
        # Optionally, you could add similar overrides for the aws_region branch if needed.


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
