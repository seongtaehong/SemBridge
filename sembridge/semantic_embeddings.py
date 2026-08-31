import torch
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from sentence_transformers import SentenceTransformer

from .runtime_logging import logger


def load_semantic_embedding_model(
    target_tokenizer: PreTrainedTokenizer,
    bridge_model_name_or_path: str,
    batch_size: int = 1024,
    device: str = "cpu",
    processes=None,
):
    """
    Load a SentenceTransformer-compatible semantic embedding model and create token embeddings for a tokenizer.
    
    Args:
        target_tokenizer (PreTrainedTokenizer): The target tokenizer.
        bridge_model_name_or_path (str): Path or name of the semantic bridge embedding model.
        batch_size (int): Batch size for embedding extraction.
        device (str): Device to run the model on.
        processes (int): Number of processes (not used but kept for compatibility).
    
    Returns:
        dict: Dictionary mapping tokens to their embeddings.
    """
    logger.info(f"Loading semantic embedding model: {bridge_model_name_or_path}")
    model = SentenceTransformer(bridge_model_name_or_path, device=device)
    
    vocab = target_tokenizer.get_vocab()
    tokens = list(vocab.keys())
    
    logger.info(f"Extracting embeddings for {len(tokens)} tokens...")
    
    # Process tokens in batches
    token_embeddings = {}
    for i in tqdm(range(0, len(tokens), batch_size), desc="Extracting token embeddings"):
        batch_tokens = tokens[i:i + batch_size]
        
        # Decode tokens to get actual text
        decoded_tokens = []
        for token in batch_tokens:
            try:
                decoded_token = target_tokenizer.decode(vocab[token]).strip()
                # Handle special tokens and empty strings
                if not decoded_token or decoded_token.startswith('[') and decoded_token.endswith(']'):
                    decoded_token = token
                decoded_tokens.append(decoded_token)
            except Exception:
                decoded_tokens.append(token)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = model.encode(decoded_tokens, convert_to_numpy=True, show_progress_bar=False)
        
        # Store embeddings
        for token, embedding in zip(batch_tokens, embeddings):
            token_embeddings[token] = embedding
    
    logger.success(f"Extracted embeddings for {len(token_embeddings)} tokens")
    return token_embeddings
