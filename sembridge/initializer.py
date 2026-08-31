from typing import Literal

import numpy as np
import torch
from fastdist import fastdist
from torch import Tensor
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizer
import entmax

from .semantic_embeddings import load_semantic_embedding_model
from .runtime_logging import logger
from .token_vocabulary import NewToken, OverlappingToken, canonicalize_vocab, construct_vocab_view, is_numerical_symbol_etc


def get_overlapping_tokens(
    target_tokenizer: PreTrainedTokenizer,
    source_tokenizer: PreTrainedTokenizer,
    match_symbols: bool,
    exact_match_all: bool,
    fuzzy_match_all: bool,
):
    """Returns overlapping tokens between two tokenizers. There are several options to select which tokens count as overlapping tokens.

    Args:
        target_tokenizer (PreTrainedTokenizer): The target tokenizer.
        source_tokenizer (PreTrainedTokenizer): The source tokenizer.
        match_symbols (bool): Tokens that satisfy `token.isnumeric() or all(c in string.punctuation for c in token) or token.isspace()` are considered.
        exact_match_all (bool): All tokens that match exactly are considered.
        fuzzy_match_all (bool): All tokens that match ignoring whitespace and case are considered.

    Returns:
        `(dict[str, OverlappingToken], dict[str, NewToken])`: A tuple with (1) information about overlapping tokens and (2) additional tokens in the target tokenizer.
    """
    target_vocab = target_tokenizer.get_vocab()
    source_vocab = source_tokenizer.get_vocab()

    canonical_source_vocab = canonicalize_vocab(source_vocab, source_tokenizer, "source")
    canonical_target_vocab = canonicalize_vocab(target_vocab, target_tokenizer, "target")

    overlap: dict[str, OverlappingToken] = {}
    additional_tokens: dict[str, NewToken] = {}
    exact_src_vocab = construct_vocab_view(canonical_source_vocab, "canonical_form")
    fuzzy_src_vocab = construct_vocab_view(canonical_source_vocab, "fuzzy_form")

    for _, target_token_info in tqdm(
        canonical_target_vocab.items(),
        desc="Getting overlapping tokens...",
        leave=False,
    ):
        # Exact match for symbols
        if (
            match_symbols
            and is_numerical_symbol_etc(target_token_info.fuzzy_form, target_tokenizer)
            and (exact_src_vocab.get(target_token_info.canonical_form) or fuzzy_src_vocab.get(target_token_info.fuzzy_form))
        ):
            overlap[target_token_info.native_form] = OverlappingToken(
                target=target_token_info,
                source=(
                    exact_src_vocab.get(target_token_info.canonical_form) or fuzzy_src_vocab.get(target_token_info.fuzzy_form)
                ),
                descriptor="numerical_symbol",
            )
        # General exact match
        elif exact_match_all and exact_src_vocab.get(target_token_info.canonical_form):
            overlap[target_token_info.native_form] = OverlappingToken(
                target=target_token_info,
                source=exact_src_vocab[target_token_info.canonical_form],
                descriptor="exact_match",
            )
        # General fuzzy match
        elif fuzzy_match_all and fuzzy_src_vocab.get(target_token_info.fuzzy_form):
            overlap[target_token_info.native_form] = OverlappingToken(
                target=target_token_info,
                source=fuzzy_src_vocab[target_token_info.fuzzy_form],
                descriptor="fuzzy_match",
            )
        # No match - it's a NewToken
        else:
            additional_tokens[target_token_info.native_form] = NewToken(target=target_token_info)
    return overlap, additional_tokens


@torch.no_grad()
def SemBridge(
    target_tokenizer: PreTrainedTokenizer,
    source_tokenizer: PreTrainedTokenizer,
    source_embeddings: Tensor,
    bridge_model_name_or_path: str,
    # match options
    exact_match_all: bool = True,
    match_symbols: bool = False,
    fuzzy_match_all: bool = False,
    extend_tokenizer: PreTrainedTokenizer | None = None,
    processes: int | None = None,
    batch_size: int = 1024,
    entmax_alpha: float = 2.5,
    seed: int = 42,
    device="cpu",
    verbosity: Literal["debug", "info", "silent"] = "info",
):
    """Transfer pretrained token embeddings to a target tokenizer using semantic auxiliary embeddings.

    Args:
        target_tokenizer (PreTrainedTokenizer): The new tokenizer in the target language.
        source_tokenizer (PreTrainedTokenizer): The tokenizer for the pretrained source embeddings.
        source_embeddings (Tensor): The pretrained source embeddings tensor.
        bridge_model_name_or_path (str): SentenceTransformer-compatible model name/path used to compute semantic bridge embeddings.
        exact_match_all (bool, optional): Match all overlapping tokens if they are an exact match. Defaults to True.
        match_symbols (bool, optional): Match overlapping symbolic tokens. Defaults to False.
        fuzzy_match_all (bool, optional): Match all overlapping tokens with fuzzy matching (whitespace and case). Defaults to False.
        extend_tokenizer (PreTrainedTokenizer | None, optional): If extending a tokenizer instead of vocabulary replacement, this should be the tokenizer that was used to extend the `source_tokenizer` (i.e. a target language specific tokenizer). The `target_tokenizer` should be the *extended* tokenizer. Defaults to None.
        processes (int | None, optional): Number of processes for parallelized workloads. Defaults to None, which uses heuristics based on available hardware.
        batch_size (int, optional): Batch size for semantic embedding extraction. Defaults to 1024.
        entmax_alpha (float, optional): Entmax alpha used to convert source-token similarities into sparse weights. Defaults to 2.5.
        seed (int, optional): Reserved for future stochastic paths. Defaults to 42.
        device (str | torch.device, optional): Defaults to "cpu".
        verbosity ("debug", "info", "silent", optional): Defaults to "info".

    Returns:
        Tensor: A tensor of shape `(len(target_tokenizer), embedding_dim)` with the initialized embeddings.
    """
    mode = {"debug": "dev", "info": "package", "silent": "silent"}[verbosity]
    logger.config(mode=mode)
    logger.info(f"Starting SemBridge initialization for target vocabulary with {len(target_tokenizer)} tokens...")
    
    ###################################################################
    # 1. Load bridge model embeddings for both source and target      #
    ###################################################################
    if not bridge_model_name_or_path:
        raise ValueError("bridge_model_name_or_path must be a non-empty model name or path.")

    target_auxiliary_model = load_semantic_embedding_model(
        target_tokenizer=extend_tokenizer or target_tokenizer,
        bridge_model_name_or_path=bridge_model_name_or_path,
        batch_size=batch_size,
        device=device,
        processes=processes,
    )
    
    logger.info("Loading auxiliary embeddings for source tokenizer...")
    source_auxiliary_model = load_semantic_embedding_model(
        target_tokenizer=source_tokenizer,
        bridge_model_name_or_path=bridge_model_name_or_path,
        batch_size=batch_size,
        device=device,
        processes=processes,
    )

    #################################################################
    # 2. Get overlapping tokens between source and target tokenizer #
    #################################################################
    overlapping_tokens, new_tokens = get_overlapping_tokens(
        target_tokenizer=target_tokenizer,
        source_tokenizer=source_tokenizer,
        match_symbols=match_symbols,
        exact_match_all=exact_match_all,
        fuzzy_match_all=fuzzy_match_all,
    )

    # Sort to ensure same order every time (especially important when executing on multiple ranks)
    sorted_overlapping_tokens = sorted(overlapping_tokens.items(), key=lambda x: x[1].target.id)
    sorted_new_tokens = sorted(new_tokens.items(), key=lambda x: x[1].target.id)
    logger.debug(f"Found {len(sorted_overlapping_tokens)} overlapping tokens.")

    ##########################################################
    # 3. Clean overlap + get auxiliary embeddings for tokens #
    ##########################################################
    # Clean overlapping tokens
    extend_tokenizer_vocab = extend_tokenizer.get_vocab() if extend_tokenizer else None

    for token, overlapping_token_info in tqdm(
        sorted_overlapping_tokens,
        desc="Populating auxiliary embeddings for overlapping token...",
        leave=False,
    ):
        embs_lst = [source_embeddings[s.id] for s in overlapping_token_info.source]
        # Ensure source embedding is on the correct device
        source_emb = embs_lst[0]
        if device is not None and source_emb.device != torch.device(device):
            source_emb = source_emb.to(device)
        overlapping_tokens[token].source_embedding = source_emb

        if len(embs_lst) > 1:
            logger.warning(
                f"{token} has multiple source embeddings (using first): {[s.native_form for s in overlapping_token_info.source][:min(5, len(embs_lst))]}"
            )

        overlapping_tokens[token].auxiliary_embedding = target_auxiliary_model[token]

    # Clean new tokens
    for token, new_token_info in tqdm(
        sorted_new_tokens,
        desc="Populating auxiliary embeddings for non-overlapping token...",
        leave=False,
    ):
        new_token_info.auxiliary_embedding = target_auxiliary_model[token]

    ####################################################
    # 4. Copy source embeddings for overlapping tokens #
    ####################################################
    target_embeddings = torch.zeros((len(target_tokenizer), source_embeddings.shape[1]), device=device)
    for _, overlapping_token in sorted_overlapping_tokens:
        target_embeddings[overlapping_token.target.id] = overlapping_token.source_embedding
    logger.success(f"Copied embeddings for {len(overlapping_tokens)} overlapping tokens.")

    ########################################################################
    # 5. Initialize additional tokens using all source-token similarities #
    ########################################################################
    target_embeddings = initialize_new_tokens_from_all_source(
        source_tokenizer,
        source_embeddings,
        source_auxiliary_model,
        new_tokens,
        target_embeddings,
        entmax_alpha=entmax_alpha,
        device=device,
    )
    logger.success(f"Initialized {len(new_tokens)} new tokens with SemBridge using all source tokens.")
    return target_embeddings.detach()


def initialize_new_tokens_from_all_source(
    source_tokenizer: PreTrainedTokenizer,
    source_embeddings: Tensor,
    source_auxiliary_model: dict,
    new_tokens: dict[str, NewToken],
    target_embeddings: Tensor,
    entmax_alpha: float,
    device: torch.device | str | None = None,
):
    """
    Initialize new tokens using similarity with ALL source tokens instead of just overlapping ones.
    This provides richer similarity information for better initialization.
    """
    # Convert new tokens to list for consistent ordering
    new_tokens_lst = list(new_tokens.values())
    
    # Get all source tokens and their info
    source_vocab = source_tokenizer.get_vocab()
    source_tokens = list(source_vocab.keys())
    source_token_ids = [source_vocab[token] for token in source_tokens]
    
    logger.info(f"Using {len(source_tokens)} source tokens for similarity computation")
    
    # Convert to numpy arrays for fastdist
    new_auxiliary_embedding_matrix = np.asarray([t.auxiliary_embedding.tolist() for t in new_tokens_lst], dtype="float32")
    source_auxiliary_embedding_matrix = np.asarray(
        [source_auxiliary_model[token].tolist() for token in source_tokens], 
        dtype="float32"
    )

    logger.debug("Computing distance matrix with all source tokens...")
    similarity_matrix = fastdist.cosine_matrix_to_matrix(
        new_auxiliary_embedding_matrix,
        source_auxiliary_embedding_matrix,
    )

    # Clean up memory
    del new_auxiliary_embedding_matrix
    del source_auxiliary_embedding_matrix

    logger.debug("Computing new embeddings using all source tokens...")

    # Prepare source embeddings tensor - use token IDs to get embeddings
    source_embs = source_embeddings[source_token_ids]  # [num_source_tokens, embedding_dim]
    
    # Move to the same device to avoid device mismatch
    if device is not None:
        source_embs = source_embs.to(device)

    for new_token_idx in tqdm(
        range(len(new_tokens_lst)),
        desc="SemBridge initialization with all source tokens...",
        total=len(new_tokens_lst),
    ):
        # Get similarity weights for this new token with all source tokens
        source_emb_weights: Tensor = entmax.entmax_bisect(
            torch.from_numpy(similarity_matrix[new_token_idx]).to(device),
            alpha=entmax_alpha,
            dim=0,
        )

        # Performance optimization - only use non-zero weights
        mask = source_emb_weights > 0.0
        masked_source_emb_weights = source_emb_weights[mask]
        masked_source_embs = source_embs[mask]

        # Compute weighted combination
        weighted_src_embs = torch.mul(masked_source_embs, masked_source_emb_weights.unsqueeze(1))
        convex_combination = torch.sum(weighted_src_embs, dim=0)

        # Assign to target embeddings
        new_token_target_vocab_idx = new_tokens_lst[new_token_idx].target.id
        target_embeddings[new_token_target_vocab_idx] = convex_combination
    
    return target_embeddings
