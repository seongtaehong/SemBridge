"""SemBridge: semantic tokenizer-embedding bridge utilities."""

__version__ = "0.1.0"


def SemBridge(*args, **kwargs):
    """Build target-tokenizer embeddings from source embeddings via semantic bridging.

    The implementation is imported lazily so `import sembridge` does not require
    ML dependencies until the transfer function is called.
    """
    from .initializer import SemBridge as _SemBridge

    return _SemBridge(*args, **kwargs)


SEMBRIDGE = SemBridge

__all__ = ["SemBridge", "SEMBRIDGE", "__version__"]
