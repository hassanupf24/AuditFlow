# ============================================================
# auditflow/features/text_features.py
# TF-IDF vectorization with audit trail
# ============================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from typing import Optional, Tuple

from auditflow.core.logger import get_logger
from auditflow.core.registry import TransformerRegistry


def tfidf_vectorize(
    df: pd.DataFrame,
    text_col: str,
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    reduce_dims: Optional[int] = None,
    prefix: str = "tfidf",
    registry: Optional[TransformerRegistry] = None,
) -> Tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Convert a text column into TF-IDF features with audit trail.

    Parameters
    ----------
    text_col     : Column containing text.
    max_features : Maximum vocabulary size.
    ngram_range  : (1,1)=unigrams, (1,2)=unigrams+bigrams.
    min_df       : Minimum document frequency for a term.
    reduce_dims  : If set, apply TruncatedSVD to reduce dimensions.
    prefix       : Column name prefix for TF-IDF features.
    registry     : If provided, stores the fitted vectorizer.

    Returns
    -------
    (DataFrame with appended TF-IDF columns, fitted TfidfVectorizer)
    """
    audit = get_logger()

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
    )
    X_tfidf = vectorizer.fit_transform(df[text_col].fillna(""))
    vocab_size = len(vectorizer.vocabulary_)

    if reduce_dims:
        svd = TruncatedSVD(n_components=reduce_dims, random_state=42)
        X_reduced = svd.fit_transform(X_tfidf)
        explained = svd.explained_variance_ratio_.sum()
        cols = [f"{prefix}_svd_{i}" for i in range(reduce_dims)]
        tfidf_df = pd.DataFrame(X_reduced, columns=cols, index=df.index)

        audit.log_decision(
            module="features.text",
            action="tfidf_vectorize_svd",
            column=text_col,
            rationale=f"Vectorized '{text_col}' with TF-IDF (vocab={vocab_size}, "
                      f"ngrams={ngram_range}) then reduced to {reduce_dims}D via SVD "
                      f"(variance explained: {explained:.1%}). SVD compression makes "
                      f"TF-IDF features compatible with dense models.",
            details={
                "vocab_size": vocab_size,
                "ngram_range": list(ngram_range),
                "svd_dims": reduce_dims,
                "variance_explained": round(explained, 4),
            },
        )
    else:
        cols = [f"{prefix}_{t}" for t in vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=cols, index=df.index)

        audit.log_decision(
            module="features.text",
            action="tfidf_vectorize",
            column=text_col,
            rationale=f"Vectorized '{text_col}' with TF-IDF. Vocabulary size: {vocab_size}, "
                      f"n-gram range: {ngram_range}. Bigrams capture phrases like "
                      f"'not good' that unigrams would miss.",
            details={
                "vocab_size": vocab_size,
                "ngram_range": list(ngram_range),
                "output_features": len(cols),
            },
        )

    if registry:
        registry.register(
            name="tfidf_vectorizer",
            transformer=vectorizer,
            columns=[text_col],
            module="features.text",
            params={"max_features": max_features, "ngram_range": list(ngram_range)},
        )

    return pd.concat([df, tfidf_df], axis=1), vectorizer
