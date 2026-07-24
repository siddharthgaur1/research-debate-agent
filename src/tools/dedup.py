"""Collapse near-duplicate sources with ChromaDB.

Web search loves to hand back the same wire story republished by six outlets. Left
alone, the Advocate cites "six sources" that are one press release, and the whole
debate inherits a false sense of weight. So sources are embedded and anything at or
above `dedup_threshold` cosine similarity is folded into the first one seen.

The survivor keeps the duplicates' urls in `merged_from`, so the citation trail can
still show the story ran in six places without pretending that is six pieces of
evidence.
"""

from __future__ import annotations

from collections.abc import Iterable

import chromadb
from langchain_openai import OpenAIEmbeddings

from ..config import get_settings
from ..state.schema import Source


def _client() -> chromadb.ClientAPI:
    """Chroma over HTTP when a host is configured, else a local file store.

    Same call sites either way: compose sets CHROMA_HOST, a laptop doesn't.
    """
    settings = get_settings()
    if settings.chroma_host:
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of source texts."""
    settings = get_settings()
    model = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    return model.embed_documents(texts)


def _fingerprint(source: Source) -> str:
    """What we embed: enough to spot a reprint, not the whole article."""
    return f"{source.title}\n{source.snippet}\n{source.text[:2000]}".strip()


def collection_for(run_id: str):
    """A per-run Chroma collection, configured for cosine distance.

    Chroma defaults to L2; without this the similarity threshold would be
    comparing against a distance that isn't cosine at all.
    """
    return _client().get_or_create_collection(
        name=f"run_{run_id}", metadata={"hnsw:space": "cosine"}
    )


def dedup_sources(run_id: str, sources: Iterable[Source]) -> list[Source]:
    """Return the pool with near-duplicates merged into their first occurrence.

    Order is preserved and the first-seen source wins, so results stay stable
    across reruns of the same pool.
    """
    sources = list(sources)
    if not sources:
        return []

    collection = collection_for(run_id)
    embeddings = _embed([_fingerprint(s) for s in sources])
    threshold = get_settings().dedup_threshold

    kept: list[Source] = []
    by_embedding_id: dict[str, Source] = {}

    for source, embedding in zip(sources, embeddings):
        duplicate_of = _nearest_duplicate(
            collection, embedding, threshold, by_embedding_id
        )
        if duplicate_of is not None:
            if source.url not in duplicate_of.merged_from:
                duplicate_of.merged_from.append(source.url)
            continue

        collection.add(
            ids=[source.id],
            embeddings=[embedding],
            metadatas=[{"url": source.url, "title": source.title}],
        )
        source.embedding_id = source.id
        by_embedding_id[source.id] = source
        kept.append(source)

    return kept


def _nearest_duplicate(
    collection, embedding: list[float], threshold: float, known: dict[str, Source]
) -> Source | None:
    """The already-kept source this embedding duplicates, or None."""
    if not known:
        return None
    result = collection.query(
        query_embeddings=[embedding], n_results=1, include=["distances"]
    )
    ids = (result.get("ids") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    if not ids or not distances:
        return None
    # cosine space: similarity = 1 - distance
    if (1.0 - float(distances[0])) >= threshold:
        return known.get(ids[0])
    return None
