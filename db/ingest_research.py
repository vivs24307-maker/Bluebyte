"""
ingest_research.py
===================
Implements README.md's "NLP for Research Ingestion — auto-extraction of
insights from marine research papers and reports", at the storage
layer: chunk a paper's text, embed each chunk, store both, so
vector_queries.semantic_search_research() can retrieve the most
relevant passages for a natural-language question.

This module only does chunking + storage; actual PDF text extraction
is out of scope here (see /mnt/skills/public/pdf-reading if wiring up
real PDF ingestion later) — feed it a title + already-extracted text.

Usage (single doc):
    from db.ingest_research import ingest_document
    await ingest_document(
        title="Marine Heatwaves in the Arabian Sea",
        raw_text=full_text,
        source="INCOIS", domain="oceanography",
    )

Usage (batch from the synthetic dataset):
    python -m db.ingest_research --file db/synthetic_data/data/research_documents.json
"""

import argparse
import asyncio
import json

from db.connection import db_manager
from db.embeddings import chunk_text, embed_text, vector_literal


async def ingest_document(title: str, raw_text: str, source: str = None, url: str = None,
                            domain: str = None, published_date=None, conn=None) -> dict:
    """Inserts the document + its chunks (with embeddings) in one
    transaction. Returns {"document_id": ..., "chunk_count": ...}."""

    async def _do(c):
        async with c.transaction():
            doc_row = await c.fetchrow(
                """
                INSERT INTO research_documents (title, source, url, domain, published_date, raw_text)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                title, source, url, domain, published_date, raw_text,
            )
            document_id = doc_row["id"]

            chunks = chunk_text(raw_text)
            for idx, chunk in enumerate(chunks):
                vec = embed_text(chunk)
                await c.execute(
                    """
                    INSERT INTO research_chunks (document_id, chunk_index, content, token_count, embedding)
                    VALUES ($1, $2, $3, $4, $5::vector)
                    """,
                    document_id, idx, chunk, len(chunk.split()), vector_literal(vec),
                )
            return {"document_id": str(document_id), "chunk_count": len(chunks)}

    if conn is not None:
        return await _do(conn)
    pool = await db_manager.connect()
    async with pool.acquire() as c:
        return await _do(c)


async def ingest_batch_from_file(path: str) -> list:
    """Expects a JSON list of {title, raw_text, source, domain, url,
    published_date} objects — see db/synthetic_data/data/research_documents.json."""
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    pool = await db_manager.connect()
    results = []
    async with pool.acquire() as conn:
        for d in docs:
            result = await ingest_document(
                title=d["title"], raw_text=d["raw_text"], source=d.get("source"),
                url=d.get("url"), domain=d.get("domain"),
                published_date=d.get("published_date"), conn=conn,
            )
            results.append(result)
            print(f"Ingested '{d['title']}' -> {result['chunk_count']} chunks")
    return results


async def _main():
    parser = argparse.ArgumentParser(description="Ingest research documents into research_documents/research_chunks")
    parser.add_argument("--file", required=True, help="Path to a JSON file of documents")
    args = parser.parse_args()
    results = await ingest_batch_from_file(args.file)
    total_chunks = sum(r["chunk_count"] for r in results)
    print(f"Done: {len(results)} documents, {total_chunks} chunks")


if __name__ == "__main__":
    asyncio.run(_main())
