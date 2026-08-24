"""
demo_vector_search.py
======================
Prints a live walkthrough of every vector-search capability against
whatever data is currently loaded (run after load_synthetic_dataset.py).
Useful both as a smoke test and as the literal script for a demo.

Usage:
    python -m db.demo_vector_search
"""

import asyncio

from db.connection import db_manager
from db import (
    semantic_search_species,
    find_similar_species,
    hybrid_species_search,
    find_similar_edna_sequences,
    semantic_search_research,
    semantic_search_grids,
)
from db.embeddings import using_real_model


def _print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


async def main():
    await db_manager.connect()
    print(f"Embedding backend: {'sentence-transformers (all-MiniLM-L6-v2)' if using_real_model() else 'offline hashing fallback'}")

    _print_section("1) Semantic species search: 'small oily schooling fish, high commercial value'")
    for r in await semantic_search_species("small oily schooling fish, high commercial value", top_k=5):
        print(f"  {r['similarity']:.3f}  {r['common_name']} ({r['scientific_name']}) — {r['habitat_type']}")

    _print_section("2) 'More like this': species similar to Indian Mackerel")
    species = await semantic_search_species("Indian Mackerel Rastrelliger kanagurta", top_k=1)
    if species:
        anchor = species[0]
        print(f"  Anchor: {anchor['common_name']}")
        for r in await find_similar_species(anchor["id"], top_k=5):
            print(f"    {r['similarity']:.3f}  {r['common_name']} ({r['scientific_name']})")

    _print_section("3) Hybrid search: 'large predatory pelagic fish' AND tolerates SST=29C")
    for r in await hybrid_species_search("large predatory pelagic fish", sst=29.0, top_k=5):
        print(f"  {r['similarity']:.3f}  {r['common_name']} — SST range [{r['min_sst']}, {r['max_sst']}]")

    _print_section("4) eDNA nearest-neighbor search on a fresh (unmutated) Yellowfin Tuna-like fragment")
    from db.synthetic_data.generate_synthetic_dataset import _reference_sequence
    probe = _reference_sequence("Thunnus albacares")[40:180]  # simulate a partial field read
    for r in await find_similar_edna_sequences(probe, top_k=5):
        print(f"  {r['similarity']:.3f}  {r['common_name']} ({r['scientific_name']}) "
              f"marker={r['marker_gene']} confidence={r['detection_confidence']}")

    _print_section("5) RAG-style research search: 'why is dissolved oxygen dropping near Kerala?'")
    for r in await semantic_search_research("why is dissolved oxygen dropping near Kerala", top_k=3):
        print(f"  {r['similarity']:.3f}  {r['title']}  [{r['domain']}]")
        print(f"      ...{r['content'][:160]}...")

    _print_section("6) Grid ecological similarity: 'warm hypoxic water'")
    for r in await semantic_search_grids("warm hypoxic water with low species diversity", top_k=5):
        print(f"  {r['similarity']:.3f}  h3={r['h3_index']}  sst={r['avg_sst']}  "
              f"do={r['avg_dissolved_oxygen']}  species_richness={r['species_richness']}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
