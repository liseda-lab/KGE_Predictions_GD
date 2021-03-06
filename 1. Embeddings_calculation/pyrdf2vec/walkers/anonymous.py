from typing import Any, List, Set, Tuple

import rdflib

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler, UniformSampler
from pyrdf2vec.walkers import RandomWalker


class AnonymousWalker(RandomWalker):
    """Defines the anonymous walking strategy.

    Attributes:
        depth: The depth per entity.
        walks_per_graph: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Default to UniformSampler().

    """

    def __init__(
        self,
        depth: int,
        walks_per_graph: float,
        sampler: Sampler = UniformSampler(),
    ):
        super().__init__(depth, walks_per_graph, sampler)

    def _extract(
        self, graph: KG, instances: List[rdflib.URIRef]
    ) -> Set[Tuple[Any, ...]]:
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            graph : The knowledge graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to extract the knowledge graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        canonical_walks = set()
        for instance in instances:
            walks = self.extract_random_walks(graph, str(instance))
            for walk in walks:
                canonical_walk = []
                str_walk = [str(x) for x in walk]  # type: ignore
                for i, hop in enumerate(walk):  # type: ignore
                    if i == 0:
                        canonical_walk.append(str(hop))
                    else:
                        canonical_walk.append(str(str_walk.index(str(hop))))
                canonical_walks.add(tuple(canonical_walk))
        return canonical_walks
