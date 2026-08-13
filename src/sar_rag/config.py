from dataclasses import dataclass, field


@dataclass(frozen=True)
class SarRagConfig:
    """Frozen online controls. Probe/batch sizes never define final evidence k."""

    probe_depth: int = 3
    adaptive_batch_size: int = 3
    # Top-3 probes need a small RRF constant; 60 makes ranks 1-3 almost equal.
    rrf_k0: int = 3
    max_selected_representations: int = 2
    representation_marginal_gain_threshold: float = 0.10
    similarity_threshold: float = 0.70
    similarity_gap_threshold: float = 0.75
    novelty_threshold: float = 0.10
    redundancy_threshold: float = 0.92
    marginal_utility_threshold: float = 0.05
    empty_batch_patience: int = 2
    max_candidates_examined: int = 60
    max_retrieval_rounds: int = 8
    max_final_context_tokens: int = 2_000
    max_final_chunks: int = 20
    confirmation_rounds: int = 1
    max_reformulations_per_requirement: int = 1
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    local_embedding_path: str | None = "models/all-MiniLM-L6-v2"
    groq_model: str = "openai/gpt-oss-120b"
    # Registered ablation switches.  All are fixed to the primary method by default.
    representation_selection_mode: str = "rrf"  # rrf | all
    adaptive_retrieval_enabled: bool = True
    novelty_enabled: bool = True
    similarity_gap_enabled: bool = True
    post_retrieval_deduplication_enabled: bool = True
    enabled_representations: tuple[str, ...] = field(default_factory=lambda: (
        "sentence_window",
        "paragraph",
        "semantic",
        "hierarchical",
    ))

    def __post_init__(self) -> None:
        if self.probe_depth < 1:
            raise ValueError("probe_depth must be positive")
        if self.adaptive_batch_size < 1:
            raise ValueError("adaptive_batch_size must be positive")
        if self.rrf_k0 < 1:
            raise ValueError("rrf_k0 must be positive")
        if self.max_selected_representations not in {1, 2}:
            raise ValueError("primary method selects one or two representations")
        for name, value in (
            ("representation_marginal_gain_threshold", self.representation_marginal_gain_threshold),
            ("novelty_threshold", self.novelty_threshold),
            ("redundancy_threshold", self.redundancy_threshold),
            ("marginal_utility_threshold", self.marginal_utility_threshold),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not -1 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be a relevance percentile in [0, 1]")
        if not 0 <= self.similarity_gap_threshold <= 5:
            raise ValueError("similarity_gap_threshold must be a standardized gap in [0, 5]")
        if self.empty_batch_patience < 1:
            raise ValueError("empty_batch_patience must be positive")
        if self.max_final_context_tokens < 1 or self.max_final_chunks < 1:
            raise ValueError("final context budgets must be positive")
        if not self.enabled_representations:
            raise ValueError("at least one representation must be enabled")
        if self.representation_selection_mode not in {"rrf", "all"}:
            raise ValueError("representation_selection_mode must be 'rrf' or 'all'")
        if self.max_candidates_examined < self.probe_depth * len(self.enabled_representations):
            raise ValueError("candidate budget must cover the mandatory all-representation probe")
        if self.confirmation_rounds != 1:
            raise ValueError("this preregistered implementation uses exactly one empty confirmation round")
        if self.max_reformulations_per_requirement != 1:
            raise ValueError("this preregistered implementation uses exactly one reformulation per requirement")
