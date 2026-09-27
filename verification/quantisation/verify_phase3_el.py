"""Tiny EL fixtures for Recall@K, MRR, K validation, and cost schemas."""
from collections import Counter
from lab.ner.quantisation.entity_linking.metrics import concept_frequency_bucket,el_metrics,seen_unseen_metrics
from lab.ner.quantisation.entity_linking.schemas import EL_COLUMNS,OfflineCosts,OnlineCosts,energy_normalization
metrics=el_metrics(["A","B"],[["A","C"],["X","B"]],[1,2])
assert metrics["accuracy_at_1"]==.5
assert metrics["recall@2"]==1
assert metrics["mrr"]==.75
seen=seen_unseen_metrics(["A","B"],[["A"],["X"]],Counter({"A":3}))
assert seen["seen_accuracy"]==1 and seen["unseen_accuracy"]==0
assert concept_frequency_bucket(0)=="unseen" and concept_frequency_bucket(21)==">20"
assert OfflineCosts().index_build_time_s is None and OnlineCosts().retrieval_time_s is None
assert {"index_type","candidate_k","biencoder_time_s","accuracy_at_1","seen_accuracy"} <= set(EL_COLUMNS)
normalized=energy_normalization(.001,100,1000,100)
assert normalized["joules_per_1000_mentions"]==36000
