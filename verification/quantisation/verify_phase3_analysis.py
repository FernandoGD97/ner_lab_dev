"""Synthetic Pareto, bootstrap, correlations, and paper-table tests."""
import pandas as pd
from lab.ner.quantisation.analysis.pareto import pareto_front
from lab.ner.quantisation.analysis.statistics import paired_bootstrap
from lab.ner.quantisation.analysis.reporting import correlations,paper_tables,compression_family

data=pd.DataFrame({"model_id":["A0","A1","A2"],"status":["SUCCESS"]*3,"f1_score":[.9,.89,.8],"precision_score":[.9,.89,.8],"recall_score":[.9,.89,.8],"checkpoint_bytes":[100,50,120],"parameters":[100,100,120],"gpu_peak_allocated_mb":[100,80,120],"model_time_s":[2,1,3],"end_to_end_time_s":[4,3,5],"total_energy_kwh":[.01,.008,.02],"layers":[12,12,12],"hidden_size":[768]*3,"intermediate_size":[3072]*3,"vocab_size":[1000]*3,"method_chain":["[\"fp32\"]","[\"fp16\"]","[\"depth-reduction\"]"],"precision":["float32","float16","float32"]})
front=pareto_front(data,cost="checkpoint_bytes")
assert set(front.loc[front.pareto_optimal,"model_id"])=={"A0","A1"}
corr = correlations(data)
assert set(corr.method) == {"pearson", "spearman"}
assert compression_family('["fp16","int8"]')=="combined"
tables=paper_tables(data)
assert {"model_characteristics","main_results","robustness"}==set(tables)
assert "Delta F1" in tables["main_results"]

def spans(rows): return pd.DataFrame(rows,columns=["filename","start_span","end_span","text","label"])
gold=spans([["a",0,2,"aa","X"],["b",0,2,"bb","X"]])
baseline=gold.copy(); compressed=gold.iloc[:1].copy()
boot=paired_bootstrap(gold,baseline,compressed,iterations=200,seed=7)
assert boot["delta"]<0 and boot["ci_low"]<=boot["delta"]<=boot["ci_high"]
