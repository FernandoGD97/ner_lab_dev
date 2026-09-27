"""End-to-end synthetic A0/A3 analysis operates only on stored Phase 2 outputs."""
import tempfile
from pathlib import Path
import pandas as pd
import yaml
from lab.ner.quantisation.analysis.workflow import analyse_results,compare_stored_predictions

def spans(path,rows):
 pd.DataFrame(rows,columns=["filename","start_span","end_span","text","label"]).to_csv(path,sep="\t",index=False)
with tempfile.TemporaryDirectory() as temporary:
 root=Path(temporary)
 for model in ("A0","A3"):(root/"models"/model/"predictions").mkdir(parents=True)
 gold=[["d1",0,4,"pain","problem"],["d2",0,5,"fever","problem"]]
 spans(root/"models/A0/predictions/gold.tsv",gold)
 spans(root/"models/A0/predictions/predictions.tsv",gold)
 spans(root/"models/A3/predictions/predictions.tsv",gold[:1])
 rows=[]
 for model,f1,size,time in (("A0",1.,100.,2.),("A3",2/3,50.,1.)):
  rows.append({"model_id":model,"status":"SUCCESS","f1_score":f1,"precision_score":f1,"recall_score":f1,"checkpoint_bytes":size,"parameters":size,"gpu_peak_allocated_mb":size,"model_time_s":time,"end_to_end_time_s":time*2,"total_energy_kwh":time/1000,"method_chain":'["fp32"]' if model=="A0" else '["int8"]',"precision":"float32" if model=="A0" else "int8","layers":1,"hidden_size":8,"intermediate_size":16,"vocab_size":10})
 pd.DataFrame(rows).to_csv(root/"benchmark_long.tsv",sep="\t",index=False)
 (root/"resolved_experiment.yaml").write_text(yaml.safe_dump({"models":[]}))
 training=root/"training.tsv"; pd.DataFrame([["t",0,4,"pain","problem"]],columns=["filename","start_span","end_span","text","label"]).to_csv(training,sep="\t",index=False)
 result=analyse_results(root,training,bootstrap_iterations=100)
 assert (root/"analysis/robustness.tsv").exists()
 assert (root/"analysis/pareto_f1_size.tsv").exists()
 assert (root/"analysis/correlations.tsv").exists()
 assert (root/"analysis/paper_main_results.tsv").exists()
 compared=compare_stored_predictions(root,"A0","A3",training)
 assert compared["gold_recovered_only_by_baseline"]==1
 assert Path(compared["details"]).exists()
