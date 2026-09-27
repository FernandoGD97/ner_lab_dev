"""Synthetic CPU-only Phase 3 NER robustness and prediction tests."""
import pandas as pd
from collections import Counter
from lab.ner.quantisation.analysis.errors import compare_predictions,error_taxonomy
from lab.ner.quantisation.analysis.robustness import annotate_gold,frequency_bucket,robustness_table,seen_unseen,subword_bucket

def spans(rows): return pd.DataFrame(rows,columns=["filename","start_span","end_span","text","label"])
gold=spans([
 ["d",0,4,"aaaa","X"],["d",10,14,"bbbb","X"],["d",20,24,"cccc","X"],
 ["d",30,34,"dddd","X"],["d",40,44,"eeee","X"],["d",50,54,"ffff","X"],
 ["d",60,64,"gggg","X"],
])
predicted=spans([
 ["d",0,4,"aaaa","X"],                    # correct
 ["d",10,14,"bbbb","Y"],                  # wrong label
 ["d",19,24," cccc","X"],                 # left
 ["d",30,35,"dddd ","X"],                  # right
 ["d",39,45," eeee ","X"],                # both
 ["d",49,53," fff","Y"],                   # overlapping
 ["d",70,74,"extra","X"],                 # false positive; gggg is FN
])
errors=error_taxonomy(gold,predicted)
assert {"correct","wrong_label","left_boundary_error","right_boundary_error","both_boundaries_error","overlapping_prediction","false_negative","false_positive"} == set(errors.error_type)
assert [frequency_bucket(n) for n in (0,1,2,6,21)] == ["unseen","singleton","2-5","6-20",">20"]
assert seen_unseen(0)=="unseen" and seen_unseen(1)=="seen"
assert [subword_bucket(n) for n in (1,2,3,5,None)] == ["1","2","3-4","5+","unavailable"]
annotated=annotate_gold(gold,Counter({"aaaa":1}))
assert annotated.loc[0,"frequency_bucket"]=="singleton" and annotated.loc[1,"seen_unseen"]=="unseen"
robust=robustness_table(gold,predicted,Counter({"aaaa":1}))
assert {"precision","recall","f1","support"} <= set(robust)
summary,detail=compare_predictions(gold,predicted,gold.iloc[:1])
assert summary["gold_recovered_by_both"]==1
assert summary["gold_recovered_only_by_baseline"]==0
assert not detail.empty
