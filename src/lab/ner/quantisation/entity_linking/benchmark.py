"""Instrument the existing NEL candidate generators and reranker; no second linker."""
from __future__ import annotations
import copy,importlib,json,time
from collections import Counter
from pathlib import Path
import pandas as pd
from .config import load_el_experiment
from .metrics import DEFAULT_K,el_metrics,seen_unseen_metrics
from .schemas import EL_COLUMNS,energy_normalization

def _timed(function):
    start=time.perf_counter(); value=function(); return value,time.perf_counter()-start

def _nel_components():
    linking=importlib.import_module("lab.nel.linking")
    pipeline=importlib.import_module("lab.nel.pipeline")
    return linking,pipeline.EntityLinkingPipeline

def _index_bytes(generator):
    index=getattr(generator,"index",None)
    if index is None:index=getattr(getattr(generator,"retriever",None),"index",None)
    if index is None:return None
    try:
        faiss=importlib.import_module("faiss")
        if hasattr(faiss,"index_gpu_to_cpu") and "Gpu" in type(index).__name__:index=faiss.index_gpu_to_cpu(index)
        return int(faiss.serialize_index(index).nbytes)
    except Exception:return None

def validate_el(config_or_path):
    config=config_or_path if hasattr(config_or_path,"task") else load_el_experiment(config_or_path)
    status=[]
    for role,path in (("mentions",config.task.mentions),("gazetteer",config.task.gazetteer)):
        status.append({"component":role,"status":"READY" if path.exists() else "MISSING","path":str(path)})
    supported_index=config.retrieval.index_type in {"flat_ip","flat_l2"}
    status.append({"component":"index","status":"READY" if supported_index else "UNSUPPORTED","index_type":config.retrieval.index_type,"reason":None if supported_index else "Current repository FAISS implementation exposes flat indexes only."})
    if config.crossencoder.enabled: status.append({"component":"crossencoder","status":"READY" if config.crossencoder.model.exists() else "MISSING","path":str(config.crossencoder.model)})
    return {"experiment_id":config.experiment.id,"components":status,"ready":all(item["status"]=="READY" for item in status)}

def run_el_experiment(path):
    linking,EntityLinkingPipeline=_nel_components()
    build_candidate_generator=linking.build_candidate_generator; build_reranker=linking.build_reranker
    gazetteer_entries=linking.gazetteer_entries; mentions_from_spans=linking.mentions_from_spans
    read_gazetteer=linking.read_gazetteer; read_spans=linking.read_spans
    config=load_el_experiment(path); validation=validate_el(config); root=config.output.directory/config.experiment.id; root.mkdir(parents=True,exist_ok=True)
    if not validation["ready"]:
        (root/"el_validation.json").write_text(json.dumps(validation,indent=2)+"\n"); return validation
    spans,context_time=_timed(lambda:read_spans(config.task.mentions)); gazetteer=read_gazetteer(config.task.gazetteer); entries=gazetteer_entries(gazetteer); mentions=mentions_from_spans(spans)
    compression=None
    if config.biencoder.model:
        manifest=Path(config.biencoder.model)/"compression_manifest.json"
        if manifest.exists():compression=json.dumps(json.loads(manifest.read_text()).get("method_chain",[]))
    method_kwargs=dict(config.retrieval.method_kwargs)
    similarity = "dot" if config.retrieval.distance_metric == "ip" else config.retrieval.distance_metric
    if config.retrieval.method in {"matrix","faiss"}:method_kwargs.setdefault("similarity",similarity)
    generator,index_build_time=_timed(lambda:build_candidate_generator(config.retrieval.method,entries,gazetteer,config.biencoder.model,max(config.retrieval.candidate_k),method_kwargs))
    index_size_bytes,index_serialization_time=_timed(lambda:_index_bytes(generator))
    if index_size_bytes is None:index_serialization_time=None
    term_vectors=getattr(generator,"_term_vectors",None)
    if term_vectors is None:term_vectors=getattr(getattr(generator,"retriever",None),"_term_vectors",None)
    embedding_memory_bytes=None if term_vectors is None else int(term_vectors.nbytes)
    reranker=build_reranker(config.crossencoder.model,{"device":"auto"}) if config.crossencoder.enabled else None
    training_counts=None
    if config.task.training_mentions:
        training=read_spans(config.task.training_mentions); training_counts=Counter(str(code) for code in training["gold_code"].dropna())
    rows=[]; raw=[]
    pipeline=EntityLinkingPipeline(generator,reranker=reranker,top_k_candidates=max(config.retrieval.candidate_k))
    for repetition in range(1,config.repetitions+1):
        generated,retrieval_time=_timed(lambda:pipeline._generate(generator,mentions))
        for candidate_k in config.retrieval.candidate_k:
            prepared,candidate_time=_timed(lambda:[copy.deepcopy(list(candidates)[:candidate_k]) for candidates in generated])
            if reranker:
                reranked,cross_time=_timed(lambda:[pipeline._rerank(mention,candidates) for mention,candidates in zip(mentions,prepared)])
            else: reranked,cross_time=prepared,None
            predictions=[[candidate.code for candidate in candidates] for candidates in reranked]; gold=[mention.code for mention in mentions]
            metrics,post_time=_timed(lambda:el_metrics(gold,predictions,DEFAULT_K)); seen=seen_unseen_metrics(gold,predictions,training_counts)
            total=context_time+retrieval_time+candidate_time+(cross_time or 0)+post_time; pairs=sum(len(c) for c in prepared)
            energy={}
            if config.energy.enabled:
                from ..experiment.energy import EnergyTracker
                tracker=EnergyTracker("el_end_to_end",config.energy.measure_power_secs); tracker.start(); energy_start=time.perf_counter(); passes=0
                while passes==0 or time.perf_counter()-energy_start<config.energy.min_duration_seconds:
                    energy_mentions=mentions_from_spans(spans)
                    energy_generated=pipeline._generate(generator,energy_mentions)
                    energy_prepared=[list(candidates)[:candidate_k] for candidates in energy_generated]
                    if reranker: [pipeline._rerank(mention,candidates) for mention,candidates in zip(energy_mentions,energy_prepared)]
                    passes+=1
                energy=tracker.stop(passes,passes*len(mentions),passes*pairs)
                energy.update(energy_normalization(energy.get("total_energy_kwh"),passes*len(mentions),passes*pairs,passes*len(mentions)))
            row={"experiment_id":config.experiment.id,"run_id":f"run-{repetition:03d}-k{candidate_k}","model_id":config.biencoder.id,"compression":compression,"mention_count":len(mentions),"concept_count":len({e.code for e in entries}),"embedding_dimension":config.biencoder.embedding_dimension,"embedding_memory_bytes":embedding_memory_bytes,"index_type":config.retrieval.index_type,"distance_metric":config.retrieval.distance_metric,"index_size_bytes":index_size_bytes,"candidate_k":candidate_k,"context_time_s":context_time,"entity_encoding_time_s":None,"index_build_time_s":index_build_time,"index_serialization_time_s":index_serialization_time,"mention_encoding_time_s":None,"retrieval_time_s":retrieval_time,"biencoder_time_s":None,"index_time_s":retrieval_time,"candidate_preparation_time_s":candidate_time,"crossencoder_time_s":cross_time,"postprocessing_time_s":post_time,"el_total_time_s":total,"candidate_pairs":pairs,"pairs_per_s":None if not cross_time else pairs/cross_time,"accuracy_at_1":metrics["accuracy_at_1"],**{f"recall_at_{k}":metrics.get(f"recall@{k}") for k in DEFAULT_K},"mrr":metrics["mrr"],"candidate_recall":metrics["candidate_recall"],"seen_accuracy":seen["seen_accuracy"],"unseen_accuracy":seen["unseen_accuracy"],"status":"SUCCESS","timing_scope":{"offline_build":"combined terminology encoding and index construction when backend does not expose separate calls","retrieval":"combined mention encoding and index search when backend does not expose separate calls"},"seen_definition":seen["definition"],"el_total_energy_kwh":energy.get("total_energy_kwh"),"el_co2_kg":energy.get("co2_kg"),**{key:energy.get(key) for key in ("joules_per_1000_mentions","joules_per_1000_candidate_pairs","joules_per_1000_queries")}}
            rows.append(row); raw.extend({"run_id":row["run_id"],"filename":mention.filename,"gold_code":mention.code,"candidate_codes":json.dumps(codes)} for mention,codes in zip(mentions,predictions))
    pd.DataFrame([{column:row.get(column) for column in EL_COLUMNS} for row in rows]).to_csv(root/"el_benchmark_long.tsv",sep="\t",index=False)
    pd.DataFrame(raw).to_csv(root/"el_predictions.tsv",sep="\t",index=False)
    summarize_el(root); (root/"el_validation.json").write_text(json.dumps(validation,indent=2)+"\n"); return {"results_dir":str(root),"runs":len(rows)}

def summarize_el(root):
    root=Path(root); frame=pd.read_csv(root/"el_benchmark_long.tsv",sep="\t"); numeric=frame.select_dtypes(include="number").columns
    summary=frame.groupby(["model_id","candidate_k"],dropna=False)[list(numeric)].agg(["mean","median","std"]).reset_index(); path=root/"el_benchmark_summary.tsv"; summary.to_csv(path,sep="\t",index=False); return path
