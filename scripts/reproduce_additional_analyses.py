#!/usr/bin/env python3
"""Reproduce the additional analyses reported for the public dataset.

The script deliberately writes outside the checksummed release.  It does not
attempt to reconstruct unavailable CFG measurements, WURCS chemistry, biological
agent aliases, or GlycoShape ensemble populations.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from glycoaudit.evaluation import evaluate_positive_kfold, holm_adjust, infer_feature_columns
from glycoaudit.evaluation.pu_ranking import _cosine_scores, _preprocess_matrix, _ranks, _fold_metrics

REPS = {
    "WURCS": ("wurcs_features.csv", "tfidf", "wurcs"),
    "SSV": ("ssv_features.csv", "standardize", "ssv"),
    "GCV_4_5": ("gcv_features.csv", "standardize", "gcv_4_5"),
}
SEED = 20260831


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def q(values: pd.Series | np.ndarray, probs=(0.25, 0.5, 0.75)) -> list[float]:
    a = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    return [float(x) for x in np.quantile(a, probs)] if len(a) else [float("nan")] * len(probs)


def smd(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / max(len(a)+len(b)-2, 1))
    return float((np.mean(a)-np.mean(b))/pooled) if pooled else float("nan")


def bootstrap_mean_difference(a: np.ndarray, b: np.ndarray, seed: int, n=10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(a, (n, len(a)), replace=True).mean(1) - rng.choice(b, (n, len(b)), replace=True).mean(1)
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def common_inputs(release: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    evidence = pd.read_csv(release / "evidence_records.csv")
    folds = pd.read_csv(release / "positive_holdout_folds.csv")
    candidates = pd.read_csv(release / "candidate_manifest.csv")
    ids = set(candidates.loc[candidates.included_in_universe.astype(str).str.lower().isin({"true", "1", "yes"}), "glytoucan_id"].astype(str))
    return evidence, folds, candidates, ids


def joinability_audit(release: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    identifier = pd.read_csv(release / "glycan_identifier_map.csv")
    structure = pd.read_csv(release / "glycan_structure_map.csv")
    evidence = pd.read_csv(release / "evidence_records.csv")
    sequences = pd.read_csv(release / "features/wurcs_sequences.csv")
    resolved = identifier.dropna(subset=["glytoucan_id"]).drop_duplicates("glytoucan_id").copy()
    resolved["glytoucan_id"] = resolved.glytoucan_id.astype(str)
    table = resolved[["glytoucan_id"]].merge(structure[["glytoucan_id", "valid_structure", "failure_reason"]], how="left", on="glytoucan_id")
    table["joinable"] = table.valid_structure.fillna(False).astype(bool)
    table["failure_category"] = np.where(table.joinable, "strict_structure_contract_passed", table.failure_reason.fillna("missing_structure_map"))
    counts = evidence.groupby("glytoucan_id").agg(evidence_record_count=("record_id", "count"), canonical_relation_count=("agent_id", "nunique"), unique_age_agent_count=("agent_id", "nunique")).reset_index()
    sequences["wurcs_sequence_length_characters"] = sequences.wurcs.astype(str).str.len()
    table = table.merge(counts, how="left", on="glytoucan_id").merge(sequences[["glytoucan_id", "wurcs_sequence_length_characters"]], how="left", on="glytoucan_id")
    for col in ["evidence_record_count", "canonical_relation_count", "unique_age_agent_count"]: table[col] = table[col].fillna(0).astype(int)
    table.to_csv(out / "glycan_level_joinability_table.csv", index=False)
    rows=[]
    for i, feature in enumerate(["wurcs_sequence_length_characters", "evidence_record_count", "canonical_relation_count", "unique_age_agent_count"]):
        a=table.loc[table.joinable, feature].dropna().to_numpy(float); b=table.loc[~table.joinable, feature].dropna().to_numpy(float)
        if not len(a) or not len(b): continue
        aq=q(a); bq=q(b); test=stats.mannwhitneyu(a,b,alternative="two-sided")
        lo,hi=bootstrap_mean_difference(a,b,SEED+i)
        rows.append({"feature":feature,"n_joinable":len(a),"n_nonjoinable":len(b),"joinable_median":aq[1],"joinable_iqr_lower":aq[0],"joinable_iqr_upper":aq[2],"nonjoinable_median":bq[1],"nonjoinable_iqr_lower":bq[0],"nonjoinable_iqr_upper":bq[2],"joinable_mean":float(a.mean()),"joinable_sd":float(a.std(ddof=1)),"nonjoinable_mean":float(b.mean()),"nonjoinable_sd":float(b.std(ddof=1)),"standardized_mean_difference_joinable_minus_nonjoinable":smd(a,b),"bootstrap_mean_difference_ci_lower":lo,"bootstrap_mean_difference_ci_upper":hi,"mann_whitney_p":float(test.pvalue)})
    continuous=pd.DataFrame(rows)
    continuous.to_csv(out / "continuous_feature_comparisons.csv",index=False)
    multiple=continuous[["feature","mann_whitney_p"]].rename(columns={"mann_whitney_p":"raw_p"})
    multiple["holm_adjusted_p"]=holm_adjust(multiple.raw_p.tolist()) if len(multiple) else []
    multiple["family"]="joinability continuous descriptive comparisons"
    multiple.to_csv(out / "multiple_testing_results.csv",index=False)
    pd.DataFrame(columns=["feature","status","reason"]).to_csv(out / "categorical_feature_comparisons.csv",index=False)
    table.loc[~table.joinable].groupby("failure_category").size().reset_index(name="n_resolved_glycans").assign(percent_of_nonjoinable=lambda x:100*x.n_resolved_glycans/x.n_resolved_glycans.sum()).to_csv(out / "failure_category_summary.csv",index=False)
    pd.DataFrame([{ "feature":x, "status":"NOT_ESTIMABLE", "reason":"No validated WURCS chemical parser is bundled in the frozen release; regex-derived chemistry was intentionally not used."} for x in ["parsed_residue_count","parsed_linkage_count","branch_count","terminal_residue_count","monosaccharide_composition","fucosylation","sialylation_linkage_class"]]).to_json(out / "NOT_ESTIMABLE.json",orient="records",indent=2)
    summary={"analysis_universe":"resolved GlyTouCan accessions only; unresolved source identifiers excluded","n_resolved_glycans":int(len(table)),"n_joinable":int(table.joinable.sum()),"n_nonjoinable":int((~table.joinable).sum()),"unresolved_source_identifiers":5,"multiple_testing":"Holm adjustment across four continuous descriptive comparisons","interpretation":"Strict structure availability selects a chemically and/or annotation-distinct subset; observed differences are descriptive selection and curation differences, not causal effects."}
    write_json(out / "analysis_manifest.json",{"dataset":"glycoaudit_data","seed":SEED,"variables":list(continuous.feature),"status":"complete"}); write_json(out / "joinability_selection_summary.json",summary)
    pd.DataFrame([{**summary, **{f"largest_absolute_smd_feature": continuous.iloc[continuous.standardized_mean_difference_joinable_minus_nonjoinable.abs().argmax()].feature if len(continuous) else "NOT_ESTIMABLE"}}]).to_csv(out / "supplementary_table_joinability.csv",index=False)
    return summary


def prepared_frame(release: Path, representation: str, ids: set[str]) -> tuple[pd.DataFrame,list[str],str]:
    file, prep, _ = REPS[representation]
    frame=pd.read_csv(release/file); frame=frame[frame.glytoucan_id.astype(str).isin(ids)].copy()
    return frame, infer_feature_columns(frame), prep


def fold_score_state(frame: pd.DataFrame, columns: list[str], prep: str, train: list[str], test: list[str]):
    frame=frame.sort_values("glytoucan_id").reset_index(drop=True); ids=frame.glytoucan_id.astype(str).to_numpy(); pos={x:i for i,x in enumerate(ids)}
    allidx=np.arange(len(frame)); testidx=np.array([pos[x] for x in test]); trainidx=np.array([pos[x] for x in train])
    matrix=_preprocess_matrix(frame[columns].to_numpy(float),np.setdiff1d(allidx,testidx),prep)
    candidates=np.setdiff1d(allidx,trainidx); cids=ids[candidates]
    return matrix, pos, candidates, cids, testidx


def rank_metrics(matrix, candidates, cids, proto, testidx):
    scores=_cosine_scores(matrix[candidates],proto); ranks=_ranks(scores,cids); rank_by_idx=dict(zip(candidates.tolist(),ranks.tolist())); held=np.asarray([rank_by_idx[int(x)] for x in testidx])
    return scores,ranks,held,_fold_metrics(held,(5,10))


def prototype_stability(release: Path, out: Path) -> dict:
    out.mkdir(parents=True,exist_ok=True); evidence,folds,_,ids=common_inputs(release); positives=evidence[evidence.evidence_status.eq("positive")][["agent_id","glytoucan_id"]].drop_duplicates(); positives=positives[positives.glytoucan_id.astype(str).isin(ids)]
    support=positives.groupby("agent_id").glytoucan_id.nunique().rename("total_known_positives")
    foldrows=[]; perturb=[]; one=[]
    for rep in REPS:
        frame,cols,prep=prepared_frame(release,rep,ids)
        for (agent, fold), group in folds.groupby(["agent_id","fold"],sort=True):
            allpos=sorted(positives.loc[positives.agent_id.eq(agent),"glytoucan_id"].astype(str)); test=sorted(group.glytoucan_id.astype(str)); train=sorted(set(allpos)-set(test))
            foldrows.append({"representation":rep,"agent_id":agent,"fold":fold,"total_known_positives":len(allpos),"n_train_positives":len(train),"n_held_out_positives":len(test),"stability_status":"NOT_ESTIMABLE_ONE_TRAINING_POSITIVE" if len(train)<2 else "ESTIMABLE"})
            matrix,pos,candidates,cids,testidx=fold_score_state(frame,cols,prep,train,test); base=matrix[[pos[x] for x in train]].mean(0); scores,ranks,held,metrics=rank_metrics(matrix,candidates,cids,base,testidx)
            if len(train)<2: continue
            for removed in train:
                kept=[x for x in train if x != removed]; pertproto=matrix[[pos[x] for x in kept]].mean(0); pscores,pranks,pheld,pmetrics=rank_metrics(matrix,candidates,cids,pertproto,testidx)
                top=lambda r,k:set(cids[np.asarray(r)<=k])
                rho=stats.spearmanr(scores,pscores).statistic
                fold_mean_rank_change=float(np.mean(pheld-held))
                perturb.append({"representation":rep,"agent_id":agent,"fold":fold,"removed_training_glycan":removed,"n_training_positives_before":len(train),"n_training_positives_after":len(kept),"removed_glycan_absent_from_perturbed_prototype":True,"score_spearman":float(rho) if np.isfinite(rho) else np.nan,"top5_jaccard":len(top(ranks,5)&top(pranks,5))/len(top(ranks,5)|top(pranks,5)),"top10_jaccard":len(top(ranks,10)&top(pranks,10))/len(top(ranks,10)|top(pranks,10)),"fold_mean_held_out_rank_change":fold_mean_rank_change,"absolute_fold_mean_held_out_rank_change":abs(fold_mean_rank_change),"mrr_change":pmetrics["mrr"]-metrics["mrr"],"recall_at_5_change":pmetrics["recall@5"]-metrics["recall@5"],"prototype_cosine":float(np.dot(base,pertproto)/(np.linalg.norm(base)*np.linalg.norm(pertproto))) if np.linalg.norm(base)*np.linalg.norm(pertproto)>0 else np.nan})
            if len(allpos)==2:
                one.append({"representation":rep,"agent_id":agent,"fold":fold,"training_glycan":train[0],"held_out_glycans":";".join(test),"held_out_ranks":";".join(map(str,held.tolist())),"one_example_prototype":True})
    foldframe=pd.DataFrame(foldrows); perturbframe=pd.DataFrame(perturb); oneframe=pd.DataFrame(one)
    foldframe.to_csv(out/"fold_training_support.csv",index=False); perturbframe.to_csv(out/"prototype_perturbation_per_fold.csv",index=False); oneframe.to_csv(out/"one_example_prototype_audit.csv",index=False)
    agent=perturbframe.groupby(["representation","agent_id"],as_index=False).agg(score_spearman=("score_spearman","mean"),top5_jaccard=("top5_jaccard","mean"),top10_jaccard=("top10_jaccard","mean"),prototype_cosine=("prototype_cosine","mean"),n_perturbations=("removed_training_glycan","count")) if len(perturbframe) else pd.DataFrame()
    agent.to_csv(out/"prototype_stability_per_agent.csv",index=False)
    primary=[]
    for rep in REPS:
        metrics=pd.read_csv(release/"results"/REPS[rep][2]/"metrics_per_agent.csv")
        for stratum, group in metrics.groupby("positive_support_stratum"):
            for metric in ["mrr","recall@5","known_positive_ap"]:
                values=group[metric].to_numpy(float); rng=np.random.default_rng(SEED+len(primary)); samples=rng.choice(values,(2000,len(values)),replace=True).mean(1)
                primary.append({"representation":rep,"support_stratum":stratum,"metric":metric,"n_agents":len(group),"n_folds":int(group.n_folds_evaluated.sum()),"median_train_support":float(group.minimum_train_positives.median()),"median_held_out_support":float(group.n_held_out_positive_evaluations.median()/group.n_folds_evaluated.median()),"agent_level_mean":float(values.mean()),"bootstrap_ci_lower":float(np.quantile(samples,.025)),"bootstrap_ci_upper":float(np.quantile(samples,.975))})
    pd.DataFrame(primary).to_csv(out/"support_strata.csv",index=False)
    threshold=[]
    for rep in REPS:
        for min_support,suffix in [(2,""),(3,"_min3"),(5,"_min5")]:
            j=json.loads((release/"results"/(REPS[rep][2]+suffix)/"aggregate_metrics.json").read_text()); threshold.append({"representation":rep,"minimum_positive_support":min_support,"n_agents":j["n_agents_evaluated"],"n_folds":j["n_agent_folds"],"mrr":j["mrr"],"recall_at_5":j["recall@5"]})
    pd.DataFrame(threshold).to_csv(out/"support_threshold_summary.csv",index=False)
    reference_metrics={}
    for rep, group in perturbframe.groupby("representation", sort=True):
        reference_metrics[rep]={
            "mean_score_spearman":float(group.score_spearman.mean()),
            "mean_top5_jaccard":float(group.top5_jaccard.mean()),
            "mean_prototype_cosine":float(group.prototype_cosine.mean()),
            "mean_absolute_fold_mean_held_out_rank_change":float(group.absolute_fold_mean_held_out_rank_change.mean()),
        }
    summary={"n_agent_folds":int(len(foldframe)/len(REPS)),"one_training_positive_folds":int((foldframe.n_train_positives.eq(1)).sum()/len(REPS)),"percent_one_training_positive_folds":float(100*(foldframe.n_train_positives.eq(1)).mean()),"one_example_agents":int((support.eq(2)).sum()),"rank_change_definition":"For each training-positive deletion, average signed held-out-rank changes within the fold, take the absolute value of that fold mean, then average over perturbations.","reference_metrics":reference_metrics,"interpretation":"Agent bootstrap intervals quantify uncertainty of the equal-weighted across-agent mean; the perturbation audit quantifies within-agent sensitivity to omitted training positives."}
    write_json(out/"prototype_stability_summary.json",summary); return summary


def paired_comparisons(release: Path, out: Path) -> dict:
    out.mkdir(parents=True,exist_ok=True); frames={rep:pd.read_csv(release/"results"/d/"metrics_per_agent.csv").set_index("agent_id") for rep,(_,_,d) in REPS.items()}; rows=[]; reps=[]; rng=np.random.default_rng(SEED)
    for left,right in [("WURCS","SSV"),("WURCS","GCV_4_5"),("SSV","GCV_4_5")]:
        shared=frames[left].index.intersection(frames[right].index)
        for metric in ["mrr","recall@5"]:
            diff=(frames[left].loc[shared,metric]-frames[right].loc[shared,metric]).astype(float); boot=rng.choice(diff.to_numpy(),(10_000,len(diff)),replace=True).mean(1); signs=rng.choice(np.array([-1,1]),(10_000,len(diff))); observed=abs(diff.mean()); p=float((np.sum(np.abs((signs*diff.to_numpy()).mean(1))>=observed)+1)/10001)
            reps.append(pd.DataFrame({"comparison":f"{left} - {right}","metric":metric,"bootstrap_replicate":np.arange(10000),"mean_difference":boot}))
            rows.append({"comparison":f"{left} - {right}","metric":metric,"n_paired_agents":len(diff),"mean_paired_difference":float(diff.mean()),"median_paired_difference":float(diff.median()),"bootstrap_ci_lower":float(np.quantile(boot,.025)),"bootstrap_ci_upper":float(np.quantile(boot,.975)),"raw_sign_flip_p":p,"fraction_agents_favoring_left":float((diff>0).mean()),"fraction_agents_favoring_right":float((diff<0).mean()),"seed":SEED})
    result=pd.DataFrame(rows); result["holm_adjusted_p"]=holm_adjust(result.raw_sign_flip_p.tolist()); result["family"]="direct representation comparison family (3 pairs x 2 metrics)"; result.to_csv(out/"paired_comparison_results.csv",index=False); pd.concat(reps).to_csv(out/"bootstrap_replicates.csv",index=False); write_json(out/"summary.json",{"seed":SEED,"bootstrap_replicates":10000,"test":"two-sided agent-paired sign-flip randomization","holm_family":"six direct representation comparisons"}); return {"n_comparisons":len(result)}


def feasibility_and_identity(release: Path, root: Path) -> None:
    evidence,_,_,ids=common_inputs(release); positives=evidence[evidence.evidence_status.eq("positive") & evidence.glytoucan_id.astype(str).isin(ids)][["agent_id","glytoucan_id"]].drop_duplicates(); scaffold=pd.read_csv(release/"scaffold_folds.csv"); scaffold=scaffold[scaffold.glytoucan_id.astype(str).isin(ids)]; out=root/"scaffold_feasibility"; out.mkdir(parents=True,exist_ok=True)
    pos=positives.merge(scaffold[["glytoucan_id","scaffold_key"]],on="glytoucan_id"); per=pos.groupby("agent_id").agg(n_positive_scaffold_groups=("scaffold_key","nunique"),n_positive_glycans=("glytoucan_id","nunique")).reset_index(); per["nonempty_group_disjoint_train_test_possible"]=per.n_positive_scaffold_groups.ge(2); per.to_csv(out/"agent_scaffold_feasibility.csv",index=False); scaffold.groupby("scaffold_key").glytoucan_id.nunique().reset_index(name="glycan_count").to_csv(out/"scaffold_group_sizes.csv",index=False)
    write_json(out/"feasibility_summary.json",{"n_glycans":len(ids),"n_scaffold_groups":int(scaffold.scaffold_key.nunique()),"n_agents_with_two_or_more_positive_scaffold_groups":int(per.nonempty_group_disjoint_train_test_possible.sum()),"n_agents_shared_with_primary":int(len(per)),"status":"FEASIBILITY_AUDIT_COMPLETE"}); write_json(out/"NOT_ESTIMABLE.json",{"status":"NOT_ESTIMABLE","reason":"A secondary grouped metric was not run automatically: the frozen scaffold manifest supplies a glycan grouping but no released agent-level grouped split manifest, and creating a new one would change the released population."})


def schema_and_ties(release: Path, root: Path) -> None:
    schema={"dataset":"glycoaudit_data","representations":{},"shared_policies":{"missing_values":"Non-finite values are imputed per feature with the fold-fit median (or zero when the fit median is unavailable).","scaling":"SSV and GCV use fold-fit standardization; WURCS uses fold-fit TF-IDF and L2 normalization.","constant_or_zero_variance_columns":"Standard deviation is replaced with 1.0 so the standardized feature is zero.","zero_vector":"Cosine score is 0 when prototype or candidate norm is zero.","tie_policy":"Scores descend; exact ties use ascending GlyTouCan accession as a deterministic secondary key; ranks are ordinal 1-indexed."},"aliases":{"contact_density":{"canonical":"contacts_per_residue","status":"input_column_mapping"},"torsion_diversity":{"canonical":"pairwise_residue_center_distance_cv","status":"input_column_mapping"}}}
    for rep in REPS:
        f,c,p=prepared_frame(release,rep,set(pd.read_csv(release/"candidate_manifest.csv").glytoucan_id.astype(str))); schema["representations"][rep]={"ordered_input_columns":c,"final_dimensionality":len(c),"five_model_means":[x for x in c if not x.endswith("_sd")],"five_model_standard_deviations":[x for x in c if x.endswith("_sd")],"preprocessing":p}
    out=root/"method_contract"; out.mkdir(parents=True,exist_ok=True)
    write_json(out/"feature_schema.json",schema)
    pd.DataFrame([{ "representation":rep,"status":"DOCUMENTED_AND_DETERMINISTIC","primary_sort":"descending cosine score","secondary_key":"ascending glytoucan_id","rank_type":"ordinal","recall_boundary":"ordinal rank <= K"} for rep in REPS]).to_csv(out/"tie_policy_summary.csv",index=False)


def model_coordinate_stability(release: Path, root: Path) -> None:
    """Quantify sensitivity to downloaded model indices, without ensemble claims."""
    evidence, folds, _, ids = common_inputs(release)
    positives=evidence[evidence.evidence_status.eq("positive")][["agent_id","glytoucan_id"]].drop_duplicates()
    positives=positives[positives.glytoucan_id.astype(str).isin(ids)]
    out=root/"model_coordinate_stability"; out.mkdir(parents=True,exist_ok=True)
    pairrows=[]; metricrows=[]
    for rep, perfile in [("SSV","features/ssv_per_model.csv"),("GCV_4_5","features/gcv_4_5_per_model.csv")]:
        per=pd.read_csv(release/perfile); per=per[per.glytoucan_id.astype(str).isin(ids)]
        cols=infer_feature_columns(per)
        for (agent, fold), group in folds.groupby(["agent_id","fold"],sort=True):
            allpos=sorted(positives.loc[positives.agent_id.eq(agent),"glytoucan_id"].astype(str)); test=sorted(group.glytoucan_id.astype(str)); train=sorted(set(allpos)-set(test))
            states={}
            for model, frame in per.groupby("model_id",sort=True):
                matrix,pos,candidates,cids,testidx=fold_score_state(frame,cols,"standardize",train,test); proto=matrix[[pos[x] for x in train]].mean(0); scores,ranks,held,metrics=rank_metrics(matrix,candidates,cids,proto,testidx)
                states[int(model)]=(scores,ranks,cids,held,metrics)
                metricrows.append({"representation":rep,"model_id":int(model),"agent_id":agent,"fold":fold,"mrr":metrics["mrr"],"recall_at_5":metrics["recall@5"],"mean_held_out_rank":float(held.mean())})
            for a,b in combinations(sorted(states),2):
                sa,ra,idsa,ha,_=states[a]; sb,rb,idsb,hb,_=states[b]
                assert np.array_equal(idsa,idsb)
                top=lambda ranks,k:set(idsa[np.asarray(ranks)<=k])
                rho=stats.spearmanr(sa,sb).statistic
                pairrows.append({"representation":rep,"agent_id":agent,"fold":fold,"model_a":a,"model_b":b,"candidate_score_spearman":float(rho) if np.isfinite(rho) else np.nan,"top5_jaccard":len(top(ra,5)&top(rb,5))/len(top(ra,5)|top(rb,5)),"top10_jaccard":len(top(ra,10)&top(rb,10))/len(top(ra,10)|top(rb,10)),"mean_held_out_rank_absolute_change":float(np.mean(np.abs(ha-hb)))})
    pairs=pd.DataFrame(pairrows); metrics=pd.DataFrame(metricrows); pairs.to_csv(out/"pairwise_candidate_stability.csv",index=False); metrics.to_csv(out/"per_model_agent_fold_metrics.csv",index=False)
    agent=metrics.groupby(["representation","agent_id"],as_index=False).agg(mrr_range=("mrr",lambda x:float(x.max()-x.min())),recall_at_5_range=("recall_at_5",lambda x:float(x.max()-x.min())),held_out_rank_range=("mean_held_out_rank",lambda x:float(x.max()-x.min())))
    agent.to_csv(out/"per_agent_model_index_ranges.csv",index=False)
    wide=metrics.groupby(["representation","model_id","agent_id"],as_index=False).mrr.mean().pivot(index="agent_id",columns=["representation","model_id"],values="mrr")
    changes=[]
    for agent,row in wide.iterrows():
        signs=[]
        for model in range(1,6):
            if ("SSV",model) in row.index and ("GCV_4_5",model) in row.index: signs.append(int(np.sign(row[("SSV",model)]-row[("GCV_4_5",model)])))
        changes.append({"agent_id":agent,"representation_ordering_changes_across_model_indices":len(set(signs))>1 if signs else np.nan})
    changes=pd.DataFrame(changes); changes.to_csv(out/"representation_ordering_changes.csv",index=False)
    write_json(out/"summary.json",{"scope":"downloaded GlycoShape level-1 model-index sensitivity; no population weighting or solution-state ensemble interpretation","n_pairwise_model_comparisons":int(len(pairs)),"fraction_agents_with_SSV_vs_GCV_MRR_ordering_change":float(changes.representation_ordering_changes_across_model_indices.mean())})


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--release-dir",type=Path,default=ROOT/"data"); p.add_argument("--outdir",type=Path,default=ROOT/"results"); args=p.parse_args(); release=args.release_dir.resolve(); out=args.outdir.resolve(); out.mkdir(parents=True,exist_ok=True)
    started=time.time(); results={"joinability":joinability_audit(release,out/"joinability_selection_audit"),"prototype_stability":prototype_stability(release,out/"prototype_stability"),"paired_comparisons":paired_comparisons(release,out/"paired_representation_comparisons")}; feasibility_and_identity(release,out); schema_and_ties(release,out); model_coordinate_stability(release,out); results["elapsed_seconds"]=round(time.time()-started,3); write_json(out/"analysis_manifest.json",{"dataset":"glycoaudit_data","seed":SEED,"outputs":sorted(x.name for x in out.iterdir()),"elapsed_seconds":results["elapsed_seconds"]}); print(json.dumps(results,indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
