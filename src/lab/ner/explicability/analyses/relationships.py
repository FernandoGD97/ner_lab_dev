"""Performance/internal-adaptation joins and correlation summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance_relationships(
    performance: pd.DataFrame, internal: pd.DataFrame,
    checkpoint_column: str = "checkpoint",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = performance.merge(internal, on=checkpoint_column, how="inner", validate="many_to_many")
    numeric_performance = [column for column in performance if column != checkpoint_column and pd.api.types.is_numeric_dtype(performance[column])]
    numeric_internal = [column for column in internal if column != checkpoint_column and pd.api.types.is_numeric_dtype(internal[column])]
    rows = []
    for metric in numeric_performance:
        for diagnostic in numeric_internal:
            pair = joined[[metric, diagnostic]].dropna()
            if len(pair) < 3 or pair[metric].nunique() < 2 or pair[diagnostic].nunique() < 2:
                coefficient = np.nan
                warning = "insufficient_variation"
            else:
                left = pair[metric].rank(method="average").to_numpy(dtype=float)
                right = pair[diagnostic].rank(method="average").to_numpy(dtype=float)
                coefficient = float(np.corrcoef(left, right)[0, 1])
                warning = None
            rows.append({"performance_metric": metric, "internal_metric": diagnostic,
                         "correlation_method": "spearman", "correlation": coefficient,
                         "sample_count": len(pair), "warning": warning,
                         "causal_interpretation": False})
    return joined, pd.DataFrame(rows)


def layer_specialization(*tables: pd.DataFrame) -> pd.DataFrame:
    """Outer-join layer/checkpoint diagnostics into the Part 3 adaptation map."""
    usable = [table for table in tables if not table.empty and {"checkpoint", "layer"} <= set(table)]
    if not usable:
        return pd.DataFrame()
    result = usable[0]
    for table in usable[1:]:
        duplicate = [column for column in table if column in result and column not in {"checkpoint", "layer"}]
        result = result.merge(table.drop(columns=duplicate), on=["checkpoint", "layer"], how="outer")
    return result.sort_values(["checkpoint", "layer"], kind="stable").reset_index(drop=True)
