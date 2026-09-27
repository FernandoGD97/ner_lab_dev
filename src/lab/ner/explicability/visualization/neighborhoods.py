"""Human-readable before/after nearest-neighbor panels."""

from __future__ import annotations

import pandas as pd

from lab.ner.explicability.visualization.layout import SvgFigure, panel_grid, panel_label
from lab.ner.explicability.visualization.style import StyleProfile


def neighborhood_panel(frame: pd.DataFrame, style: StyleProfile, observation_id: str | None = None) -> SvgFigure:
    detail = frame[frame["record_type"] == "neighbor"] if "record_type" in frame else frame
    if detail.empty:
        raise ValueError("No neighbor-detail records.")
    observation_id = observation_id or str(detail["observation_id"].iloc[0])
    detail = detail[detail["observation_id"].astype(str) == observation_id]
    checkpoints = list(dict.fromkeys(detail["checkpoint"].astype(str)))
    selected = checkpoints if len(checkpoints) <= 3 else [checkpoints[0], checkpoints[len(checkpoints)//2], checkpoints[-1]]
    figure = SvgFigure(style)
    boxes = panel_grid(figure, len(selected), columns=len(selected), margins=(35, 45, 25, 30))
    for index, (checkpoint, box) in enumerate(zip(selected, boxes)):
        panel_label(figure, box, index)
        figure.text(box[0] + box[2] / 2, box[1], checkpoint, anchor="middle", weight="bold")
        subset = detail[detail["checkpoint"].astype(str) == checkpoint].sort_values("rank").head(10)
        for row_index, row in enumerate(subset.itertuples()):
            figure.text(box[0] + 8, box[1] + 28 + row_index * 20,
                        f"{int(row.rank):>2}. {row.neighbor_id}")
            figure.text(box[0] + box[2] - 8, box[1] + 28 + row_index * 20,
                        f"{row.similarity:.3f}", anchor="end")
    return figure
