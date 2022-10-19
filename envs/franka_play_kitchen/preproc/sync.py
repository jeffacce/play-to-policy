import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from ..utils import subset_each


# make a timestamped frame index dataframe
def make_ts_df(ts, colname) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": ts,
            colname: np.arange(len(ts)),
        }
    )


# merge timestamped dataframes
def merge_ts_dfs(dfs) -> pd.DataFrame:
    result = dfs[0]
    for i in range(1, len(dfs)):
        result = result.merge(dfs[i], how="outer", on="ts")
    return result.set_index("ts").sort_index()


def sync(
    data: Dict,
    fps: float,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> Dict:
    dfs = [make_ts_df(v[0], k) for k, v in data.items()]
    valid_start = max([df.ts.min() for df in dfs])
    valid_end = min([df.ts.max() for df in dfs])
    start = start or valid_start  # if None, use default
    end = end or valid_end
    if start < valid_start:
        logging.warning(
            f"start ({start}) must be after all streams have started ({valid_start}); defaulting to this value."
        )
        start = valid_start
    if end > valid_end:
        logging.warning(
            f"end ({end}) must be before any stream has ended ({valid_end}); defaulting to this value."
        )
        end = valid_end
    frames = int((end - start) * fps)
    clock = np.linspace(start, end, frames, endpoint=False)
    dfs.append(make_ts_df(clock, "clock"))
    df = merge_ts_dfs(dfs)

    valid_idx = ~df["clock"].isna()
    df = df.interpolate("nearest", limit_direction="both")
    df = df[valid_idx].astype(int)

    # subset with synchronized indices
    result = {}
    for k, v in data.items():
        idx = df[k].tolist()
        result[k] = subset_each(v, idx)
    return result
