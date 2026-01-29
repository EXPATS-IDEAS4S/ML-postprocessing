import numpy as np
import pandas as pd


def check_continuous_timestamps(
    df,
    id_col="storm_id",
    time_col="datetime",
    freq="15min",
):
    """
    Check temporal continuity of trajectories.

    For each storm_id:
      - detect missing timestamps
      - count gaps
      - compute gap lengths
      - compute trajectory duration and length

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    id_col : str
        Storm identifier column.
    time_col : str
        Datetime column.
    freq : str
        Expected temporal frequency (e.g. '15min').

    Returns
    -------
    summary_df : pd.DataFrame
        One row per storm with continuity diagnostics.
    """

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    expected_dt = pd.to_timedelta(freq)

    records = []

    for storm_id, g in df.groupby(id_col):
        g = g.sort_values(time_col)

        times = g[time_col].dropna()

        if len(times) < 2:
            records.append({
                id_col: storm_id,
                "n_points": len(times),
                "start_time": times.min(),
                "end_time": times.max(),
                "duration_min": 0.0,
                "expected_points": len(times),
                "missing_points": 0,
                "n_gaps": 0,
                "gap_lengths_min": [],
                "max_gap_min": 0.0,
                "is_continuous": True,
            })
            continue

        # time differences
        dt = times.diff().dropna()

        # identify gaps
        gap_mask = dt > expected_dt

        gap_lengths = (
            (dt[gap_mask] / pd.Timedelta(minutes=1))
            .values
            .tolist()
        )

        start = times.min()
        end = times.max()

        duration_min = (end - start) / pd.Timedelta(minutes=1)

        expected_points = int(duration_min / (expected_dt / pd.Timedelta(minutes=1))) + 1
        missing_points = expected_points - len(times)

        records.append({
            id_col: storm_id,
            "n_points": len(times),
            "start_time": start,
            "end_time": end,
            "duration_min": duration_min,
            "expected_points": expected_points,
            "missing_points": max(missing_points, 0),
            "n_gaps": int(gap_mask.sum()),
            "gap_lengths_min": gap_lengths,
            "max_gap_min": max(gap_lengths) if gap_lengths else 0.0,
            "is_continuous": gap_mask.sum() == 0,
        })

    summary_df = pd.DataFrame(records)

    return summary_df



#give me a function to split df by region (lat) but look where the majority of crops beloning to the same storm_id are located
def split_traj_by_region(df, lat_column="lat", lat_division=47):
    df = df.copy()
    #get the storm ids
    storm_ids = df["storm_id"].unique()
    region_dfs = {"NORTH": pd.DataFrame(), "SOUTH": pd.DataFrame()}

    for storm_id in storm_ids:
        df_storm = df[df["storm_id"] == storm_id]
        #get the majority region
        n_north = (df_storm[lat_column] >= lat_division).sum()
        n_south = (df_storm[lat_column] < lat_division).sum()

        if n_north >= n_south:
            region_dfs["NORTH"] = pd.concat([region_dfs["NORTH"], df_storm])
        else:
            region_dfs["SOUTH"] = pd.concat([region_dfs["SOUTH"], df_storm])

    return region_dfs



def compute_transitions_and_persistence(
    df,
    label_col="label",
    time_col="datetime",
    lat_col="lat_centre",
    lon_col="lon_centre",
    labels=None,
    n_bootstrap=100,
    ci_level=95,
):

    df = df.copy()

    # --- ensure datetime ---
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col, label_col, lat_col, lon_col])

    # --- label handling ---
    if labels is None:
        labels = np.sort(df[label_col].unique())
    labels = np.asarray(labels)
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # --- containers ---
    T = np.zeros((n, n), dtype=int)
    persistence_durations = {l: [] for l in labels}
    trajectories = []

    # ======================
    # Main pass (point estimates)
    # ======================
    for _, traj in df.groupby("storm_id"):
        traj = traj.sort_values(time_col)

        traj_labels = []
        prev_label = None
        run_start_time = None
        prev_time = None

        for _, row in traj.iterrows():
            curr_label = row[label_col]
            curr_time = row[time_col]

            if curr_label not in label_to_idx:
                continue

            traj_labels.append(label_to_idx[curr_label])

            if prev_label is None:
                run_start_time = curr_time

            elif curr_label != prev_label:
                duration = (prev_time - run_start_time).total_seconds() / 3600
                persistence_durations[prev_label].append(duration)

                T[label_to_idx[prev_label], label_to_idx[curr_label]] += 1
                run_start_time = curr_time

            prev_label = curr_label
            prev_time = curr_time

        if prev_label is not None and run_start_time is not None:
            duration = (prev_time - run_start_time).total_seconds() / 3600
            persistence_durations[prev_label].append(duration)

        if len(traj_labels) > 1:
            trajectories.append(traj_labels)

    # --- transition probabilities ---
    row_sums = T.sum(axis=1, keepdims=True)
    mask_valid = row_sums.squeeze() < 50

    P = np.zeros_like(T, dtype=float)
    np.divide(T, row_sums, where=row_sums != 0, out=P)
    P[mask_valid, :] = np.nan

    # --- persistence probability (point estimate) ---
    dt_hours = 15 / 60
    persistence_prob = np.full(n, np.nan)

    for l, idx in label_to_idx.items():
        durs = np.asarray(persistence_durations[l])
        if len(durs) > 0:
            persistence_prob[idx] = np.mean(durs > dt_hours)

    # ======================
    # Bootstrap over trajectories
    # ======================
    boot_P = []
    boot_persist = []

    for _ in range(n_bootstrap):
        T_b = np.zeros((n, n), dtype=int)
        persist_counts = {i: [] for i in range(n)}

        sampled = np.random.choice(len(trajectories), len(trajectories), replace=True)

        for idx in sampled:
            traj = trajectories[idx]

            run_len = 1
            for t in range(len(traj) - 1):
                T_b[traj[t], traj[t + 1]] += 1

                if traj[t + 1] == traj[t]:
                    run_len += 1
                else:
                    persist_counts[traj[t]].append(run_len)
                    run_len = 1

            persist_counts[traj[-1]].append(run_len)

        # transition matrix
        row_sums_b = T_b.sum(axis=1, keepdims=True)
        P_b = np.zeros_like(T_b, dtype=float)
        np.divide(T_b, row_sums_b, where=row_sums_b != 0, out=P_b)
        P_b[row_sums_b.squeeze() < 50, :] = np.nan
        boot_P.append(P_b)

        # persistence probability
        p_b = np.full(n, np.nan)
        for i in range(n):
            runs = np.asarray(persist_counts[i])
            if len(runs) > 0:
                p_b[i] = np.mean(runs > 1)  # >1 timestep
        boot_persist.append(p_b)

    boot_P = np.asarray(boot_P)
    boot_persist = np.asarray(boot_persist)

    alpha = (100 - ci_level) / 2
    P_ci_low = np.nanpercentile(boot_P, alpha, axis=0)
    P_ci_high = np.nanpercentile(boot_P, 100 - alpha, axis=0)

    persist_ci_low = np.nanpercentile(boot_persist, alpha, axis=0)
    persist_ci_high = np.nanpercentile(boot_persist, 100 - alpha, axis=0)

    # --- entropy ---
    entropy = np.full(n, np.nan)
    entropy[~mask_valid] = -np.sum(
        P[~mask_valid] * np.log(P[~mask_valid] + 1e-12), axis=1
    )

    return {
        "labels": labels,
        "transition_matrix": P,
        "transition_ci_low": P_ci_low,
        "transition_ci_high": P_ci_high,
        "persistence_prob": persistence_prob,
        "persistence_ci_low": persist_ci_low,
        "persistence_ci_high": persist_ci_high,
        "persistence_durations": persistence_durations,
        "entropy": entropy,
    }
