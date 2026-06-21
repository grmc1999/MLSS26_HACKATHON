
import glob
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

task_name_mapping = {
    "cifar10_training" : "cifar10",
    "google-research-identify-contrails-reduce-global-warming" : "identify-contrails",
}

HUMAN_PERFORMANCE = {
    "cifar10": 0.9,
    "identify-contrails": 0.6,
}

lower_the_better_tasks = []

print_labels = {
    "no_retrieval_gpt4" : "GPT-4",
    "sanity_check" : "Baseline",
}

print_task_labels = {
    "cifar10_training" : "cifar10",
    "google-research-identify-contrails-reduce-global-warming" : "identify-contrails",
}


def is_float_in_list(lst):
    return all(isinstance(x, (int, float)) for x in lst if x is not None)


def extract_timestamp_from_dirname(dirname):
    parts = dirname.split("_")
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return dirname


def get_improvement(df, baseline, thresh=None, prefix=""):
    if prefix:
        df[f"{prefix}increase"] = df[[f"{prefix}score", "task"]].apply(
            lambda x: (x[f"{prefix}score"] - baseline[(baseline["task"] == x["task"])]["final_score"].values[0])
            / baseline[(baseline["task"] == x["task"])]["final_score"].values[0]
            if x[f"{prefix}score"] is not None
            else None,
            axis=1,
        )
    else:
        df["increase"] = df[["score", "task"]].apply(
            lambda x: (x["score"] - baseline[(baseline["task"] == x["task"])]["final_score"].values[0])
            / baseline[(baseline["task"] == x["task"])]["final_score"].values[0]
            if x["score"] is not None
            else None,
            axis=1,
        )
    return df


def parse_log(log_folder):
    df = pd.DataFrame()
    for model_name in os.listdir(log_folder):
        model_dir = os.path.join(log_folder, model_name)
        if not os.path.isdir(model_dir):
            continue
        for run_dirname in os.listdir(model_dir):
            run_dir = os.path.join(model_dir, run_dirname)
            if not os.path.isdir(run_dir):
                continue
            agent_log = os.path.join(run_dir, "agent_log")
            if not os.path.exists(agent_log):
                continue
            trace_files = glob.glob(os.path.join(run_dir, "env_log", "traces", "*.json"))
            for trace_file in trace_files:
                with open(trace_file) as f:
                    trace = json.load(f)
                task = trace.get("task", "")
                if task in task_name_mapping:
                    task = task_name_mapping[task]
                score = trace.get("final_score")
                df = pd.concat([df, pd.DataFrame([{
                    "exp": model_name,
                    "task": task,
                    "final_score": score,
                    "run": run_dirname,
                }])], ignore_index=True)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-folder", type=str, default="final_exp_logs")
    parser.add_argument("--output", type=str, default="plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = parse_log(args.log_folder)
    print(df)
    if len(df) > 0:
        df.to_csv(os.path.join(args.output, "results.csv"), index=False)
        print(f"Results saved to {args.output}/results.csv")
    else:
        print("No results found.")
