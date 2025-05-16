import json
from collections import defaultdict
import pandas as pd
import argparse

# Load data
def print_results(data_path, model_names=None):
    # Load data
    with open(data_path, "r") as f:
        data = json.load(f)

    # Get model names if none
    if model_names is None:
        # take set of all models for each instance
        model_sets = [set(instance["results"].keys()) for instance in data]
        # find the union of all models
        model_names = list(set.union(*model_sets))

    # only keep instances for which we have results for all models
    data = [instance for instance in data if set(instance["results"].keys()) == set(model_names)]

    # Dictionary to store results
    results_summary = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"count": 0, "correct": 0})))

    # Process data
    for entry in data:
        scenario = entry["scenario_name"]
        task = entry["task_name"]
        for model, result in entry["results"].items():
            results_summary[scenario][task][model]["count"] += 1
            results_summary[scenario][task][model]["correct"] += int(result["correct"])

    # Prepare data for DataFrame
    rows = []
    for scenario, tasks in results_summary.items():
        for task, models in tasks.items():
            for model, stats in models.items():
                avg_score = stats["correct"] / stats["count"]
                rows.append([scenario, task, model, stats["count"], avg_score])

    # Create DataFrame
    df = pd.DataFrame(rows, columns=["Scenario", "Task", "Model", "Instances", "Average Score"])

    # Compute overall averages per scenario and model
    overall_avg = df.groupby(["Scenario", "Model"]).agg({"Instances": "sum", "Average Score": "mean"}).reset_index()

    # Compute overall averages per task and model
    overall_avg_task = df.groupby(["Task", "Model"]).agg({"Instances": "sum", "Average Score": "mean"}).reset_index()

    # Print results
    if model_names:
        df = df[df["Model"].isin(model_names)]
        overall_avg = overall_avg[overall_avg["Model"].isin(model_names)]

    # Save to CSV
    df.to_csv("model_scores_by_task.csv", index=False)
    overall_avg.to_csv("overall_model_scores.csv", index=False)
    overall_avg_task.to_csv("overall_model_scores_by_task.csv", index=False)

    print("Detailed Results:")
    print(df.to_string(index=False))
    print("\nOverall Averages:")
    print(overall_avg.to_string(index=False))
    print("\nOverall Averages per Task:")
    print(overall_avg_task.to_string(index=False))

    # for model with "nova" in it, spatial understanding is bad. Let's print results for the spatial understanding task across each scenario
    nova_models = [model for model in model_names if "nova" in model.lower()]
    print(f"{nova_models=}")
    if nova_models:
        print("\nSpatial Understanding Results for Nova Models:")
        nova_df = df[df["Model"].isin(nova_models)]
        nova_df = nova_df[nova_df["Task"] == "SpatialUnderstanding"]
        print(nova_df.to_string(index=False))

        # print distraction results for nova models
        print("\nOCR Results for Nova Models:")
        nova_df = df[df["Model"].isin(nova_models)]
        nova_df = nova_df[nova_df["Task"] == "OCRPassthrough"]
        print(nova_df.to_string(index=False))
    
    # Print ns - distraction results for all models:
    print("\nNatural Selection - Distraction Results:")
    ns_df = df[df["Scenario"] == "NaturalSelection"]
    distraction_df = df[df["Scenario"] == "Distraction"]
    # subtract the score from ns_df from distraction_df
    ns_df = ns_df[["Model", "Average Score"]].rename(columns={"Average Score": "NS Score"})
    distraction_df = distraction_df[["Model", "Average Score"]].rename(columns={"Average Score": "Distraction Score"})
    ns_df = ns_df.merge(distraction_df, on="Model")
    ns_df["Diff"] = ns_df["Distraction Score"] - ns_df["NS Score"]
    print(ns_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--model_names", type=str, nargs="+", help="List of model names to include in the results. If not provided, all models will be printed.")
    args = parser.parse_args()
    print_results(args.data_path, args.model_names)
