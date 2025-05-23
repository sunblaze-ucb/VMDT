"""Note that everywhere, we want to average over tasks to get the scenario average. Then, we average over scenarios to get the overall average. E.g., average over OB, AC, etc. to get average for NS, then for Misleading, etc. then average those. This is instead of treating each instance equally.

"""
import argparse
import json
import re
import math
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import asyncio
from tqdm.asyncio import tqdm_asyncio

figures_dir = "figures"

surrogate_models = ["InternVideo2Chat8B", "VideoChat2", "VideoLLaVA"]

# Define a mapping from shorthand names to nice names
model_mapping = {
    "InternVideo2Chat8B": "InternVideo 2Chat 8B",
    "VideoChat2": "Video Chat 2",
    "VideoLLaVA": "Video LLaVA",

    "qwen_vl25_3b": "Qwen2.5-VL-3B",
    "qwen_vl25_7b_cv": "Qwen2.5-VL-7B",
    "qwen_vl25_72b": "Qwen2.5-VL-72B",

    "llava_video_7b_fixed": "LLaVA-Video-7B",
    "llava_video_72b_fixed": "LLaVA-Video-72B",

    "videollama2_1_7b_new": "Video-LLaMA-2.1-7B",
    "videollama2_72b": "Video-LLaMA-2-72B",

    "internvl2_5_1b": "InternVL2.5-1B",
    "internvl2_5_2b": "InternVL2.5-2B",
    "internvl2_5_4b": "InternVL2.5-4B",
    "internvl2_5_8b": "InternVL2.5-8B",
    "internvl2_5_26b": "InternVL2.5-26B",
    "internvl2_5_38b": "InternVL2.5-38B",
    "internvl2_5_78b": "InternVL2.5-78B",

    "gpt4o_1fps_lowdetail": "GPT-4o",
    "gpt4omini_1fps_lowdetail": "GPT-4o-mini",

    "nova_lite_fixed": "Nova-Lite",
    "nova_pro": "Nova-Pro",

    "claudesonnet": "Claude-3.5-Sonnet"
}
model_mapping_reverse = {v: k for k, v in model_mapping.items()}

# For Overleaf:
macro_model_mapping = {
    "InternVideo2Chat8B": "InternVideo 2Chat 8B",  # No macro provided; printed as is.
    "VideoChat2": "Video Chat 2",                 # No macro provided; printed as is.
    "VideoLLaVA": "Video LLaVA",                   # No macro provided; printed as is.
    
    "qwen_vl25_3b": r"\qwen{3}",
    "qwen_vl25_7b_cv": r"\qwen{7}",
    "qwen_vl25_72b": r"\qwen{72}",
    
    "llava_video_7b_fixed": r"\llava{7}",
    "llava_video_72b_fixed": r"\llava{72}",
    
    "videollama2_1_7b_new": r"\videollama{7}",
    "videollama2_72b": r"\videollama{72}",
    
    "internvl2_5_1b": r"\internvl{1}",
    "internvl2_5_2b": r"\internvl{2}",
    "internvl2_5_4b": r"\internvl{4}",
    "internvl2_5_8b": r"\internvl{8}",
    "internvl2_5_26b": r"\internvl{26}",
    "internvl2_5_38b": r"\internvl{38}",
    "internvl2_5_78b": r"\internvl{78}",
    
    "gpt4o_1fps_lowdetail": r"\gpt",
    "gpt4omini_1fps_lowdetail": r"\gptmini",
    
    "nova_lite_fixed": r"\novalite",
    "nova_pro": r"\novapro",
    
    "claudesonnet": r"\claude"
}

def create_main_body_latex_table(final_results):

    """
    Create a LaTeX table for V2T evaluation results where:
      - Rows represent models.
      - Columns represent scenarios, along with an average scenario column.
    
    Parameters:
      - final_results: List of dictionaries from the combined V2T JSON results.
    
    Returns:
      - table_str: A LaTeX table string.
    """
    model_order = [
        "InternVideo2Chat8B", "VideoChat2", "VideoLLaVA",
        "qwen_vl25_3b", "qwen_vl25_7b_cv", "qwen_vl25_72b",
        "llava_video_7b_fixed", "llava_video_72b_fixed",
        "videollama2_1_7b_new", "videollama2_72b",
        "internvl2_5_1b", "internvl2_5_2b", "internvl2_5_4b",
        "internvl2_5_8b", "internvl2_5_26b", "internvl2_5_38b", "internvl2_5_78b",
        "gpt4o_1fps_lowdetail", "gpt4omini_1fps_lowdetail",
        "nova_lite_fixed", "nova_pro",
        "claudesonnet"
    ]

    # Create a nested dictionary data[model][scenario] = {"correct": int, "total": int}
    data = {}
    for instance in final_results:
        scenario = instance.get("scenario_name", "Unknown")
        
        for model, result in instance.get("results", {}).items():
            if model not in data:
                data[model] = {}
            if scenario not in data[model]:
                data[model][scenario] = {"correct": 0, "total": 0}
            data[model][scenario]["total"] += 1
            if result.get("correct", False):
                data[model][scenario]["correct"] += 1

    # Define the scenarios we're interested in.
    scenarios = ["NaturalSelection", "Misleading", "Distraction", "Counterfactual"]

    # Start building the table.
    table_str = ""
    table_str += "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}\n"
    table_str += "\\begin{table*}[h]\n"
    table_str += "  \\begin{tabularx}{\\textwidth}{c|" + "Y" * len(scenarios) + "|c}\n"
    table_str += "    \\toprule\n"
    
    header_scenarios = ["\\textbf{" + s + "}" for s in scenarios]
    table_str += "    \\textbf{Model} & " + " & ".join(header_scenarios) + " & \\textbf{Avg} \\\\\n"
    table_str += "    \\midrule\n"

    # Iterate over models in the predefined order.
    for model in model_order:
        if model not in data:
            continue
        total_score = 0.0
        valid_scenarios = 0
        scores = []
        
        for scenario in scenarios:
            if scenario in data[model]:
                cell = data[model][scenario]
                score = 100.0 * (cell["correct"] / cell["total"])
                scores.append(f"{score:.1f}")
                total_score += score
                valid_scenarios += 1
            else:
                scores.append('--')

        # Calculate average score.
        avg_score = (total_score / valid_scenarios) if valid_scenarios > 0 else "--"
        avg_score_str = f"{avg_score:.1f}" if avg_score != "--" else "--"
        
        # Construct the row for this model.
        row = f"    {model} & " + " & ".join(scores) + f" & {avg_score_str} \\\\\n"
        table_str += row

    table_str += "    \\bottomrule\n"
    table_str += "  \\end{tabularx}\n"
    table_str += "  \\caption{Evaluation of V2T models across scenarios.}\n"
    table_str += "  \\label{tab:model-scenario-results}\n"
    table_str += "\\end{table*}\n"

    print(table_str)
    return table_str

def create_v2t_latex_table(final_results):
    """
    Create two LaTeX tables for V2T evaluation results by splitting the scenarios into:
      - Table 1: "NaturalSelection" + "Misleading" (which includes an OCR column).
      - Table 2: "Counterfactual" + "Distraction" (which omits OCR).
    
    Based on final_results (a list of dictionaries), the function aggregates performance
    by scenario, task, and model and uses a macro model mapping to display the model names.
    
    Parameters:
      - final_results: List of dictionaries from the combined V2T JSON results.
    
    Returns:
      - (table1_str, table2_str): A tuple of two LaTeX table strings.
    """

    # Mapping from full task name to abbreviated name.
    task_mapping = {
        "ObjectRecognition": "ob",
        "AttributeRecognition": "at",
        "ActionRecognition": "ac",
        "Counting": "cn",
        "SpatialUnderstanding": "spa",
        "SceneUnderstanding": "sun",
        "OCRPassthrough": "ocr"
    }

    # Placeholder for ordering models.
    # (Note: models not present in model_order are sorted later to the end.)
    model_order = [
        # "InternVideo2Chat8B", "VideoChat2", "VideoLLaVA",
        "qwen_vl25_3b", "qwen_vl25_7b_cv", "qwen_vl25_72b",
        "llava_video_7b", "llava_video_72b",
        "videollama2_1_7b", "videollama2_72b",
        "internvl2_5_1b", "internvl2_5_2b", "internvl2_5_4b",
        "internvl2_5_8b", "internvl2_5_26b", "internvl2_5_38b", "internvl2_5_78b",
        "gpt4omini_1fps_lowdetail", "gpt4o_1fps_lowdetail",
        "nova_lite", "nova_pro",
        "claudesonnet"
    ]

    # Build combined data: data[scenario][model][task_abbr] = {"correct": int, "total": int}.
    data = {}
    for instance in final_results:
        scenario = instance.get("scenario_name", "Unknown")
        task_full = instance.get("task_name", "")
        task_abbr = task_mapping.get(task_full, task_full)

        if scenario not in data:
            data[scenario] = {}

        for model, result in instance.get("results", {}).items():
            # If you are skipping surrogate models, adjust accordingly.
            try:
                if model in surrogate_models:  # if model not in surrogate_models:
                    continue
            except NameError:
                pass

            if model not in data[scenario]:
                data[scenario][model] = {}
            if task_abbr not in data[scenario][model]:
                data[scenario][model][task_abbr] = {"correct": 0, "total": 0}
            data[scenario][model][task_abbr]["total"] += 1
            if result.get("correct", False):
                data[scenario][model][task_abbr]["correct"] += 1

    # Define scenario groups.
    group1_scenarios = ["NaturalSelection", "Misleading"]
    group2_scenarios = ["Distraction", "Counterfactual"]

    # Define tasks order:
    # For table1 we include OCR.
    tasks_order_1 = ["ob", "at", "ac", "cn", "spa", "sun", "ocr"]
    # For table2, we omit OCR.
    tasks_order_2 = ["ob", "at", "ac", "cn", "spa", "sun"]

    # Helper function to build a table given the scenario list and task ordering.
    def build_table(scenarios_list, tasks_order):
        table_str = ""
        table_str += "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}\n"
        table_str += "\\begin{table*}[h]\n"
        table_str += "  \\begin{tabularx}{\\textwidth}{c|c|" + "Y" * len(tasks_order) + "|c}\n"
        table_str += "    \\toprule\n"
        # Build the header.
        header_tasks = []
        for t in tasks_order:
            if t == "ocr":
                header_tasks.append("OCR")
            else:
                header_tasks.append("\\" + t)
        header_tasks = ["\\textbf{" + h + "}" for h in header_tasks]
        table_str += "    \\textbf{Scenario} & \\textbf{Model} & " + " & ".join(header_tasks) + " & \\textbf{Avg} \\\\\n"
        table_str += "    \\midrule\n"

        # Iterate over each scenario in the provided order.
        for scenario in scenarios_list:
            if scenario not in data:
                continue
            # Sort models as in model_order (others sorted to the end).
            models_sorted = sorted(data[scenario].keys(), key=lambda x: model_order.index(x) if x in model_order else float('inf'))
            # Pre-compute max score per task (for bolding) and store the scores.
            max_scores = {task: 0 for task in tasks_order}
            model_scores = {}  # model_scores[model] will be a dictionary with keys for each task and overall score.
            overall_correct = 0  # instead of averaging over task scores, we average over task instances.
            overall_total = 0
            for model in models_sorted:
                disp_model = macro_model_mapping.get(model, model)
                model_scores[model] = {}
                total_score = 0.0
                valid_count = 0
                for task in tasks_order:
                    if task in data[scenario][model]:
                        cell = data[scenario][model][task]
                        score = 100.0 * (cell["correct"] / cell["total"])
                        overall_correct += cell["correct"]
                        overall_total += cell["total"]
                        model_scores[model][task] = score
                        max_scores[task] = max(max_scores[task], score)
                        total_score += score
                        valid_count += 1
                    else:
                        model_scores[model][task] = '--'
                # We want to average over tasks to get the scenario average. Then, we average over scenarios to get the overall average.
                overall = (total_score / valid_count) if valid_count > 0 else "--"
                # overall = 100.0 * (overall_correct / overall_total) if overall_total > 0 else "--"
                model_scores[model]['overall'] = overall

            # Also compute the maximum overall score among models (for bolding).
            overall_max = max([model_scores[m]['overall'] for m in models_sorted if model_scores[m]['overall'] != '--'] or [0])

            first_row = True
            for model in models_sorted:
                disp_model = macro_model_mapping.get(model, model)
                # Use the scenario name only on the first row for that scenario.
                if first_row:
                    row = "    " + scenario + " & " + disp_model + " & "
                    first_row = False
                else:
                    row = "    " + " & " + disp_model + " & "
                task_score_strs = []
                for task in tasks_order:
                    score = model_scores[model][task]
                    if score == '--':
                        task_score_strs.append('--')
                    elif score == max_scores[task]:
                        task_score_strs.append(f"\\textbf{{{score:.1f}}}")
                    else:
                        task_score_strs.append(f"{score:.1f}")
                overall = model_scores[model]['overall']
                if overall == '--':
                    overall_str = "--"
                elif overall == overall_max:
                    overall_str = f"\\textbf{{{overall:.1f}}}"
                else:
                    overall_str = f"{overall:.1f}"
                row += " & ".join(task_score_strs) + " & " + overall_str + " \\\\\n"
                table_str += row
            # if not the last scenario, add a midrule.
            if scenario != scenarios_list[-1]:
                table_str += "    \\midrule\n"
        table_str += "    \\bottomrule\n"
        table_str += "  \\end{tabularx}\n"
        table_str += "  \\caption{Evaluation of V2T models on the selected scenarios.}\n"
        table_str += "  \\label{tab:hallucination-v2t-task-scenario-results}\n"
        table_str += "\\end{table*}\n"
        return table_str

    # New function: build a model-versus-scenario table.
    #
    # This table’s rows are the different models (displayed using their LaTeX macro names)
    # and the columns are the scenarios from the scenarios_list parameter.
    # For each model-scenario pair we compute an average score over all tasks performed in that scenario.
    # An additional "Avg" column shows, for each model, the average performance over all scenarios.
    # Similar to build_table, the best (maximum) score in each column is bolded.
    def build_model_scenario_table(scenarios_list):
        table_str = ""
        table_str += "\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}\n"
        table_str += "\\begin{table*}[h]\n"
        num_cols = len(scenarios_list) + 1  # one extra column for the overall average.
        table_str += "  \\begin{tabularx}{\\textwidth}{c|" + "Y" * (num_cols - 1) + "|c}\n"
        table_str += "    \\toprule\n"
        # Build header row.
        header_scenarios = ["\\textbf{Scenario}"]  # first column header (can be "Model")
        for scenario in scenarios_list:
            header_scenarios.append("\\textbf{" + scenario + "}")
        header_scenarios.append("\\textbf{Avg}")
        table_str += "    " + " & ".join(header_scenarios) + " \\\\\n"
        table_str += "    \\midrule\n"

        # Obtain all models that appear in any of the given scenarios.
        models_set = set()
        for scenario in scenarios_list:
            if scenario in data:
                models_set.update(data[scenario].keys())
        # Sort models based on model_order; models missing in model_order go at the end.
        models_sorted = sorted(models_set, key=lambda x: model_order.index(x) if x in model_order else float('inf'))
        
        # First, compute the average score for each model in each scenario.
        # For each cell in the table, we compute the overall average for that scenario as:
        #    avg = (sum_{task in scenario} (100 * correct/total)) / (number of tasks)
        model_scores = {}  # model_scores[model][scenario] = average score (or '--')
        scenario_max = {scenario: 0 for scenario in scenarios_list}
        for model in models_sorted:
            model_scores[model] = {}
            for scenario in scenarios_list:
                if scenario in data and model in data[scenario]:
                    total_score = 0.0
                    valid_count = 0
                    # Iterate over all tasks available for this model in this scenario.
                    for task, cell in data[scenario][model].items():
                        score = 100.0 * (cell["correct"] / cell["total"])
                        total_score += score
                        valid_count += 1
                    if valid_count > 0:
                        avg_score = total_score / valid_count
                    else:
                        avg_score = "--"
                else:
                    avg_score = "--"
                model_scores[model][scenario] = avg_score
                if avg_score != "--":
                    scenario_max[scenario] = max(scenario_max[scenario], avg_score)
        
        # Compute overall (across scenarios) average per model.
        overall_max = 0
        for model in models_sorted:
            total_model = 0.0
            count_model = 0
            for scenario in scenarios_list:
                score = model_scores[model][scenario]
                if score != "--":
                    total_model += score
                    count_model += 1
            overall = (total_model / count_model) if count_model > 0 else "--"
            model_scores[model]["overall"] = overall
            if overall != "--":
                overall_max = max(overall_max, overall)
        
        # Build table rows.
        for model in models_sorted:
            # Use LaTeX macro name for display if available.
            disp_model = macro_model_mapping.get(model, model)
            row = "    " + disp_model
            total_model = model_scores[model]["overall"]
            # Process each scenario score.
            for scenario in scenarios_list:
                score = model_scores[model][scenario]
                if score == "--":
                    cell_str = "--"
                elif score == scenario_max[scenario]:
                    cell_str = f"\\textbf{{{score:.1f}}}"
                else:
                    cell_str = f"{score:.1f}"
                row += " & " + cell_str
            # Process overall column.
            if total_model == "--":
                overall_str = "--"
            elif total_model == overall_max:
                overall_str = f"\\textbf{{{total_model:.1f}}}"
            else:
                overall_str = f"{total_model:.1f}"
            row += " & " + overall_str + " \\\\\n"
            table_str += row
        
        table_str += "    \\bottomrule\n"
        table_str += "  \\end{tabularx}\n"
        table_str += "  \\caption{Model performance averaged over tasks for each scenario.}\n"
        table_str += "  \\label{tab:model-vs-scenario-results}\n"
        table_str += "\\end{table*}\n"
        return table_str

    # Build the two tables.
    table1_str = build_table(group1_scenarios, tasks_order_1)
    table2_str = build_table(group2_scenarios, tasks_order_2)
    # Build the model-versus-scenario table.
    table3_str = build_model_scenario_table(group1_scenarios + group2_scenarios)

    # You can print or return the tables along with the macros.
    # print(latex_macros)  # Print/emit the macro definitions.
    print(table1_str)
    print(table2_str)
    print(table3_str)
    return table1_str, table2_str

def is_matching(instance1, instance2):
    """match on scenario_name", "question", "gt", "answer_choices", "task_name": "ObjectRecognition"""
    return instance1["scenario_name"] == instance2["scenario_name"] and instance1["question"] == instance2["question"] and instance1["gt"] == instance2["gt"] and instance1["answer_choices"] == instance2["answer_choices"] and instance1["task_name"] == instance2["task_name"]

def combine_results(jsons):
    combined_results = []
    for json_file in jsons:
        with open(json_file, "r") as f:
            data = json.load(f)
        for item in data:
            match = False
            for existing in combined_results:
                if is_matching(existing, item):
                    existing["results"].update(item["results"])
                    match = True
                    break
            if not match:
                combined_results.append(item)
    return combined_results

def average(results_file):
    # Use seaborn's colorblind-friendly palette.
    # sns.set_palette("colorblind")
    # sns.set_style("whitegrid")

    with open(results_file, "r") as f:
        model_class_results = {"all": json.load(f)}

    # Get latex table
    # create_main_body_latex_table(model_class_results["all"])
    create_v2t_latex_table(model_class_results["all"])

    # 2. Aggregate performance by task and by scenario.
    # performance_by_task: {task_name: {model: {correct: int, total: int, average: float}}}
    # performance_by_scenario: {scenario_name: {model: {correct: int, total: int, average: float}}}
    performance_by_task = {}
    performance_by_scenario = {}
    performance_by_scenario_task = {}
    
    for mclass in model_class_results:
        for item in model_class_results[mclass]:
            task = item["task_name"]
            scenario = item["scenario_name"]
            for model, result in item["results"].items():
                correct = result["correct"]

                # COMMENT THE BELOW OUT so we don't get confused.
                # # # Update by task.
                # if task not in performance_by_task:
                #     performance_by_task[task] = {}
                # if model not in performance_by_task[task]:
                #     performance_by_task[task][model] = {"correct": 0, "total": 0}
                # performance_by_task[task][model]["total"] += 1
                # if correct:
                #     performance_by_task[task][model]["correct"] += 1
                
                # # Update by scenario.
                # if scenario not in performance_by_scenario:
                #     performance_by_scenario[scenario] = {}
                # if model not in performance_by_scenario[scenario]:
                #     performance_by_scenario[scenario][model] = {"correct": 0, "total": 0}
                # performance_by_scenario[scenario][model]["total"] += 1
                # if correct:
                #     performance_by_scenario[scenario][model]["correct"] += 1

                # Update by scenario and task.
                if scenario not in performance_by_scenario_task:
                    performance_by_scenario_task[scenario] = {}
                if task not in performance_by_scenario_task[scenario]:
                    performance_by_scenario_task[scenario][task] = {}
                if model not in performance_by_scenario_task[scenario][task]:
                    performance_by_scenario_task[scenario][task][model] = {"correct": 0, "total": 0}
                performance_by_scenario_task[scenario][task][model]["total"] += 1
                if correct:
                    performance_by_scenario_task[scenario][task][model]["correct"] += 1

    # Calculate performance by task and by scenario using performance_by_scenario_task.
    # 1. Compute average across tasks.
    # 2. Compute average across scenarios.

    # Compute average across tasks.
    for task in performance_by_task:
        for model in performance_by_task[task]:
            stats = performance_by_task[task][model]
            performance_by_task[task][model]["average"] = stats["correct"] / stats["total"]

    print("Performance by task:")
    for task in performance_by_task:
        print(f"Task: {task}")
        task_avg_over_models = sum([performance_by_task[task][model]["average"] for model in performance_by_task[task]]) / len(performance_by_task[task])
        print(f"  Average accuracy: {task_avg_over_models:.2f}")
        # InternVL2.5-78B accuracy:
        print(f"  InternVL2.5-78B accuracy: {performance_by_task[task]['internvl2_5_78b']['average']:.2f}")
            
    # Compute average across scenarios.
    for scenario in performance_by_scenario:
        for model in performance_by_scenario[scenario]:
            stats = performance_by_scenario[scenario][model]
            performance_by_scenario[scenario][model]["average"] = stats["correct"] / stats["total"]

    print("Performance by scenario:")
    for scenario in performance_by_scenario:
        print(f"Scenario: {scenario}")
        scenario_avg_over_models = sum([performance_by_scenario[scenario][model]["average"] for model in performance_by_scenario[scenario]]) / len(performance_by_scenario[scenario])
        print(f"  Average accuracy: {scenario_avg_over_models:.2f}")
        # InternVL2.5-78B accuracy:
        print(f"  InternVL2.5-78B accuracy: {performance_by_scenario[scenario]['internvl2_5_78b']['average']:.2f}")

    # Compute average across tasks and scenarios.
    for scenario in performance_by_scenario_task:
        for task in performance_by_scenario_task[scenario]:
            for model in performance_by_scenario_task[scenario][task]:
                stats = performance_by_scenario_task[scenario][task][model]
                performance_by_scenario_task[scenario][task][model]["average"] = stats["correct"] / stats["total"]
                print(f"M: {model}, S: {scenario}, T: {task}, A: {performance_by_scenario_task[scenario][task][model]['average']:.3f}")


    # 3. Compute overall stats per model (aggregating across tasks in this example).
    # compute_avg_over_instances = False  # Set to True to compute average over all instances. If False, compute average over all scenarios. We will average over scenarios, so keep False.
    # overall_stats = {}
    # for task in performance_by_task:
    #     for model in performance_by_task[task]:
    #         stats = performance_by_task[task][model]
    #         if model not in overall_stats:
    #             overall_stats[model] = {"correct": 0, "total": 0}
    #         overall_stats[model]["correct"] += stats["correct"]
    #         overall_stats[model]["total"] += stats["total"]
    # for model in overall_stats:
    # if compute_avg_over_instances:
    #     overall_stats[model]["average"] = overall_stats[model]["correct"] / overall_stats[model]["total"]
    # else:
    #     overall_stats[model]["average"] = sum([performance_by_scenario[scenario][model]["average"] for scenario in performance_by_scenario]) / len(performance_by_scenario)
    # IGNORE THE ABOVE.
    # Let's instead put into a df where we can group by model, scenario, and task.
    df_data = []
    for scenario in performance_by_scenario_task:
        for task in performance_by_scenario_task[scenario]:
            for model in performance_by_scenario_task[scenario][task]:
                if model in surrogate_models:
                    continue
                stats = performance_by_scenario_task[scenario][task][model]
                df_data.append({
                    "model": model,
                    "scenario": scenario,
                    "task": task,
                    "average": stats["correct"] / stats["total"]
                })
    df = pd.DataFrame(df_data)
    # Now we can group by model, scenario, and task.
    df_grouped = df.groupby(["model", "scenario", "task"]).mean().reset_index()
    print(f"{df_grouped=}")
    # df_grouped.to_csv("df_grouped.csv", index=False)
    df_grouped2 = df_grouped.groupby(["model", "scenario"])["average"].mean().reset_index()
    print(f"{df_grouped2=}")
    df_grouped2.to_csv("df_grouped2.csv", index=False)
    scenario_avg2 = df_grouped2.groupby(["scenario"])["average"].mean().reset_index()
    print(f"{scenario_avg2=}")
    df_grouped3 = df_grouped.groupby(["model", "task"])["average"].mean().reset_index()
    task_avg3 = df_grouped3.groupby(["task"])["average"].mean().reset_index()
    print(f"{task_avg3=}")
    # quit()
    df_grouped = df.groupby(["task", "scenario"])["average"].mean().reset_index()
    task_avg = df_grouped.groupby(["task"])["average"].mean().reset_index()
    scenario_avg = df_grouped.groupby(["scenario"])["average"].mean().reset_index()
    print(f"{task_avg=}")
    print(f"{scenario_avg=}")
    # quit()
    # Now we can compute the average across tasks for each scenario.
    # We instead want to average over tasks to get the scenario average. Then, we average over scenarios to get the overall average. This is better as it gives equal weight to each task/scenario.
    overall_stats = {}
    scenario_scores = {}
    for scenario in performance_by_scenario_task:
        for task in performance_by_scenario_task[scenario]:
            for model in performance_by_scenario_task[scenario][task]:
                stats = performance_by_scenario_task[scenario][task][model]["average"]
                if scenario_scores.get(scenario) is None:
                    scenario_scores[scenario] = {}
                if scenario_scores[scenario].get(model) is None:
                    scenario_scores[scenario].update({model: {"stats": 0, "total": 0}})
                scenario_scores[scenario][model]["stats"] += stats
                scenario_scores[scenario][model]["total"] += 1
        # Now we have averages over all tasks for each scenario. We can average over scenarios to get the overall average.
        # for model in scenario_scores[scenario]:
        #     stats = scenario_scores[scenario][model]
        #     if model not in overall_stats:
        #         overall_stats[model] = {"scenario_stats": 0, "scenario_total": 0}
        #     overall_stats[model]["scenario_stats"] += stats["stats"]
        #     overall_stats[model]["scenario_total"] += stats["total"]
    # for model in overall_stats:
    #     overall_stats[model]["average"] = overall_stats[model]["scenario_stats"] / overall_stats[model]["scenario_total"]
    for scenario in scenario_scores:
        for model in scenario_scores[scenario]:
            scenario_average = scenario_scores[scenario][model]["stats"] / scenario_scores[scenario][model]["total"]
            if model not in overall_stats:
                overall_stats[model] = {"average": 0, "total": 0}
            overall_stats[model]["average"] += scenario_average
            overall_stats[model]["total"] += 1
    for model in overall_stats:
        overall_stats[model]["average"] /= overall_stats[model]["total"]
        # print(f"Model: {model}, Scenario stats: {stats['stats']}, Scenario total: {stats['total']}, Scenario average: {overall_stats[model]['average']:.4f}, Task average for scenario: {[performance_by_scenario_task[scenario][task][model]['average'] for task in performance_by_scenario_task[scenario]]}")

    # print videochat2 and internvideo2chat8b
    # for model in ["VideoChat2", "InternVideo2Chat8B"]:
    for model in overall_stats:
        if model in surrogate_models:
            continue
        print(f"{model} overall accuracy: {overall_stats[model]['average']:.2f}")
        # print by task
        for task in performance_by_task:
            if model in performance_by_task[task]:
                print(f"  {task} accuracy: {performance_by_task[task][model]['average']:.2f}")
            else:
                print(f"  {task} accuracy: N/A")
        # print by scenario
        for scenario in performance_by_scenario:
            if model in performance_by_scenario[scenario]:
                print(f"  {scenario} accuracy: {performance_by_scenario[scenario][model]['average']:.2f}")
            else:
                print(f"  {scenario} accuracy: N/A")

    # Create average.csv to save in results/v2t_results/hallucination with two columns: model and average
    df = pd.DataFrame(overall_stats)
    df = df.transpose()
    df = df.reset_index()
    # rename 'index' to 'model' and drop 'total'
    df = df.rename(columns={"index": "model"})
    df = df.drop(columns=["total"])
    df.to_csv("VMDT/results/v2t_results/hallucination/average.csv", index=False)
    print(f"Average CSV saved to VMDT/results/v2t_results/hallucination/average.csv")
    print(f"Average CSV:\n{df}")

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Average V2T results.")
    argparse.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the results file.",
    )
    parser = argparse.parse_args()
    average(parser.results_file)
