import argparse
import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model_dict = {
    "VideoCrafter2": r"\videocrafter",
    "CogVideoX-5b": r"\cogvideo{5}",
    "OpenSora1.2": r"\opensora",
    "Vchitect2": r"\vchitect",
    "Luma": r"\luma",
    "Pika": r"\pika",
    "NovaReel": r"\novareel",
}
task_dict = {
    "ObjectRecognition": r"\ob",
    "AttributeRecognition": r"\at",
    "ActionRecognition": r"\ac",
    "Counting": r"\cn",
    "SpatialUnderstanding": r"\spa",
}
scenario_dict = {
    "NaturalSelection": "Natural Selection",
    "CoOccurrence": "Co-Occurrence",
    "ComplexBackground": "Complex Background"
}

def get_latex_main_body_table(df):
    """
    Create an overall table that displays models as rows and scenarios as columns, ignoring tasks.
    We include all non-OCR evaluations aggregated per scenario, the OCR column (computed as the mean
    over all OCR rows for a model) and an overall average (across all scenario columns for that model).

    IMPORTANTLY, we average by task first, then average over all scenarios for each model. E.g., average for NS object, NS attribute, etc. then average over all NS, then average over NS and all other scenarios, with tasks treated the same way.

    The evaluation_metric is assumed to be in [0,1] and we format numbers as percentages (×100, one decimal).
    Missing values are rendered as ‘--’.
    """
    ##############################################################################
    # 1. Process Non‐OCR data: compute overall per scenario (ignoring which task).
    #
    # We take all rows where task_name != "OCRPassthrough", and then for each combination
    # of model_name and scenario_name we compute the mean evaluation_metric.
    ##############################################################################
    df_nonocr = df[df["task_name"] != "OCRPassthrough"].copy()
    # Group by model and scenario and task and compute the average score.
    nonocr_group = df_nonocr.groupby(["model_name", "scenario_name", "task_name"])["evaluation_metric"].mean().reset_index()
    # Get task average across all scenarios to (and only to) print.
    ### --- NOTE: THIS IS NOT USED IN THE TABLE --- ###
    task_avg = nonocr_group.groupby(["model_name", "task_name"])["evaluation_metric"].mean().reset_index()
    task_avg_across_models = task_avg.groupby("task_name")["evaluation_metric"].mean()
    print(f"Task average: {task_avg_across_models}")
    ### --- NOTE: THIS IS NOT USED IN THE TABLE --- ###
    # Get scenario average over all already-averaged tasks:
    nonocr_group = nonocr_group.groupby(["model_name", "scenario_name"])["evaluation_metric"].mean().reset_index()
    # Pivot so that each row is a model and each column a scenario.
    pivot_nonocr = nonocr_group.pivot(index="model_name", columns="scenario_name", values="evaluation_metric")
    # Sort the scenario columns alphabetically (or specify a list here if a custom order is desired).
    pivot_nonocr = pivot_nonocr.reindex(sorted(pivot_nonocr.columns), axis=1)
    
    ##############################################################################
    # 2. Process OCR data: aggregate OCR evaluations for each model.
    #
    # For OCR we ignore scenario (or, if the OCR rows have a scenario label with a trailing "OCR"
    # we remove that suffix), and compute the average across OCR evaluations per model.
    ##############################################################################
    df_ocr = df[df["task_name"] == "OCRPassthrough"].copy()
    # In case OCR rows have scenario names like "ComplexBackgroundOCR", remove the trailing "OCR"
    if "scenario_name" in df_ocr.columns:
        df_ocr["scenario_name"] = df_ocr["scenario_name"].str.replace(r"OCR$", "", regex=True)
    # Compute the average OCR score per model.
    # ocr_avg = df_ocr.groupby("model_name")["evaluation_metric"].mean()
    # First, average over each task (there is just one task here so no need). Then, average over all scenarios.
    ocr_scenario_avg = df_ocr.groupby(["model_name", "scenario_name"])["evaluation_metric"].mean().reset_index()
    ocr_avg = ocr_scenario_avg.groupby("model_name")["evaluation_metric"].mean()

    ##############################################################################
    # 3. Combine the two sources:
    #    Build a new DataFrame in which each row corresponds to a model.
    #    Columns: all unique scenario names from non-OCR data, plus an "OCR" column.
    ##############################################################################
    overall_df = pivot_nonocr.copy()
    # Add the OCR column (if a model has no OCR evaluation we simply have missing data).
    overall_df["OCR"] = ocr_avg
    # Compute the overall average for each model across all available (non-NaN) cells.
    overall_df["Overall"] = overall_df.mean(axis=1)
    
    # Optionally, if you want a custom row order for models, define a list here and reindex.
    # For example:
    model_order = ["VideoCrafter2", "CogVideoX-5b", "OpenSora1.2", "Vchitect2", "Luma", "Pika", "NovaReel"]
    # model_order = ["NovaReel"]
    # model_order = ["Pika", "NovaReel"]
    overall_df = overall_df.reindex(model_order)
    
    # Reset index so that model_name becomes a column.
    overall_df = overall_df.reset_index()
    
    ##############################################################################
    # 4. Format numbers as percentages.
    #    Multiply by 100 and print with up to 5 characters (up to two digits before the decimal and one after).
    #    For example, 0.5 becomes "50.0" and 0.087 becomes "8.7". Missing values show as a dash.
    ##############################################################################
        # Determine the numeric columns (all columns except "model_name")
    numeric_cols = overall_df.columns.drop("model_name")
    # Compute the maximum value (in [0,1]) of each numeric column ignoring NaNs.
    col_max = {}
    for col in numeric_cols:
        col_max[col] = overall_df[col].max()

    # Define a formatting function that converts a floating point to a percentage string
    # and bold it if it equals the column maximum.
    def fmt_percent_bold(x, col):
        if pd.notna(x):
            formatted = f"{x*100:5.1f}".strip()
            # Bold if x equals the maximum (tolerance set to a very small value)
            if abs(x - col_max[col]) < 1e-8:
                formatted = r"\textbf{" + formatted + "}"
            return formatted
        else:
            return "--"
    
    # Apply formatting for each numeric column separately so we know which column we are in.
    for col in numeric_cols:
        overall_df[col] = overall_df[col].apply(lambda x: fmt_percent_bold(x, col))
    # def fmt_percent(x):
    #     if pd.notna(x):
    #         return f"{x*100:5.1f}"
    #     else:
    #         return "--"
    
    # # Apply formatting to all columns except the model name.
    # # First, determine the columns that are numeric.
    # numeric_cols = overall_df.columns.drop("model_name")
    # overall_df[numeric_cols] = overall_df[numeric_cols].applymap(fmt_percent)
    
    ##############################################################################
    # 5. Build the LaTeX code.
    #
    # Build a table with columns: Model, then one column per scenario, then OCR, then Overall.
    # We will use the booktabs style with \toprule, \midrule and \bottomrule.
    ##############################################################################
    # Determine column order: “Model” first, then sort scenario names (the ones coming
    # from non-OCR data), then “OCR” and “Overall”. We assume that all columns in overall_df
    # except "model_name", "OCR", and "Overall" are scenario columns.
    all_cols = overall_df.columns.tolist()
    scenario_cols = [col for col in all_cols if col not in ["model_name", "OCR", "Overall"]]
    # (Adjust the scenario order if desired; here we use alphabetical order.)
    # scenario_cols = sorted(scenario_cols)
    scenario_order = ["NaturalSelection", "Distraction", "Misleading", "Counterfactual", "Temporal", "CoOccurrence"]
    scenario_cols = [col for col in scenario_order if col in scenario_cols]
    final_col_order = ["model_name"] + scenario_cols + ["OCR", "Overall"]
    overall_df = overall_df[final_col_order]

    # Now replace model names
    for k, v in model_dict.items():
        overall_df["model_name"] = overall_df["model_name"].replace(k, v)
    
    # Build the LaTeX lines:
    latex_lines = []
    # Define LaTeX tabular column format. 
    # For example, use l for model name and c for the remaining columns, with vertical lines if you wish.
    # col_format = "l" + "c"*(len(final_col_order)-1)
    # latex_lines.append(r"\newcolumntype{Y}{>{\centering\arraybackslash}X}")
    latex_lines.append(r"\resizebox{\textwidth}{!}{")
    col_format = "c|" + "c"*len(scenario_cols) + "c|c"
    latex_lines.append("\\begin{tabular}{" + col_format + "}")
    latex_lines.append("\\toprule")
    
    # Build header row: make each header bold.
    header_cells = []
    header_map = {
        "model_name": "Model",
        "OCR": "OCR",
        "Overall": "Overall"
    }
    for col in final_col_order:
        # Use the header_map for known values; else, use the scenario name as is.
        header = header_map.get(col, col)
        header_cells.append("\\textbf{" + header + "}")
    header_line = " & ".join(header_cells) + " \\\\"
    latex_lines.append(header_line)
    latex_lines.append("\\midrule")
    
    # Add one row per model.
    for idx, row in overall_df.iterrows():
        # The "model_name" cell is printed as is.
        row_cells = []
        for col in final_col_order:
            row_cells.append(str(row[col]))
        latex_lines.append(" & ".join(row_cells) + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")  # end of resizebox
    
    full_latex = "\n".join(latex_lines)
    
    print("LaTeX overall table (models as rows):")
    print(full_latex)
    return full_latex

def get_latex_overall_table(df):
    ##############################################################################
    # 1. Separate OCR rows from visual (non‐OCR) rows.
    ##############################################################################
    df_ocr = df[df["task_name"] == "OCRPassthrough"].copy()
    df_nonocr = df[df["task_name"] != "OCRPassthrough"].copy()
    
    ##############################################################################
    # 2. Process the visual (non‐OCR) data.
    ##############################################################################
    # Pivot the non‐OCR data so each row is (scenario, model) with multiple tasks
    df_pivot = df_nonocr.pivot_table(index=["scenario_name", "model_name"],
                                     columns="task_name",
                                     values="evaluation_metric",
                                     aggfunc="mean").reset_index()
    # Rename tasks using abbreviations.
    task_mapping = {
        "ObjectRecognition": "OB",
        "AttributeRecognition": "AT",
        "ActionRecognition": "AC",
        "Counting": "CN",
        "SpatialUnderstanding": "SP",
    }
    df_pivot = df_pivot.rename(columns=task_mapping)
    
    # Ensure the tasks appear in a fixed order and recombine.
    task_order = ["OB", "AT", "AC", "CN", "SP"]
    df_pivot = df_pivot.reindex(columns=["scenario_name", "model_name"] + task_order)
    
    # Compute the average (Avg) column from the 5 tasks.
    df_pivot["Avg"] = df_pivot[task_order].mean(axis=1)

    # (Optional) Sort scenarios if you prefer a given order.
    # For instance, uncomment and adjust the following if needed:
    scenario_order = ["NaturalSelection", "Distraction", "Misleading", "Counterfactual", "Temporal", "CoOccurrence"]
    df_pivot["scenario_name"] = pd.Categorical(df_pivot["scenario_name"], categories=scenario_order, ordered=True)
    df_pivot = df_pivot.sort_values(["scenario_name", "model_name"]).reset_index(drop=True)

    # (Optional) Sort models if you prefer a given order.
    # For instance, uncomment and adjust the following if needed:
    model_order = ["VideoCrafter2", "CogVideoX-5b", "OpenSora1.2", "Vchitect2", "Luma", "Pika", "NovaReel"]
    # model_order = ["NovaReel"]
    df_pivot["model_name"] = pd.Categorical(df_pivot["model_name"], categories=model_order, ordered=True)
    df_pivot = df_pivot.sort_values(["scenario_name", "model_name"]).reset_index(drop=True)
    
    # Define the final column order for the visual table.
    main_col_order = ["scenario_name", "model_name"] + task_order + ["Avg"]
    
    ##############################################################################
    # 3. Process the OCR table.
    ##############################################################################
    # Pivot OCR data (we assume one evaluation per row for OCR).
    df_ocr_pivot = df_ocr.pivot_table(index=["scenario_name", "model_name"],
                                      columns="task_name",
                                      values="evaluation_metric",
                                      aggfunc="mean").reset_index()
    # Rename the OCR column.
    df_ocr_pivot = df_ocr_pivot.rename(columns={"OCRPassthrough": "OCR"})
    # Remove the trailing "OCR" from scenario names (e.g. "ComplexBackgroundOCR" becomes "ComplexBackground").
    df_ocr_pivot["scenario_name"] = df_ocr_pivot["scenario_name"].str.replace(r"OCR$", "", regex=True)
    df_ocr_pivot = df_ocr_pivot.sort_values(["scenario_name", "model_name"]).reset_index(drop=True)

    # Reorder models for OCR.
    df_ocr_pivot["model_name"] = pd.Categorical(df_ocr_pivot["model_name"], categories=model_order, ordered=True)
    df_ocr_pivot = df_ocr_pivot.sort_values(["scenario_name", "model_name"]).reset_index(drop=True)
    
    # Define the desired column order for OCR.
    ocr_col_order = ["scenario_name", "model_name", "OCR"]
    
    ##############################################################################
    # 4. Format numbers as percentages.
    #    Multiply by 100 and print with up to two digits before the decimal and one after.
    #    For example, 0.5 becomes "50.0" and 0.087 becomes "8.7". Missing values
    #    show as a dash.
    ##############################################################################
    def fmt_percent(x):
        if pd.notna(x):
            return f"{x*100:5.1f}"
        else:
            return "--"
    
    # Create formatted versions.
    df_main_format = df_pivot[main_col_order].copy()
    for col in task_order + ["Avg"]:
        df_main_format[col] = df_main_format[col].apply(fmt_percent)
        
    df_ocr_format = df_ocr_pivot[ocr_col_order].copy()
    df_ocr_format["OCR"] = df_ocr_format["OCR"].apply(fmt_percent)
    
    ##############################################################################
    # 5. Bold the best (maximum) values per scenario for each numeric column.
    #    In the event of ties, all winners are bolded.
    ##############################################################################
    # For the visual table.
    for scenario, idxs in df_pivot.groupby("scenario_name").groups.items():
        for col in task_order + ["Avg"]:
            vals = df_pivot.loc[idxs, col].dropna()
            if not vals.empty:
                best = vals.max()
                for i in idxs:
                    v = df_pivot.loc[i, col]
                    if pd.notna(v) and np.isclose(v, best):
                        df_main_format.loc[i, col] = "\\textbf{" + df_main_format.loc[i, col].strip() + "}"
                        
    # For the OCR table.
    for scenario, idxs in df_ocr_pivot.groupby("scenario_name").groups.items():
        vals = df_ocr_pivot.loc[idxs, "OCR"].dropna()
        if not vals.empty:
            best = vals.max()
            for i in idxs:
                v = df_ocr_pivot.loc[i, "OCR"]
                if pd.notna(v) and np.isclose(v, best):
                    df_ocr_format.loc[i, "OCR"] = "\\textbf{" + df_ocr_format.loc[i, "OCR"].strip() + "}"
    
    ##############################################################################
    # 6. Build the LaTeX code.
    #
    # For the visual table we use a tabular* environment with a total width of \textwidth.
    # The column layout is defined as:
    #   c|c|ccccc|c
    # This means vertical lines appear on the left of the Scenario column, between
    # the Scenario and Model columns, and before the Avg column. No vertical lines appear
    # between the five task columns.
    #
    # We also insert a horizontal rule (\\midrule) after each scenario group.
    ##############################################################################
    main_lines = []
    # Define column format:
    # main_col_format = "@{\\extracolsep{\\fill}}c|c|ccccc|c"
    main_lines.append("\newcolumntype{Y}{>{\centering\arraybackslash}X}")
    main_col_format = "c|c|XXXXX|c"
    main_lines.append("\\begin{tabularx}{\\textwidth}{" + main_col_format + "}")
    # Use booktabs for top rule.
    main_lines.append("\\toprule")
    # Header row.
    headers = ["Scenario", "Model", "OB", "AT", "AC", "CN", "SP", "Avg"]
    # bold all headers
    headers = [f"\\textbf{{{h}}}" for h in headers]
    header_main = " & ".join(headers) + " \\\\"
    main_lines.append(header_main)
    main_lines.append("\\midrule")
    
    # Group rows by scenario and print them.
    for scenario, group in df_main_format.groupby("scenario_name", sort=False):
        group_rows = group.to_dict(orient="records")
        # If more than one row exists in this scenario, use multirow:
        if not group_rows:
            continue
        elif len(group_rows) > 1:
            scenario_cell = "\\multirow{" + str(len(group_rows)) + "}{*}{" + scenario + "}"
            for j, row in enumerate(group_rows):
                if j == 0:
                    line = scenario_cell + " & " + row["model_name"] + " & " \
                           + " & ".join([row[col] for col in task_order]) + " & " + row["Avg"] + " \\\\"
                else:
                    line = " & " + row["model_name"] + " & " \
                           + " & ".join([row[col] for col in task_order]) + " & " + row["Avg"] + " \\\\"
                main_lines.append(line)
        else:
            # Only one row in the scenario group.
            row = group_rows[0]
            line = row["scenario_name"] + " & " + row["model_name"] + " & " \
                   + " & ".join([row[col] for col in task_order]) + " & " + row["Avg"] + " \\\\"
            main_lines.append(line)
        # Add a horizontal line after each scenario group.
        # if not the last scenario, add a midrule.
        if scenario != df_main_format["scenario_name"].iloc[-1]:
            main_lines.append("\\midrule")
    
    main_lines.append("\\bottomrule")
    main_lines.append("\\end{tabularx}")
    main_table_latex = "\n".join(main_lines)
    
    ##############################################################################
    # 7. Build the OCR table.
    #
    # For the OCR table we use a three‐column layout: we want vertical lines after the
    # Scenario and the Model columns. (Column format: |l|l|c|)
    # We also insert a horizontal line after each OCR scenario group.
    ##############################################################################

    # Get the list of scenarios and models – preserving their order.
    scenario_list = df_ocr_format["scenario_name"].unique().tolist()
    model_list    = df_ocr_format["model_name"].unique().tolist()

    # Pivot the DataFrame so that each row corresponds to a model.
    df_pivot = df_ocr_format.pivot(index="model_name", columns="scenario_name", values="OCR")

    # Helper function to extract a numeric value.
    def extract_numeric(cell):
        if isinstance(cell, str):
            m = re.search(r'\\textbf\{([^}]+)\}', cell)
            if m: 
                s_clean = m.group(1)
            else: 
                s_clean = cell
        else:
            s_clean = cell
        try:
            return float(s_clean)
        except (ValueError, TypeError):
            return np.nan

    # Compute the average OCR for each model.
    model_avgs = {}
    for model in model_list:
        numeric_vals = []
        for scenario in scenario_list:
            cell = df_pivot.loc[model, scenario] if scenario in df_pivot.columns else ""
            numeric_vals.append(extract_numeric(cell))
        # Compute the average (using np.nanmean to ignore missing values)
        avg_val = np.nanmean(numeric_vals)
        model_avgs[model] = avg_val

    # Find the maximum average among all models.
    max_avg = max(model_avgs.values())

    # Build the LaTeX lines.
    latex_lines = []

    # Build the table column format: one for model, one for each scenario and one for avg, with vertical rules.
    col_format = "c|" + "c"*len(scenario_list) + "|c"
    latex_lines.append("\\begin{tabular}{" + col_format + "}")
    latex_lines.append("\\toprule")

    # Build the header row (note: only the header text for the average stays bold).
    bold_scenario_list = [f"\\textbf{{{s}}}" for s in scenario_list]
    header_cells = ["\\textbf{Model}"] + bold_scenario_list + ["\\textbf{Avg}"]
    header_line = " & ".join(header_cells) + " \\\\"
    latex_lines.append(header_line)
    latex_lines.append("\\midrule")

    # For each model, output a row with the OCR values and the average (bolding only the max average).
    for model in model_list:
        row_cells = [model]
        numeric_vals = []
        for scenario in scenario_list:
            cell = df_pivot.loc[model, scenario] if scenario in df_pivot.columns else ""
            row_cells.append(cell)
            numeric_vals.append(extract_numeric(cell))
        avg_val = np.nanmean(numeric_vals)
        # Format to one decimal place.
        avg_str = f"{avg_val:.1f}" if not np.isnan(avg_val) else " - "
        # Bold this average only if it is the maximum.
        if avg_val == max_avg:
            avg_str = "\\textbf{" + avg_str + "}"
        row_cells.append(avg_str)
        
        latex_lines.append(" & ".join(row_cells) + " \\\\")
        # Optionally add horizontal lines between rows.
        # latex_lines.append("\\hline")
    
    latex_lines.append("\\bottomrule")

    latex_lines.append("\\end{tabular}")

    ocr_table_latex = "\n".join(latex_lines)

    ##############################################################################
    # 8. Combine both tables.
    ##############################################################################
    full_latex = main_table_latex + "\n\n% --- OCR Table ---\n\n" + ocr_table_latex

    # Now replace model names
    for k, v in model_dict.items():
        full_latex = full_latex.replace(k, v)
    
    print("LaTeX table:")
    print(full_latex)
    return full_latex

def average(evaluation_files):
    # Load and process data
    data = []

    for f_idx, file in enumerate(evaluation_files):
        if os.path.exists(file):
            with open(file, "r") as f:
                entries = json.load(f)
                for entry in entries:
                    task_name = entry.get("task_name", "").replace("CoOccurrence", "").strip()
                    scenario_name = entry.get("scenario_name")
                    if entry.get("temporal"):
                        scenario_name = "Temporal"
                    # if task_name == "OCRPassthrough":
                    #     scenario_name = "OCR"
                    if scenario_name == "AllDistraction":
                        scenario_name = "Distraction"
                    eval_results = entry.get("evaluation_results", {})

                    for model, result in eval_results.items():
                        if "error" in entry["results"][model]:
                            continue
                        metric = result.get("evaluation_metric")
                        parsing_error = result.get("parsing_error", 0)

                        if parsing_error == -1:
                            continue
                        else:
                            if isinstance(metric, dict):
                                metric = np.mean(list(metric.values()))  # Average if dict
                            
                        if isinstance(metric, (int, float)):  # Ensure valid numbers
                            data.append({
                                "prompt": entry.get("prompt"),
                                "video_path": entry.get("results"),  # ["Luma"],
                                "ground_truth": entry.get("ground_truth"),
                                "task_name": task_name,
                                "scenario_name": scenario_name,
                                "model_name": model,
                                "evaluation_metric": metric
                            })

    # Convert to DataFrame
    original_df = pd.DataFrame(data)
    # get_latex_main_body_table(df=original_df)
    # get_latex_overall_table(df=original_df)  # appendix table

    # Create average.csv to save in results/t2v_results/hallucination with two columns: model and average
    average_df = original_df.groupby("model_name")["evaluation_metric"].mean().reset_index()
    average_df = average_df.rename(columns={"evaluation_metric": "average"})
    average_df = average_df.rename(columns={"model_name": "Model"})
    # Sort the DataFrame by average in descending order
    average_df = average_df.sort_values(by="average", ascending=False)
    # Save to CSV
    average_df.to_csv("VMDT/results/t2v_results/hallucination/average.csv", index=False)
    print(f"Average CSV saved to VMDT/results/t2v_results/hallucination/average.csv")
    print(f"Average CSV:\n{average_df}")

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Process JSON files and generate LaTeX tables.")
    args.add_argument(
        "--evaluation_files",
        type=str,
        nargs='+',
        required=True,
        help="List of evaluation JSON files to process. Will combine them into one table."
    )
    args = args.parse_args()
    average(args.evaluation_files)
