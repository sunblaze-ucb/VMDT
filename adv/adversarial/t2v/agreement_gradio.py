import gradio as gr
import json
from typing import Any
from adversarial.common.utils import append_jsonl

# Load the data
with open('data/t2v/sampled_videos.jsonl', 'r') as f:
    records = [json.loads(line) for line in f]

# State to keep track of current index
index_state = {"idx": 0}

# Output file for labeled results
OUTPUT_PATH = 'data/t2v/sampled_videos_labeled.jsonl'


def get_current_record() -> dict[str, Any]:
    idx = index_state["idx"]
    if idx < len(records):
        return records[idx]
    return {}

def label_and_next(is_correct: bool) -> tuple[str, str, str]:
    idx = index_state["idx"]
    if idx >= len(records):
        return "Done!", "", ""
    record = records[idx].copy()
    record["kyle_score"] = 1.0 if is_correct else 0.0
    append_jsonl(record, OUTPUT_PATH)
    index_state["idx"] += 1
    if index_state["idx"] >= len(records):
        return "Done!", "", ""
    next_record = records[index_state["idx"]]
    return (
        next_record["video_path"],
        next_record["evaluation_question"],
        f"Record {index_state['idx']+1} / {len(records)}"
    )

def start_app():
    first = get_current_record()
    video_path = first.get("video_path", "")
    question = first.get("evaluation_question", "")
    progress = f"Record 1 / {len(records)}"
    with gr.Blocks() as app:
        video = gr.Video(label="Video", value=video_path, interactive=False)
        question_box = gr.Textbox(label="Evaluation Question", value=question, interactive=False)
        progress_box = gr.Textbox(label="Progress", value=progress, interactive=False)
        with gr.Row():
            correct_btn = gr.Button("Correct")
            incorrect_btn = gr.Button("Incorrect")
        
        def on_click(is_correct: bool):
            return label_and_next(is_correct)
        
        correct_btn.click(
            lambda: on_click(True),
            inputs=[],
            outputs=[video, question_box, progress_box],
        )
        incorrect_btn.click(
            lambda: on_click(False),
            inputs=[],
            outputs=[video, question_box, progress_box],
        )
    return app

if __name__ == "__main__":
    app = start_app()
    app.launch(server_port=8082)
