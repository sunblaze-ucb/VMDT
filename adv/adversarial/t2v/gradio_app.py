import argparse
import json
import ast
import gradio as gr
from adversarial.common.utils import read_jsonl, append_jsonl
from adversarial.t2v.t2v_utils import T2VInstance

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

# Define a function to extract data for a given instance index
def display_instance(instances, index):
    try:
        # Check if index is valid
        if index < 0 or index >= len(instances):
            return "Invalid index", None, None, None, None, None, None, None, None, None, None, None, None, None, None

        # Get the instance
        item = instances[index]

        # Extract relevant data
        id = item["id"]
        task = item["task"]
        clean_property = item["clean_property"]
        target_property = item["target_property"]
        evaluation_question = item["evaluation_question"]
        clean_prompt = item["clean_prompt"]
        target_prompt = item["target_prompt"]
        adversarial_prompt = item["adversarial_prompt"]
        clean_score = item["eval_results"][args.model]["clean_score"]
        clean_xclip = item["eval_results"][args.model]["clean_xclip"]
        adv_score = item["eval_results"][args.model]["adv_score"]
        adv_xclip = item["eval_results"][args.model]["adv_xclip"]
        clean_video_path = item["eval_results"][args.model]["clean_vid_id"]
        adv_video_path = item["eval_results"][args.model]["adv_vid_id"]

        # Return the data
        return id, task, clean_property, target_property, evaluation_question, clean_prompt, target_prompt, adversarial_prompt, clean_score, clean_xclip, adv_score, adv_xclip, clean_video_path, adv_video_path, item

    except Exception as e:
        return f"Error: {str(e)}", None, None, None, None, None, None, None, None, None, None, None, None, None, None

# Gradio app
def create_app(instances):
    # Outputs
    index_display = gr.Textbox(label="Index", interactive=False)
    id_output = gr.Textbox(label="ID", interactive=False)
    task_output = gr.Textbox(label="Task", interactive=False)
    clean_property_output = gr.Textbox(label="Clean Property", interactive=False)
    target_property_output = gr.Textbox(label="Target Property", interactive=False)
    evaluation_question_output = gr.Textbox(label="Evaluation Question", interactive=False)
    clean_prompt_output = gr.Textbox(label="Clean Prompt", interactive=False)
    target_prompt_output = gr.Textbox(label="Target Prompt", interactive=False)
    adversarial_prompt_output = gr.Textbox(label="Adversarial Prompt", interactive=False)
    clean_score_output = gr.Textbox(label="Clean Score", interactive=False)
    clean_xclip_output = gr.Textbox(label="Clean XCLIP", interactive=False)
    adv_score_output = gr.Textbox(label="Adversarial Score", interactive=False)
    adv_xclip_output = gr.Textbox(label="Adversarial XCLIP", interactive=False)
    video_clean_output = gr.Video(label="Clean Video", interactive=False, autoplay=True, loop=True)
    video_adv_output = gr.Video(label="Adversarial Video", interactive=False, autoplay=True, loop=True)
    full_instance_output = gr.Textbox(label="Instance")

    # Function to update outputs based on index
    def update(direction, index):
        new_index = index + direction
        result = display_instance(instances, new_index)
        return (new_index, new_index) + result
    
    # Initial data for instance 0
    initial_data = display_instance(instances, 0)

    # Layout
    with gr.Blocks() as app:
        state = gr.State(value=0)  # Initialize state with index 0
            
        with gr.Row():
            index_display.render()
            id_output.render()
            task_output.render()
            
        with gr.Row():
            clean_property_output.render()
            target_property_output.render()
            evaluation_question_output.render()
            
        with gr.Row():
            clean_prompt_output.render()
            target_prompt_output.render()
            adversarial_prompt_output.render()

        with gr.Row():
            clean_score_output.render()
            clean_xclip_output.render()
            adv_score_output.render()
            adv_xclip_output.render()

        with gr.Row():
            video_clean_output.render()
            video_adv_output.render()

        with gr.Row():
            gr.Button("Previous").click(
                update,
                inputs=[gr.Number(value=-1, visible=False), state],
                outputs=[state, index_display, id_output, task_output, clean_property_output, target_property_output, evaluation_question_output, clean_prompt_output, target_prompt_output, adversarial_prompt_output, clean_score_output, clean_xclip_output, adv_score_output, adv_xclip_output, video_clean_output, video_adv_output, full_instance_output],
            )
            gr.Button("Next").click(
                update,
                inputs=[gr.Number(value=1, visible=False), state],
                outputs=[state, index_display, id_output, task_output, clean_property_output, target_property_output, evaluation_question_output, clean_prompt_output, target_prompt_output, adversarial_prompt_output, clean_score_output, clean_xclip_output, adv_score_output, adv_xclip_output, video_clean_output, video_adv_output, full_instance_output],
            )
        with gr.Row():
            full_instance_output.render()
            
        with gr.Row():
            gr.Button("Submit").click(
                lambda x: append_jsonl(ast.literal_eval(x), args.output_path),
                inputs=[full_instance_output],
                outputs=None,
            )
        
        app.load(
            inputs=[],
            outputs=[state, index_display, id_output, task_output, clean_property_output, target_property_output, evaluation_question_output, clean_prompt_output, target_prompt_output, adversarial_prompt_output, clean_score_output, clean_xclip_output, adv_score_output, adv_xclip_output, video_clean_output, video_adv_output, full_instance_output],
            fn=lambda: (0, 0) + initial_data,
        )

    return app

data = read_jsonl(args.input_path)
app = create_app(data)
app.launch(server_port=8081)