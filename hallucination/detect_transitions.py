import cv2
import numpy as np
# from scenedetect import open_video, SceneManager
# from scenedetect.detectors import ContentDetector
import time

def detect_scenes_with_frames(video_path, threshold=30.0, min_scene_len=15):
    """
    Detect scenes and display/save frames at the start and end of each scene.

    Parameters:
    video_path : str
        Path to the input video (e.g., 'my_video.mp4').
    threshold : float
        Threshold for content detection. Lower values detect more scenes.
    """
    # Open the video file
    video = open_video(video_path)

    # Create a scene manager and add a content detector
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))

    # Perform scene detection
    scene_manager.detect_scenes(video, show_progress=True)

    # Retrieve scene list with start and end times
    scene_list = scene_manager.get_scene_list()
    # print(f"Detected Scenes: {scene_list}")

    # OpenCV VideoCapture object to access frames
    cap = cv2.VideoCapture(video_path)

    print("Detected Scenes with Boundary Frames:")

    for i, (start_time, end_time) in enumerate(scene_list, start=1):
        start_frame = start_time.get_frames()
        end_frame = end_time.get_frames() - 1
        
        # Read and display the frame at the start of the scene
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, start_img = cap.read()
        if ret:
            # cv2.imshow(f'Scene {i} Start Frame', start_img)
            cv2.imwrite(f'scene_{i}_start.jpg', start_img)  # Optional: Save image

        # Read and display the frame at the end of the scene
        cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
        ret, end_img = cap.read()
        if ret:
            # cv2.imshow(f'Scene {i} End Frame', end_img)
            cv2.imwrite(f'scene_{i}_end.jpg', end_img)  # Optional: Save image
        
        # Display result till a key is pressed
        # cv2.waitKey(0)
        
        # Close windows manually or programmatically
        # cv2.destroyAllWindows()

        print(f"Scene {i}: frames [{start_frame} - {end_frame}]")

    cap.release()

    # get frame numbers for each scene
    scene_list = [(start_time.get_frames(), end_time.get_frames()) for start_time, end_time in scene_list]
    return scene_list

def scene_detection_pyscenedetect(video_path):
    scene_list1 = detect_scenes_with_frames(video_path, threshold=22.5, min_scene_len=10)
    return scene_list1
    # scene_list2 = []  # detect_scenes_with_frames(video_path, threshold=0.0, min_scene_len=10)
    # scene_lists = scene_list1 if len(scene_list1) > len(scene_list2) else scene_list2  # arbitrary choice if both detect scenes. Note for evaluation we will detect the first scene.
    # return scene_lists

import cv2
import base64
import numpy as np
import json
import asyncio
from utils.call_openai import async_call_openai
from utils.data_utils import POSSIBLE_TRANSITIONS

# Assumed to already exist in your file: EVAL_MODEL, MAX_TOKENS, async_call_openai

async def scene_detection(prompt: str, video_path: str, transition: str, num_frames: int = 21) -> dict:
    """
    Extracts num_frames (default 8) evenly spaced frames from video_path,
    sends them to the OpenAI API for scene transition classification, and returns a JSON object containing:
      - scene_ranges: a list of two lists. The first list is the range of video frame indices (1-indexed) before the detected transition.
                      The second list is the range of video frame indices after (including) the transition. 
                      For example, if a transition is detected between frame 20 and frame 21 in a video with 50 frames, 
                      the returned value should be [[1, 20], [21, 50]]. If no transition is detected, then set the first range 
                      as [1, total_frames] and the second as an empty list.
      - found_any_transition: boolean, True if any scene transition is detected.
      - found_desired_transition: boolean, True if the observed transition matches the desired transition (given in the input).
    """
    # Open video and read total frame count.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video at {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("No frames found in video")
    
    # Compute evenly spaced frame indices over the entire video.
    selected_indices = np.linspace(0, total_frames - 1, num_frames, endpoint=True, dtype=int)
    
    # Gather the selected frames and encode as Base64 JPEGs.
    frame_encodings = []
    for i in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue  # skip frame if unable to read
        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        # Save the actual video frame index (plus one to 1-index the frames) along with the encoding.
        frame_encodings.append((i, frame_b64))
    cap.release()
    
    # Build the instructions.
    instructions = (
        "You are a video transition classifier. You are provided with a series of video frames from a video containing "
        "a total of N frames (the frame labels below indicate their index out of the full video, e.g., 'Frame 10/50'). "
        f"The desired transition is \"{transition}\". "
        f"To help you, the prompt upon which the video was generated is \"{prompt}\". However, note that the video does NOT have to follow the prompt. This is why we are doing the classification. You can ignore the content and just focus on if any semblance of the transition in the prompt is present. The prompt should help you dictate the scene list as well."
        "\n\nYour tasks are as follows:\n"
        "1. Provide detailed reasoning steps for your classification. You should refer to frame labels to refer to specific frames.\n"
        "2. Determine whether the desired scene transition occurs among the provided frames. The scene transitions won't necessarily move away from the current scene, but instead reveal some kind of change in the video. "
        "It does not need to be a significant change, but something that is noticable. You can assume that the video is a single continuous shot, where the transition is not necessarily a cut but could be a change in the camera angle, zoom, or other effects. "
        "If the transition is occurring the entire video (e.g., a continuous zoom out), the scene list should always be split based on which frames did not have a new/changed property and which frames did. This is because in every prompt, we either change a property following the transition or add one (from among object, attribute, action, count, spatial relations). "
        # f"These transitions are among the following:\n{', '.join(POSSIBLE_TRANSITIONS)}\n"
        "3. If a transition occurs, determine the frame at which the transition happens and separate the video "
        "into two continuous ranges:\n"
        "      a. The first range contains all frame indices (1-indexed) from the start of the video up to and including the last frame before the transition.\n"
        "      b. The second range contains all frame indices from the transition frame to the end of the video.\n"
        "   For example, if the transition occurs between frame 20 and 21 in a video of 50 frames, the scene ranges would be [[1, 20], [21, 50]].\n"
        "4. If no transition is detected, return the entire video range in the first range and an empty list for the second range.\n"
        "Return your answer as a valid JSON object with exactly the following keys:\n"
        "  - scene_ranges: an array of two arrays. The first element is the range [start, end] for frames before the transition; "
        "     the second element is the range [start, end] for frames after (and including) the transition. If no transition occurred, "
        "     use [1, N] for the first range and an empty array for the second.\n"
        "  - found_any_transition: a boolean indicating if any scene transition is detected among the frames.\n"
        "  - found_desired_transition: a boolean indicating if the detected transition matches the desired transition.\n\n"
        "Do not include any extra keys or commentary. Replace N with the total number of frames in the video."
    )
    
    # Replace the placeholder N with the actual total_frames.
    instructions = instructions.replace("N", str(total_frames))
    
    # Build the messages payload; include instructions and frame images.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instructions}
            ]
        }
    ]
    
    # Append each frame’s label and image to the message.
    for (frame_index, frame_b64) in frame_encodings:
        # Use the full video frame index (adding 1 to convert from 0-indexed to 1-indexed)
        frame_label = f"Frame {frame_index+1}/{total_frames}"
        messages[0]["content"].append({"type": "text", "text": frame_label})
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}", "detail": "low"}
        })
    
    # Define the expected response schema: note that scene_ranges is now a list of two arrays.
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "transition_classification",
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning_steps": {
                        "type": "string",
                        "description": "The reasoning steps taken to determine the transition."
                    },
                    "scene_ranges": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "A list of two arrays. The first array is the range [start, end] of frame indices (1-indexed) before the transition. The second array is the range [start, end] from the transition frame to the end of the video. If no transition, then the first array should be [1, total_frames] and the second should be empty."
                        },
                        "description": (
                            "A list of two arrays. The first array is the range [start, end] of frame indices (1-indexed) "
                            "before the transition. The second array is the range [start, end] from the transition frame to the end of the video. "
                            "If no transition, then the first array should be [1, total_frames] and the second should be empty."
                        )
                    },
                    # "found_any_transition": {
                    #     "type": "boolean",
                    #     "description": "True if any scene transition is detected among the frames."
                    # },
                    "found_desired_transition": {
                        "type": "boolean",
                        "description": "True if the detected transition matches the desired transition provided."
                    }
                },
                # "required": ["reasoning_steps", "scene_ranges", "found_any_transition", "found_desired_transition"],
                "required": ["reasoning_steps", "scene_ranges", "found_desired_transition"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    
    # Call the OpenAI API with the assembled messages.
    response_str = await async_call_openai(
        messages=messages,
        model="gpt-4o-mini",
        system_prompt="Classify the video frame transitions based on the provided instructions.",
        response_format=response_format,
        max_tokens=4096,
        temperature=0.0
    )
    
    try:
        classification = json.loads(response_str)
    except json.JSONDecodeError as e:
        # raise ValueError(f"Failed to parse OpenAI response: {e}\nResponse text: {response_str}")
        print(f"Failed to parse OpenAI response: {e}\nResponse text: {response_str}")
        classification = {"scene_ranges": [[1, total_frames], []], "found_desired_transition": False, "reasoning_steps": "Error: Failed to parse OpenAI response."}
    
    return classification

async def process_video(prompt: str, video_file: str, expected_transition: str) -> tuple:
    """
    Process one video asynchronously.
      - Prints the file and its expected transition.
      - Calls classify_transition_frames and extracts scene_ranges.
      - Determines if a transition was detected based on scene_ranges: if the second range is non-empty, a transition exists.
      - Returns a tuple: (video_file, expected_transition, transition_detected)
    """
    print(f"\nProcessing {video_file} with expected transition: {expected_transition}")
    
    # Call our async classification function.
    classification = await scene_detection(prompt, video_file, expected_transition)
    print(f"OpenAI response: {classification}")
    
    # classification is expected to have keys: scene_ranges, found_any_transition, found_desired_transition.
    # We assume that if a transition exists, the returned scene_ranges is a list of two arrays and the second array is non-empty.
    scene_ranges = classification.get("scene_ranges", [])
    print(f"scene_ranges = {scene_ranges}\n")
    found_desired_transition = classification.get("found_desired_transition", False)
    
    # Determine if a transition has been detected: we want the second range to be non-empty.
    transition_detected = False
    if len(scene_ranges) == 2 and scene_ranges[1] and found_desired_transition:
        transition_detected = True
    
    return video_file, expected_transition, transition_detected

from typing import List
async def main_async(data: List[tuple]):
    # Counters for statistics.
    num_correct_with_transitions, num_correct_without_transitions = 0, 0
    num_with_transitions, num_without_transitions = 0, 0
    bad_videos = {}
    
    # Create a list of async tasks, one per video.
    tasks = []
    for prompt, video_file, expected_transition in data:
        tasks.append(process_video(prompt, video_file, expected_transition))
    
    # Wait for all tasks to complete.
    results = await asyncio.gather(*tasks)
    
    # Process each result.
    for video_file, expected_transition, transition_detected in results:
        # Decide stats based on expected transition.
        if expected_transition != "none":
            num_with_transitions += 1
            if transition_detected:
                print(f"Transition detected for {video_file}")
                num_correct_with_transitions += 1
            else:
                # For videos expected to show a transition, mark as bad if no transition is detected.
                bad_videos[video_file] = expected_transition
        else:
            num_without_transitions += 1
            if not transition_detected:
                print(f"No transition detected for {video_file}")
                num_correct_without_transitions += 1
            else:
                # For videos expected not to show a transition, mark as bad if a transition is detected.
                bad_videos[video_file] = expected_transition
        print(f"Processed {video_file}: detected transition: {transition_detected} (expected: {expected_transition})")
    
    # Compute and print overall accuracy statistics.
    if num_with_transitions:
        print(f"\nTotal transitions detected: {num_correct_with_transitions}/{num_with_transitions} = {num_correct_with_transitions/num_with_transitions*100:.2f}%")
    if num_without_transitions:
        print(f"\nTotal non-transitions detected: {num_correct_without_transitions}/{num_without_transitions} = {num_correct_without_transitions/num_without_transitions*100:.2f}%")
    
    total_videos = num_with_transitions + num_without_transitions
    overall_accuracy = ((num_correct_with_transitions + num_correct_without_transitions) / total_videos * 100) if total_videos > 0 else 0
    print(f"\nTotal accuracy: {overall_accuracy:.2f}%")
    print(f"\nBad videos: {bad_videos}")

if __name__ == "__main__":
    data = {
        "../inference/videos/2025-01-07_14-07-53/c0c92159-4c62-454d-93ae-d7a1c40cb8e9/video.mp4": "pan",

        "../inference/videos/2025-01-07_13-25-28/0f853a62-0ec5-45ae-9a1e-3c02412d1a2b/video.mp4":  "none",
        "../inference/videos/2025-01-07_13-25-28/0fa5cb41-886d-44b6-bb56-73ffa8245dee/video.mp4":  "none",
        "../inference/videos/2025-01-07_13-25-28/3ea65c82-3daf-44f5-a8f0-96242baae45d/video.mp4":  "none",
        "../inference/videos/2025-01-07_13-25-28/7c86d724-c831-4f7d-8cab-601dd3abd9dd/video.mp4":  "none",
        "../inference/videos/2025-01-07_13-25-28/9f736bef-2308-4ada-8e28-dc06ba093a67/video.mp4":  "zoom",

        # "../inference/videos/2025-01-07_13-25-28/10e6630d-c843-43f7-b6f7-e038ceef3d27/video.mp4":  "slight movement; could be either.",
        "../inference/videos/2025-01-07_13-25-28/67cdad94-339c-4757-a472-a4eea47df3db/video.mp4":  "zoom",
        "../inference/videos/2025-01-07_13-25-28/2599a4c5-a1dc-44fa-a31a-0692e3c91900/video.mp4":  "pan",
        "../inference/videos/2025-01-07_13-25-28/7172b22e-382a-4cf4-92ba-792e955b7d0e/video.mp4":  "none",
        "../inference/videos/2025-01-07_13-25-28/12427e4c-79f0-44e2-bfe4-ac57774e4b56/video.mp4":  "tilt",

        "../inference/videos/2025-01-07_13-25-28/64476510-6f41-4b48-baaf-be9099c8d917/video.mp4":  "none",
        "../inference/videos/2025-01-07_13-25-28/a1b65029-b621-423e-b012-1f9d09cb0db2/video.mp4":  "zoom",
        "../inference/videos/2025-01-07_13-25-28/d62c6873-2c62-4527-841b-07633cec2e98/video.mp4":  "none",
        "../inference/videos/2025-01-07_13-25-28/d534c541-3875-44b8-9b59-3e3b09689856/video.mp4":  "none",
        "../inference/videos/2025-01-07_13-25-28/f69ead31-8e30-4c0a-adf3-fd84ee6af239/video.mp4":  "none",

        "../inference/videos/2025-01-07_14-07-53/5a0de203-ca65-49ea-8005-526db303a9f1/video.mp4": "none",
        "../inference/videos/2025-01-07_14-07-53/0a8a5dfe-2ded-4b6a-a703-8585dec24dcb/video.mp4": "none",
        "../inference/videos/2025-01-07_14-07-53/0a67ae1e-b0db-4c72-ab33-dba9358f2f96/video.mp4": "none",  # girl pops up but no camera movement
        "../inference/videos/2025-01-07_14-07-53/0a76a014-7d01-4f3c-ad4d-b0405e90aecd/video.mp4": "none",
        "../inference/videos/2025-01-07_14-07-53/0a871cbf-5fd8-46a3-ad03-5a8972c69f2c/video.mp4": "pan",
        "../inference/videos/2025-01-07_14-07-53/0bbde0da-a447-4caa-acbf-8e43a5bec011/video.mp4": "none",
        "../inference/videos/2025-01-07_14-07-53/0ac39b9e-c27d-409f-87fc-984ac4adbb7f/video.mp4": "pan",
        "../inference/videos/2025-01-07_14-07-53/0b4af317-5d88-455d-a544-7b223e02f3fd/video.mp4": "none",
        "../inference/videos/2025-01-07_14-07-53/0b5787f8-0a25-4bb6-91c6-838fc733f830/video.mp4": "none",
        "../inference/videos/2025-01-07_14-07-53/0b64328d-52ac-4b52-9fd2-f664b5ec53e3/video.mp4": "none",
        "../inference/videos/2025-01-07_14-07-53/0a4171da-9f85-48ae-ac51-1e6378a19a1c/video.mp4": "none",  # slight tilt
        "../inference/videos/2025-01-07_14-07-53/0b9a833a-137f-4aa4-aded-c39317130356/video.mp4": "zoom",
        "../inference/videos/2025-01-07_14-07-53/0aa465a6-9a35-404a-b96b-97e2c8ade37c/video.mp4": "pan",
        "../inference/videos/2025-01-07_14-07-53/0ba63e0e-2d0e-488c-8cb7-97b4c2a83cb4/video.mp4": "none",
        "../inference/videos/2025-01-07_14-07-53/0bad6d53-22de-448f-86f4-92c0f2b3551a/video.mp4": "none",
        "../inference/videos/2025-01-07_14-07-53/0bb6c949-e157-40ed-81d6-e7731478ba5e/video.mp4": "none",
        "../inference/videos/2025-01-07_14-07-53/0bd8d265-98c7-40a5-a9e1-a98b500d4e27/video.mp4": "none",
        "../inference/videos/2025-01-07_14-07-53/0c0c1b7c-670f-40c5-820e-7617bcfba0f9/video.mp4": "zoom",
        "../inference/videos/2025-01-07_14-07-53/0c9e9799-f6ba-4c6c-bb5c-af383a96e43d/video.mp4": "none",
        "../inference/videos/2025-01-07_14-07-53/0c81b977-d9a6-4c58-a83b-f748b85cbc5e/video.mp4": "none",  # camera motion but no real transition
        "../inference/videos/2025-01-07_14-07-53/0c827f83-e307-4848-b358-a329eed9ffcb/video.mp4": "pan"
    }
    data = [
        ("A little girl is in a room with a little boy, upset and yelling because she was called a loser by the boy. They are having a conversation at home. The camera pans to reveal a colorful toy box overflowing with stuffed animals in the corner of the room, contrasting the earlier tension between the siblings. The little girl and boy are still present, but their focus shifts as they notice the toys, momentarily distracted from their argument.", "../inference/videos/2025-01-07_14-07-53/c0c92159-4c62-454d-93ae-d7a1c40cb8e9/video.mp4", "pan"),

        # "../inference/videos/2025-01-07_13-25-28/0f853a62-0ec5-45ae-9a1e-3c02412d1a2b/video.mp4":  "none",
        # "../inference/videos/2025-01-07_13-25-28/0fa5cb41-886d-44b6-bb56-73ffa8245dee/video.mp4":  "none",
        # "../inference/videos/2025-01-07_13-25-28/3ea65c82-3daf-44f5-a8f0-96242baae45d/video.mp4":  "none",
        # "../inference/videos/2025-01-07_13-25-28/7c86d724-c831-4f7d-8cab-601dd3abd9dd/video.mp4":  "none",
        # "../inference/videos/2025-01-07_13-25-28/9f736bef-2308-4ada-8e28-dc06ba093a67/video.mp4":  "zoom",

        # # "../inference/videos/2025-01-07_13-25-28/10e6630d-c843-43f7-b6f7-e038ceef3d27/video.mp4":  "slight movement; could be either.",
        # "../inference/videos/2025-01-07_13-25-28/67cdad94-339c-4757-a472-a4eea47df3db/video.mp4":  "zoom",
        # "../inference/videos/2025-01-07_13-25-28/2599a4c5-a1dc-44fa-a31a-0692e3c91900/video.mp4":  "pan",
        # "../inference/videos/2025-01-07_13-25-28/7172b22e-382a-4cf4-92ba-792e955b7d0e/video.mp4":  "none",
        # "../inference/videos/2025-01-07_13-25-28/12427e4c-79f0-44e2-bfe4-ac57774e4b56/video.mp4":  "tilt",

        # "../inference/videos/2025-01-07_13-25-28/64476510-6f41-4b48-baaf-be9099c8d917/video.mp4":  "none",
        # "../inference/videos/2025-01-07_13-25-28/a1b65029-b621-423e-b012-1f9d09cb0db2/video.mp4":  "zoom",
        # "../inference/videos/2025-01-07_13-25-28/d62c6873-2c62-4527-841b-07633cec2e98/video.mp4":  "none",
        # "../inference/videos/2025-01-07_13-25-28/d534c541-3875-44b8-9b59-3e3b09689856/video.mp4":  "none",
        # "../inference/videos/2025-01-07_13-25-28/f69ead31-8e30-4c0a-adf3-fd84ee6af239/video.mp4":  "none",

        ("A young man with glasses and a surprised expression is sitting on a chair at home, blowing out a single candle that is placed on a cake, while others around him cheer and offer compliments. As the camera zooms in on the cake, a colorful gift box with a ribbon suddenly appears on the table beside it, enhancing the festive atmosphere of the celebration.","../inference/videos/2025-01-07_14-07-53/5a0de203-ca65-49ea-8005-526db303a9f1/video.mp4", "none"),
        ("In an indoor gym filled with weightlifting equipment, one weightlifter, a large man with short hair, is positioned at the center, focused on lifting two heavy weights on a weightlifting machine. He is supported by two male spotters, one on either side, who are assisting him as he exerts himself in this intense fitness session. As the camera zooms in on the weightlifting machine, the large man continues to lift, now with three heavy weights stacked on the machine. The two male spotters remain on either side, providing support as he pushes through the intense workout.","../inference/videos/2025-01-07_14-07-53/0a8a5dfe-2ded-4b6a-a703-8585dec24dcb/video.mp4", "none"),
        ("In a cozy kitchen, a young boy wearing a blue shirt stands at the table, enjoying several fries while licking his fingers. He is closer to the camera than the girl who is approaching him, clearly upset about the fries he is savoring. The kitchen setting adds a warm atmosphere to the sibling dispute unfolding over the table, with the boy positioned to the left of the fries. As the camera pans to the right, the boy in the blue shirt shifts his position, now farther from the camera than the girl who is still upset. The girl moves closer to the table, while the boy remains at the table, munching on several fries. The spatial arrangement changes as the boy is now positioned to the right of the fries, emphasizing the ongoing sibling dispute in the inviting kitchen.","../inference/videos/2025-01-07_14-07-53/0a67ae1e-b0db-4c72-ab33-dba9358f2f96/video.mp4", "none"),  # girl pops up but no camera movement
        ("A young boy is wearing a brown top and demonstrating how to tie a green, dotty necktie around his neck. As the camera zooms in on the boy, it reveals that the necktie now has a shiny finish, glistening in the light.","../inference/videos/2025-01-07_14-07-53/0a76a014-7d01-4f3c-ad4d-b0405e90aecd/video.mp4", "none"),
        ("A woman dressed in a vibrant green dress stands confidently at a podium, delivering an inspiring speech on a stage. Behind her, several wooden barrels are positioned against the backdrop, while two men stand nearby, attentively listening to her words, as a large crowd gathers to hear her message. As the camera zooms in on the podium, the focus shifts to reveal that it is now adorned with beautiful floral arrangements, adding a touch of elegance and color to the stage, just as the woman concludes her impactful speech.","../inference/videos/2025-01-07_14-07-53/0a871cbf-5fd8-46a3-ad03-5a8972c69f2c/video.mp4", "pan"),
        ("In an outdoor snowy environment in Ottawa during the winter of 2008, one person, a male resident, demonstrates an instructional snow clearing technique. He skillfully uses one large shovel to move seven deep snow piles up a hill. As he works, he occasionally loses his balance and falls, adding a humorous touch to the practical demonstration. As the camera pans to the right, the male resident continues his snow clearing demonstration, now using one large shovel to move five deep snow piles up a hill. He is surrounded by the remaining snow, making the task more challenging. He occasionally loses his balance and falls, adding a humorous touch to the practical demonstration.","../inference/videos/2025-01-07_14-07-53/0bbde0da-a447-4caa-acbf-8e43a5bec011/video.mp4", "none"),
        ("A man is seated in a bustling shopping area, strumming his guitar while singing into a microphone, captivating the attention of passersby. As the camera zooms in, the man concludes his song, stands up from his seat, and bows gracefully to the applauding audience.","../inference/videos/2025-01-07_14-07-53/0ac39b9e-c27d-409f-87fc-984ac4adbb7f/video.mp4", "pan"),
        ("In an indoor setting filled with an array of weaving materials, two elderly female demonstrators showcase their skills in various weaving techniques. They skillfully use one white yarn and one colorful rope, expertly weaving them together to create intricate items. In their hands, they hold one small basket, illustrating the final product of their craftsmanship. The instructional scene highlights their methods, as they weave fabric using the yarn and rope, while also demonstrating how to weave a basket with various colors and widths of material. As the camera pans across the indoor setting filled with weaving materials, a third elderly female demonstrator enters the scene, joining the first two. They now skillfully use one white yarn and one colorful rope, expertly weaving them together to create intricate items. In their hands, they hold three small baskets, illustrating the final products of their craftsmanship. The instructional scene highlights their methods, as they weave fabric using the yarn and rope, while also demonstrating how to weave baskets with various colors and widths of material.","../inference/videos/2025-01-07_14-07-53/0b4af317-5d88-455d-a544-7b223e02f3fd/video.mp4", "none"),
        ("A man in a white polo t-shirt and khaki shorts is practicing his golf swing on a lush outdoor golf course. He grips a golf club firmly in his hand as he strikes a golf ball resting on the green grass, intently watching its trajectory as it soars through the air. As the camera pans to the right, the man in the white polo t-shirt and khaki shorts continues his practice. He swings the golf club again, hitting another golf ball that has just rolled into position on the green. Now, there are three golf balls scattered on the grass, ready for his next swing.","../inference/videos/2025-01-07_14-07-53/0b5787f8-0a25-4bb6-91c6-838fc733f830/video.mp4", "none"),
        ("A dark-skinned female is skillfully demonstrating her knife skills, slicing and dicing several vegetables, including potatoes and mushrooms, on a sturdy table. As the camera zooms in, the focus shifts to the knife in her hand, revealing its wooden handle while she continues to prepare the vegetables.","../inference/videos/2025-01-07_14-07-53/0b64328d-52ac-4b52-9fd2-f664b5ec53e3/video.mp4", "none"),
        ("On the front porch of a house, a smiling woman stands, using a hairdryer to blow dry the hair of a young girl who is sitting nearby. The girl appears a bit confused as the warm air flows through her hair, creating a casual and relaxed atmosphere. The hairdryer is held firmly in the woman's hand, positioned closer to the camera than the girl, and to the right of the woman, emphasizing the intimate moment shared between them. As the camera zooms in on the front porch, the woman continues to blow dry the girl's hair, but now the hairdryer has moved to the left of the woman, creating a slightly different angle of the intimate moment. The girl still appears a bit confused as the warm air flows through her hair, maintaining the casual and relaxed atmosphere.","../inference/videos/2025-01-07_14-07-53/0a4171da-9f85-48ae-ac51-1e6378a19a1c/video.mp4", "none"),  # slight tilt
        ("In an indoor ice hockey rink, two teams, consisting of men and children dressed in bright orange uniforms, are actively engaged in a thrilling game of ice hockey. The atmosphere is charged with energy as the lively audience, seated in the stands, watches intently, their cheers echoing throughout the rink. Positioned above the rink, a female commentator narrates the unfolding action, her voice adding to the excitement of the match. Notably, the team is positioned closer to the camera than the audience, who are below the commentator. As the camera zooms out, the scene shifts to reveal the audience now standing on a raised platform, their cheers growing louder and more enthusiastic, enhancing the electrifying atmosphere. The commentator remains in her box, continuing her narration, but now the audience is positioned above her, creating a dynamic shift in the spatial arrangement.","../inference/videos/2025-01-07_14-07-53/0b9a833a-137f-4aa4-aded-c39317130356/video.mp4", "none"),
        ("In a vibrant room adorned with a multitude of paintings and murals, two men stand fervently speaking in front of a wall that showcases an array of artwork, including pieces of religious significance. One man is dressed in a crisp white shirt complemented by a colorful scarf, his passion evident as he engages with the art surrounding him. As the camera pans to capture the vibrant room, a third man enters the scene, joining the two already in front of the wall. The three men now passionately discuss the artwork, their animated conversation filling the space with energy. Each is dressed in colorful attire, reflecting the art that surrounds them.","../inference/videos/2025-01-07_14-07-53/0aa465a6-9a35-404a-b96b-97e2c8ade37c/video.mp4", "pan"),
        ("A man is pacing in his kitchen, visibly overweight, as he holds a plate in one hand. He then picks up an electric drill and begins to screw the refrigerator shut, determined to avoid temptation. The camera zooms in on the refrigerator, revealing it covered in colorful magnets, as the man steps back to admire his handiwork after securing it shut with the drill.","../inference/videos/2025-01-07_14-07-53/0ba63e0e-2d0e-488c-8cb7-97b4c2a83cb4/video.mp4", "none"),
        ("A child, who is male, is talking to a person who is interacting with a sandwich placed on a wooden table, tapping and patting it. As the camera zooms in on the table, the person continues to interact with the sandwich, revealing that the table is adorned with a colorful tablecloth.","../inference/videos/2025-01-07_14-07-53/0bad6d53-22de-448f-86f4-92c0f2b3551a/video.mp4", "none"),
        ("A male in a wetsuit is paddling a canoe back and forth on a serene body of water, possibly a stream or lake. He grips a paddle with a rope attached, skillfully maneuvering the canoe as he enjoys the outdoor setting. The man is positioned to the left of the paddle, which is resting in the canoe. As the camera pans, the male in the wetsuit shifts his position in the canoe, now moving to the right of the paddle. He continues to paddle with the rope-attached paddle, maintaining his focus on the tranquil water around him.","../inference/videos/2025-01-07_14-07-53/0bb6c949-e157-40ed-81d6-e7731478ba5e/video.mp4", "none"),
        ("In a cozy small studio, one person, a male musician, sits at a drum set, fully immersed in his performance. He wears headphones that envelop his ears and a stylish hat that adds to his artistic vibe. The rhythmic beats resonate as he skillfully plays the drum set, creating a captivating musical atmosphere. As the camera pulls out, revealing more of the studio, it becomes clear that there are now three drum sets arranged in a semi-circle around the one person. He continues to play energetically, fully engaged in his performance, while the additional drum sets enhance the vibrant atmosphere of the studio.","../inference/videos/2025-01-07_14-07-53/0bd8d265-98c7-40a5-a9e1-a98b500d4e27/video.mp4", "none"),
        ("A young man stands confidently on stage, beatboxing into a microphone, making electronic and funny sounds with his mouth, while rhythmically moving his hands as if he were playing a drum. As the camera pulls out from the young man on stage, it reveals him transitioning to drumming on a nearby table, creating rhythmic beats with his hands on the surface, while the audience enthusiastically claps along.","../inference/videos/2025-01-07_14-07-53/0c0c1b7c-670f-40c5-820e-7617bcfba0f9/video.mp4", "zoom"),
        ("A young girl is sitting at a small wooden table, demonstrating how to toast several mini marshmallows using a candle that is placed in a jar. As the camera zooms in, the candle is now revealed to be decorated with colorful stickers, adding a playful touch to the scene.","../inference/videos/2025-01-07_14-07-53/0c9e9799-f6ba-4c6c-bb5c-af383a96e43d/video.mp4", "none"),
        ("In a vibrant outdoor setting, a person wearing a white helmet skillfully maneuvers a motocross bike across a rocky stream, where the water rushes rapidly. Positioned to the left of the stream, the rider expertly navigates the challenging terrain, but suddenly, the bike loses traction, causing the rider to tumble into the swirling waters. As the rider splashes into the stream, the camera pans smoothly to the right, revealing the person now standing on the bank, positioned to the right of the stream, carefully evaluating the situation. The motocross bike remains partially submerged in the turbulent water, while the stream continues its swift flow.","../inference/videos/2025-01-07_14-07-53/0c81b977-d9a6-4c58-a83b-f748b85cbc5e/video.mp4", "pan"),
        ("A lively outdoor scene unfolds as a group of contestants, all male construction workers, enthusiastically participate in a food eating competition. Spectators, a mix of adults and children, gather around long tables, eagerly watching the action unfold. The air is filled with excitement as the contestants tackle plates of hot dogs and slices of pizza, creating a vibrant atmosphere. As the camera pans across the bustling parking lot, it captures the same long tables from the competition. The movement reveals that these tables are now adorned with colorful banners, enhancing the festive spirit of the event.","../inference/videos/2025-01-07_14-07-53/0c827f83-e307-4848-b358-a329eed9ffcb/video.mp4", "pan")
    ]
    # take only n from data
    n = len(data)
    data = data[:n]
    # data = dict(list(data.items())[:n])
    # Run the main asynchronous function.
    asyncio.run(main_async(data))

    # num_correct_with_transitions, num_correct_without_transitions = 0, 0
    # num_with_transitions, num_without_transitions = 0, 0
    # bad_videos = {}
    # for video_file, expected_transition in data.items():
    #     print(f"\nProcessing {video_file} with expected transition: {expected_transition}")
    #     # print("Detecting transitions with threshold=30.0:")
    #     # scene_list1 = detect_scenes_with_frames(video_file, threshold=30.0, min_scene_len=12)  # do one call with threshold=30.0, one with threshold=10.0 then say transition is detected if either threshold is met
    #     # print(f"{scene_list1}")
    #     # time.sleep(1)
    #     # print("\nDetecting transitions with threshold=10.0:")
    #     # scene_list2 = detect_scenes_with_frames(video_file, threshold=10.0, min_scene_len=12)
    #     scene_list1 = scene_detection(video_file, expected_transition)
    #     print(f"{scene_list1=}")
    #     print("\n")
    #     scene_lists = scene_list1
    #     if len(scene_lists) > 1:
    #         if expected_transition != "none":
    #             print(f"Transition detected for {video_file}")
    #             num_correct_with_transitions += 1
    #         else:
    #             bad_videos[video_file] = expected_transition
    #     else:
    #         if expected_transition == "none":
    #             print(f"No transition detected for {video_file}")
    #             num_correct_without_transitions += 1
    #     if expected_transition != "none":
    #         num_with_transitions += 1
    #     else:
    #         num_without_transitions += 1
    #     break
    # print(f"\nTotal transitions detected: {num_correct_with_transitions}/{num_with_transitions} = {num_correct_with_transitions/num_with_transitions*100:.2f}%")
    # print(f"\nTotal non-transitions detected: {num_correct_without_transitions}/{num_without_transitions} = {num_correct_without_transitions/num_without_transitions*100:.2f}%")
    # print(f"\nTotal accuracy: {(num_correct_with_transitions + num_correct_without_transitions)/len(data)*100:.2f}%")
    # print(f"\nBad videos: {bad_videos}")
