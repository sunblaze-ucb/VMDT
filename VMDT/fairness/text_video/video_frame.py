import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent

def video_to_frames(input_loc, output_loc,videofilename):
    """Function to extract 5 evenly spaced frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        #os.mkdir(output_loc)
        Path(output_loc).mkdir(exist_ok=True, parents=True)

    except OSError:
        pass
    
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    
    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract (5 evenly spaced frames)
    frames_to_extract = 5
    frame_indices = [int(i * (total_frames - 1) / (frames_to_extract - 1)) for i in range(frames_to_extract)]
    
    #print(f"Total frames in video: {total_frames}")
    #print(f"Extracting frames at indices: {frame_indices}")
    
    count = 0
    current_frame = 0
    
    # Start extracting the frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame in frame_indices:
            # Write the results back to output location
            cv2.imwrite(f"{output_loc}/{videofilename}_{count+1}.jpg", frame)
            count += 1
            
        current_frame += 1
        
        # If we've extracted all required frames, stop
        if count >= frames_to_extract:
            break
    
    # Release the feed
    cap.release()

def extract_videoframe(model, types=['stereotype','finance','education','factual_accuracy','hiring'] ):
    #types=['stereotype','finance','education','factual_accuracy','hiring'] 
    
    for type in types:
        video_names=pd.read_csv(f'{HERE}/model_responses/{model}_{type}.csv')
        output_loc = Path(f'{HERE}/model_responses_videoframe/{model}/{type}')
        output_loc.mkdir(exist_ok=True, parents=True)
        for name in video_names['output'].tolist():
            videofilename=name.split('/')[2]
            if '.mp4' in videofilename:
                videofilename=videofilename[:videofilename.index('.mp4')] 
                input_loc = f'{HERE}/model_responses/{name}' 
            else:
                input_loc = f'{HERE}/model_responses/{name}.mp4' 
            video_to_frames(input_loc, output_loc,videofilename)

#extract_videoframe("Vchitect2",['factual_accuracy'])