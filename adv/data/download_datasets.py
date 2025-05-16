"""Download the datasets used in the experiments.

The datasets, along with their split and a link to download, are as follows:
- [VATEX](https://eric-xw.github.io/vatex-website/download.html): A large-scale multilingual video description dataset. We use the English version and the public test set, as is done normally for evaluation (e.g., in [Gemini](https://arxiv.org/abs/2312.11805)).

"""

import os
import requests
import zipfile
import pandas as pd
import spacy
import json
from collections import defaultdict
from math import sqrt
from tqdm import tqdm
from utils.cooccurrence import get_co_occurrence


def download_vatex(download_videos=False):
    """Download the VATEX dataset. By default, we only download the captions, not the videos.
    
    Here are the captions: https://eric-xw.github.io/vatex-website/data/vatex_public_test_english_v1.1.json.

    Annotation format from https://eric-xw.github.io/vatex-website/download.html:
    ```
    {
    'videoID': 'YouTubeID_StartTime_EndTime',
    'enCap': 
        [
            'Regular English Caption #1',
            'Regular English Caption #2',
            'Regular English Caption #3',
            'Regular English Caption #4',
            'Regular English Caption #5',
            'Parallel English Caption #1',
            'Parallel English Caption #2',
            'Parallel English Caption #3',
            'Parallel English Caption #4',
            'Parallel English Caption #5'
        ],
    'chCap': 
        [
            'Regular Chinese Caption #1',
            'Regular Chinese Caption #2',
            'Regular Chinese Caption #3',
            'Regular Chinese Caption #4',
            'Regular Chinese Caption #5',
            'Parallel Chinese Caption #1',
            'Parallel Chinese Caption #2',
            'Parallel Chinese Caption #3',
            'Parallel Chinese Caption #4',
            'Parallel Chinese Caption #5'
        ]
    }
    ```
    
    """
    # let's download that json file
    os.makedirs("vatex", exist_ok=True)
    caption_file = "vatex/public_test_english_v1.1.json"
    if os.path.exists(caption_file):
        print(f"File {caption_file} already exists.")
    else:
        url = "https://eric-xw.github.io/vatex-website/data/vatex_public_test_english_v1.1.json"
        r = requests.get(url)
        with open(caption_file, "wb") as f:
            f.write(r.content)
    
    # also download train captions for co-occurrence
    caption_file = "vatex/training_v1.0.json"
    if os.path.exists(caption_file):
        print(f"File {caption_file} already exists.")
    else:
        url = "https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json"
        r = requests.get(url)
        with open(caption_file, "wb") as f:
            f.write(r.content)

    if download_videos:
        raise NotImplementedError("Downloading videos is not yet supported.")
    
    print("Downloaded VATEX dataset.")

def download_activitynet_captions():
    """Download the ActivityNet captions.

    Here is the captions dataset: https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip.
    
    """
    os.makedirs("activitynet", exist_ok=True)
    caption_file = "activitynet/captions.zip"
    if os.path.exists(caption_file):
        print(f"File {caption_file} already exists.")
    else:
        url = "https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip"
        r = requests.get(url)
        with open(caption_file, "wb") as f:
            f.write(r.content)
        # Now unzip the file
        with zipfile.ZipFile(caption_file, "r") as zip_ref:
            zip_ref.extractall("activitynet")
    
    print("Downloaded ActivityNet captions dataset.")

def download_wordnet(letter_bound=8):
    """See https://wordnet.princeton.edu/download/standoff-files. This dataset is for the selection of text for OCR."""
    os.makedirs("wordnet", exist_ok=True)
    link = "https://wordnetcode.princeton.edu/standoff-files/core-wordnet.txt"  # "A semi-automatically compiled list of 5000 "core" word senses in WordNet (approximately the 5000 most frequently used word senses, followed by some manual filtering and adjustment)"
    if os.path.exists("wordnet/core-wordnet.txt"):
        print("File already exists.")
    else:
        r = requests.get(link)
        with open("wordnet/core-wordnet.txt", "wb") as f:
            f.write(r.content)
        words = []
        with open("wordnet/core-wordnet.txt") as f2:
            for line in f2:
                word = line.split(" ")[1]
                word = word[1:-1].split("%")[0]
                words.append(word)
        words = list(set(words))
        words = [w for w in words if len(w) < letter_bound]
        with open("wordnet/core-wordnet-readable.txt", "w") as f:
            for i, word in enumerate(words):
                f.write(f"{word}\n") if i != len(words) - 1 else f.write(word)

    print("Downloaded WordNet.")

def download_pexels():
    """Download the Pexels dataset from https://huggingface.co/datasets/jovianzm/Pexels-400k. This dataset is used for co-occurrence analysis."""
    os.makedirs("pexels", exist_ok=True)
    link = "https://huggingface.co/datasets/jovianzm/Pexels-400k/resolve/main/pexels-400k.parquet"
    if os.path.exists("pexels/pexels-400k.parquet"):
        print("File already exists.")
    else:
        r = requests.get(link)
        with open("pexels/pexels-400k.parquet", "wb") as f:
            f.write(r.content)

    print("Downloaded Pexels dataset.")

    # Now, let's clean the titles and extract the entities with POS tagging.
    import pandas as pd
    df = pd.read_parquet("pexels/pexels-400k.parquet")
    print(df.head())
    # find min and max view count
    # print(df["view_count"].min(), df["view_count"].max())  # is 2, 119 million.
    if os.path.exists("pexels/objects.json"):
        print("File already exists.")
    else:
        # Get titles and remove " · Free Stock Video" which is common to all titles.
        titles = df["title"].str.replace(" · Free Stock Video", "").str.lower().tolist()
        # for tagging, use the spacy english pipeline
        nlp = spacy.load("en_core_web_sm")
        # focus on nouns (objects)
        objects = []  # want a separate list for each title
        for title in titles:
            doc = nlp(title)
            objects.append({title: [token.text for token in doc if token.pos_ == "NOUN"]})
        # Save the objects.
        with open("pexels/objects.json", "w") as f:
            json.dump(objects, f, indent=4)
    
    # Clean the objects list and save co-occurrence frequencies.
    ignore_words = ["video", "footage", "stock", "free", "hd", "4k", "download", "background"]
    with open("pexels/objects.json") as f:
        objects = json.load(f)

    if os.path.exists("pexels/co_occurrence.json"):
        print("File already exists.")
    else:
        # remove ignore words and do co-occurrence analysis
        for obj in objects:
            for title, words in obj.items():
                obj[title] = [word for word in words if word not in ignore_words]
        co_occurrence = get_co_occurrence(objects, "pexels")
        
        # Save the co-occurrences
        with open("pexels/co_occurrence.json", "w") as f:
            json.dump(co_occurrence, f, indent=4)

    print("Generated co-occurrence frequencies.")

def download_clevrer(download_videos=False):
    """Download the CLEVRER dataset (validation set, since test has no answer annotations). By default, we only download the questions, not the videos.

    File is https://data.csail.mit.edu/clevrer/questions/validation.json. Also https://data.csail.mit.edu/clevrer/annotations/validation/annotation_validation.zip. More details are http://clevrer.csail.mit.edu/#Paper.
    From [README.txt](https://data.csail.mit.edu/clevrer/README.txt):
        Questions in CLEVRER include 4 types: descriptive, explanatory, predictive and counterfactual. Descriptive questions are answered by a single word token while the other three question types are multiple choice. Each question in the training and validation set comes with program and answer annotations. The programs are represented by lists of module names arranged in postfix notation. These informations are omitted in the test split. Subtypes of descriptive questions are also included for further diagnostics. Each multiple choice question may include more than one or no correct choices. Structures of the training (validation) and test question files are shown below.
        ```
        questions/train.json:
        [
            {
                scene_index: 0,
                video_filename: 'video_00000.mp4',
                questions: [
                    // Descriptive question
                    {
                        question_id: 0,
                        question: 'What is the shape of the object to collide with the purple object?',
                        question_type: 'descriptive',
                        question_subtype: 'query_shape',
                        program: [...],
                        answer: 'sphere',
                    },
                    ...
                    // Multiple choice question
                    {
                        question_id: 21,
                        question: 'Which event will happen if the cylinder is removed?',
                        question_type: 'counterfactual',
                        program: [...],
                        choices: [
                            {choice_id: 0, choice: 'The blue rubber sphere collides with the cube', program: [...], answer: 'wrong'},
                            ...
                        ]
                    },
                    ...
                ],
            },
            ...
        ]
        ``` 
        There are validation (index 10000 - 14999) videos.
    """
    os.makedirs("clevrer", exist_ok=True)
    question_file = "clevrer/validation.json"
    if os.path.exists(question_file):
        print(f"File {question_file} already exists.")
    else:
        url = "https://data.csail.mit.edu/clevrer/questions/validation.json"
        r = requests.get(url)
        with open(question_file, "wb") as f:
            f.write(r.content)

    annotation_file = "clevrer/annotation_validation.zip"
    if os.path.exists(annotation_file):
        print(f"File {annotation_file} already exists.")
    else:
        url = "https://data.csail.mit.edu/clevrer/annotations/validation/annotation_validation.zip"
        r = requests.get(url)
        with open(annotation_file, "wb") as f:
            f.write(r.content)
        # Now unzip the file
        with zipfile.ZipFile(annotation_file, "r") as zip_ref:
            zip_ref.extractall("clevrer")
    
    # Clean the data
    with open("clevrer/validation.json") as f:
        clevrer_data = json.load(f)
    # Start with count.
    clevrer_count = []
    for video in clevrer_data:
        for q in video["questions"]:
            if q.get("question_subtype") == "count":
                clevrer_count.append({**q, "scene_index": video["scene_index"], "video_filename": video["video_filename"]})
    print(f"Found {len(clevrer_count)} counting questions in CLEVRER.")
    with open(f"../data/clevrer/validation_count.json", "w") as f:
        json.dump(clevrer_count, f, indent=4)
    # Now let's do spatial. We need to do this using the annotations. Basically for every json in clevrer/annotation_10000-11000/, clevrer/annotation_11000-12000/, clevrer/annotation_12000-13000/, clevrer/annotation_13000-14000/, clevrer/annotation_14000-15000/, we need to iterate over the files, each a json, and design spatial questions. The spatial questions are similar to those asked in VSI-Bench in both "Relative Direction" and "Relative Distance". For each file, we will output all possible relationships in that video.
    def process_motion_data(data, threshold=0.5, camera_view_threshold=10):
        """Get the closest distances between objects and spatial relations between objects in a video. threshold is the minimum distance between objects to consider them as having a spatial relation. camera_view_threshold is the minimum number of times an object must be in camera view to be considered. Both chosen somewhat arbitrarily but with tweaking."""
        # Initialize data structures
        min_distances = defaultdict(dict)  # For closest distances between objects
        relations = defaultdict(lambda: defaultdict(lambda: {"horizontal": [], "vertical": []}))  # For spatial relations
        num_times_in_camera_view = defaultdict(int)  # For counting how many times an object is in camera view
        # first get number of times in camera view
        for frame in data["motion_trajectory"]:
            objects = frame["objects"]
            for obj in objects:
                if obj["inside_camera_view"]:
                    num_times_in_camera_view[obj["object_id"]] += 1

        # For pairs of objects with all relations below the threshold. We want to keep these, throwing out the pair when there is a relation below the threshold but in the relations dict. To do so, we need to keep track of all instances that have a potential relation that is below the drop threshold by saving a dict with obj1: obj2: set(relation with high enough value). We will then check if that pair has any of such relations above the drop threshold by checking the relations dict. If not, we will discard the pair.
        below_threshold_relations = defaultdict(lambda: defaultdict(set))
        for frame in data["motion_trajectory"]:
            objects = frame["objects"]
            locations = {obj["object_id"]: obj["location"][:2] for obj in objects if obj["inside_camera_view"] and num_times_in_camera_view[obj["object_id"]] >= camera_view_threshold}  # Use x, y only and only if in camera view. Note that coords are (y, x) in CLEVRER, and y is inverted, i.e., increasing y goes down. Decreasing x goes left, as expected.

            # Calculate pairwise distances and relations
            for obj1_id, loc1 in locations.items():
                for obj2_id, loc2 in locations.items():
                    if obj1_id == obj2_id:
                        continue

                    # Distance calculation
                    dist = sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
                    if obj2_id not in min_distances[obj1_id] or dist < min_distances[obj1_id][obj2_id]:
                        min_distances[obj1_id][obj2_id] = dist

                    # Spatial relations
                    y_diff, x_diff = loc2[0] - loc1[0], loc2[1] - loc1[1]  # y_diff is inverted, i.e., if y_diff is positive, y_2 > y_1 meaning obj2 is below obj1, so obj1 is above obj2.
                    # x_diff, y_diff = loc2[0] - loc1[0], loc2[1] - loc1[1]

                    # Set up pair key to add to discard_pairs if needed
                    # pair_key = f"{obj1_id}-{obj2_id}"

                    # Horizontal (left/right)
                    if x_diff > 0:
                        new_relation = "left of"
                    else:
                        new_relation = "right of"
                    if abs(x_diff) > threshold:
                        # Append if it differs from the last recorded horizontal relation
                        if not relations[obj1_id][obj2_id]["horizontal"] or relations[obj1_id][obj2_id]["horizontal"][-1] != new_relation:
                            relations[obj1_id][obj2_id]["horizontal"].append(new_relation)
                    else:
                        below_threshold_relations[obj1_id][obj2_id].add(new_relation)

                    # Vertical (above/below)
                    if y_diff > 0:
                        new_relation = "above"
                    else:
                        new_relation = "below"
                    if abs(y_diff) > threshold:
                        # Append if it differs from the last recorded vertical relation
                        if not relations[obj1_id][obj2_id]["vertical"] or relations[obj1_id][obj2_id]["vertical"][-1] != new_relation:
                            relations[obj1_id][obj2_id]["vertical"].append(new_relation)
                    else:
                        below_threshold_relations[obj1_id][obj2_id].add(new_relation)
                    
        # Finalize ranking distances (sorted lists, one entry per closest pair)
        ranked_distances = {
            obj1_id: sorted(
                [(obj2_id, dist) for obj2_id, dist in min_distances[obj1_id].items()],
                key=lambda x: x[1],
            )
            for obj1_id in min_distances
        }

        # print the default dict
        # for obj1_id in relations:
        #     for obj2_id in relations[obj1_id]:
        #         if relations[obj1_id][obj2_id]["horizontal"] or relations[obj1_id][obj2_id]["vertical"]:
        #             print(f"obj1_id: {obj1_id}, obj2_id: {obj2_id}, {relations[obj1_id][obj2_id]}")
        # print(discard_pairs)
        # Remove pairs with at least one relation where all the values are below the drop threshold
        # new_relations = defaultdict(lambda: defaultdict(lambda: {"horizontal": [], "vertical": []}))
        # for obj1_id in relations:
        #     for obj2_id in relations[obj1_id]:
        #         if f"{obj1_id}-{obj2_id}" in keep_pairs:
        #             new_relations[obj1_id][obj2_id] = relations[obj1_id][obj2_id]
        # relations = new_relations

        # Ensure that every relation in below_threshold_relations belongs in relations. If not, we have an invalid pair as one of the relations will be too close to call.
        for obj1_id in below_threshold_relations:
            for obj2_id in below_threshold_relations[obj1_id]:
                # if obj1_id == 1 and obj2_id == 2:
                #     print(f"obj1_id: {obj1_id}, obj2_id: {obj2_id}\n{below_threshold_relations[obj1_id][obj2_id]=}\n{relations[obj1_id][obj2_id]=}")
                if not all(rel in relations[obj1_id][obj2_id]["horizontal"] + relations[obj1_id][obj2_id]["vertical"] for rel in below_threshold_relations[obj1_id][obj2_id]):
                    # print(f"dropping pair {obj1_id}-{obj2_id} due to below threshold relations")
                    del relations[obj1_id][obj2_id]

        # Convert spatial relations to JSON-friendly format
        spatial_relations = {obj1_id: dict(relations[obj1_id]) for obj1_id in relations}

        return ranked_distances, spatial_relations

    os.makedirs(f"clevrer/spatial", exist_ok=True)
    for i in range(10000, 15000, 1000):
        for j in range(i, i + 1000):
            # if os.path.exists(f"../data/clevrer/spatial/{j}.json"):
            #     continue
            with open(f"clevrer/annotation_{i}-{i+1000}/annotation_{j}.json") as f:
                data = json.load(f)
            ranked_distances, spatial_relations = process_motion_data(data)
            with open(f"../data/clevrer/spatial/{j}.json", "w") as f:
                json.dump({"ranked_distances": ranked_distances, "spatial_relations": spatial_relations, "scene_index": data["scene_index"], "video_filename": data["video_filename"], "object_property": data["object_property"]}, f, indent=4)

    if download_videos:
        # raise NotImplementedError("Downloading videos is not yet supported.")
        # use http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip
        # want to extract to clevrer/video_validation/
        video_file = "clevrer/video_validation.zip"
        if os.path.exists(video_file):
            print(f"File {video_file} already exists.")
        else:
            print("Downloading videos... may take a while.")
            url = "https://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip"
            # add download progress bar
            r = requests.get(url, stream=True)
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            t = tqdm(total=total_size, unit='iB', unit_scale=True)
            with open(video_file, "wb") as f:
                for data in r.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()
            if total_size != 0 and t.n != total_size:
                print("An error occurred while downloading the file.")
            else:
                print("Downloaded video file.")
            # Now unzip the file
            with zipfile.ZipFile(video_file, "r") as zip_ref:
                zip_ref.extractall("clevrer")
    
    print("Downloaded CLEVRER dataset.")

def download_neptune(download_videos=False):
    """Download the Neptune dataset. By default, we only download the existing questions, not the videos.
    
    Here are the questions: 
    Details from: https://storage.mtls.cloud.google.com/neptunedata/neptune_mmh.json. Since we are not using audio now, we choose to download the MMH subset. From Sec 5.1 of the paper, ``In order to create a more challenging visual benchmark, we also provide Neptune-MMH (multimodal human annotated), where we identify videos where vision should play an important role. This is created by using the rater annotations for what modalities are required to answer the question (described in Sec. 4.5), and discarding questions which the raters marked can be solved by audio-only, and consists of 1,171 QADs for 1,000 videos.''
        We provide links to json files that contain the YouTube IDs and annotations for each split below. Please see the paper for details regarding each split.

        The json files contains the following fields:

        key: Unique identifier for each question
        video_id: YouTube URL
        question: Free-form question
        answer: Free-form answer
        answer_choice_{i}: Decoys for MCQ evaluation, i in range(0,4)
        answer_id: ID of the correct answer in the decoys
        question type: Question type
    
    """
    # let's download that json file
    os.makedirs("neptune", exist_ok=True)
    question_file = "neptune/neptune_mmh.json"
    if os.path.exists(question_file):
        print(f"File {question_file} already exists.")
    else:
        url = "https://storage.mtls.cloud.google.com/neptunedata/neptune_full.json"
        raise NotImplementedError(f"The link is not working due to the need for authentification. Please download the file manually at {url} and place it in the 'neptune' directory.")
        r = requests.get(url)
        with open(question_file, "wb") as f:
            f.write(r.content)

    if download_videos:
        raise NotImplementedError("Downloading videos is not yet supported.")
    
    print("Downloaded Neptune dataset.")

if __name__ == "__main__":
    download_vatex()
    # download_activitynet_captions()
    # download_wordnet()
    # download_pexels()
    # download_clevrer(download_videos=True)
    download_neptune()