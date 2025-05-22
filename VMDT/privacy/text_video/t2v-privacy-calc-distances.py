import pickle

from models import T2VModelName, t2v_model_list

if __name__ == "__main__":
    ### Inputs:
    all_webvid_info_dir = ""
    top_n = 10
    
    model_names = [str(t2v_model) for t2v_model in t2v_model_list]

    for model_name in models_names:
        file = os.path.join(out_dir, model_name, "all_webvid_info.pkl")
        # Open and read the pickle file
        with open(file, 'rb') as f:
            data = pickle.load(f)

        model = {}
        for info in data:
            
            name = info['name']
            num_webvid_subset_frames = info['num_webvid_subset_frames']
            num_model_subset_frames = info['num_model_subset_frames']
            frame_pairs_to_distance = info['frame_pairs_to_distance']
            
            pair_idx_l2_dist = [(idx, distances['l2']) for idx, distances in frame_pairs_to_distance.items()]
            pair_idx_l2_dist.sort(key=lambda x: x[1])

            l2_top_n_average = sum(n for _,n in pair_idx_l2_dist[:top_n])/top_n
            
            pair_idx_clip_dist = [(idx, distances['clip_dist']) for idx, distances in frame_pairs_to_distance.items()]
            pair_idx_clip_dist.sort(key=lambda x: x[1])

            clip_dist_top_n_average = sum(n for _,n in pair_idx_clip_dist[:top_n])/top_n
            
            pair_idx_clip_sim_dist = [(idx, distances['clip_sim']) for idx, distances in frame_pairs_to_distance.items()]
            pair_idx_clip_sim_dist.sort(key=lambda x: abs(x[1]))

            clip_sim_top_n_average = sum(n for _,n in pair_idx_clip_sim_dist[:top_n])/top_n
            
            top_n_average = {"l2_top_n_average": l2_top_n_average, "clip_dist_top_n_average": clip_dist_top_n_average, "clip_sim_top_n_average": clip_sim_top_n_average}
            
            model[name] = top_n_average
        
        l2_top_n_average_overall = 0.0
        clip_dist_top_n_average_overall = 0.0
        clip_sim_top_n_average_overall = 0.0

        for name, val in model.items():
            l2_top_n_average_overall += val["l2_top_n_average"]
            clip_dist_top_n_average_overall  += val["clip_dist_top_n_average"]

        l2_top_n_average_overall/=len(model.items())    
        clip_dist_top_n_average_overall/=len(model.items())    
        clip_sim_top_n_average_overall/=len(model.items())    

        print(file)
        print(l2_top_n_average_overall)
        print(clip_dist_top_n_average_overall)
