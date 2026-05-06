from datasets import Audio, load_dataset
import pickle
import os
import numpy as np
import math
def load_birdset_data(subset = "NES", default_lat_lon = None, default_time = "12:00:00", save_dir = "/scratch/e1583377/huggingface/", split = "test_5s"):
    dataset = load_dataset("DBD-research-group/BirdSet", subset, cache_dir = save_dir, download_mode="reuse_cache_if_exists")


    # the dataset comes without an automatic Audio casting, this has to be enabled via huggingface
    # this means that each time a sample is called, it is decoded (which may take a while if done for the complete dataset)
    # in BirdSet, this is all done on-the-fly during training and testing (since the dataset size would be too big if mapping and saving it only once)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))
    if default_lat_lon != None:
        default_lat, default_lon = default_lat_lon
        # Function to fill missing coordinates
        def fill_missing_coordinates(example):
            # Check if lat is missing or None and fill with default
            if example["lat"] is None or math.isnan(example["lat"]):
                example["lat"] = default_lat
            
            # Check if lon is missing or None and fill with default
            if example["long"] is None or math.isnan(example["lon"]):
                example["long"] = default_lon
            local_time = example["local_time"]
            if local_time is None or local_time == "NaT" or ":" not in local_time:
                example["local_time"] = default_time
            
            return example
        
        # Apply the function to fill missing coordinates
        dataset[split] = dataset[split].map(fill_missing_coordinates)
    return dataset[split]
    
if __name__ == '__main__':
    task = "XCL"
    print("starting download!")
    huggingface_dataset = load_birdset_data(task, save_dir = "/scratch/Projects/CFP04/CFP04-CF-029/birdset", split = "train")
    print("Download complete!")