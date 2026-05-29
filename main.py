
from dataset_loader import DataLoader, preprocessor
from feature_extractor import optical_flow, template_matching
from visualization import visualization
import cv2


class Main:
    def __init__(self):
        print("load dataset")
        loader = DataLoader()
        ds = loader.load()
        print(f"Dataset loaded with {len(ds)} images.")

        print("preprocess dataset with converting to grayscale and resizing")
        prep = preprocessor()
        ds_preprocessed = prep.convertToGrayScaleAndResize(ds)
        print(f"Preprocessed dataset with {len(ds_preprocessed)} frame pairs.")

        
        ds_images = []
        for frame_pair in ds_preprocessed:
            ds_images.append((frame_pair[0], frame_pair[1]))

        print(f"length of ds_images: {len(ds_images)}")

        of_extractor = optical_flow()
        optical_flow_features = of_extractor.extract_features(ds_images)

        visualizer = visualization()
        visualizer.produce_bounding_boxes(ds, optical_flow_features)

        # print("Extracted optical flow features for the first frame pair.")
        # print(f"Feature shape: {len(features[0])}x{len(features[0][0])}")
        
        
if __name__ == "__main__":
    print("Starting main program...")
    Main()