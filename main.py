
from dataset_loader import DataLoader, preprocessor
from detector import background_subtractor, yolo_detector
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

        detector = yolo_detector()
        # only give first frame pair for detecting using YOLO, 
        # because it is not a tracking algorithm, 
        # it is a detection algorithm. 
        # We will use the detected bounding box as the template for 
        # template matching in the next step.
        ds_gray_frame_pair = [(frame_pair[0], frame_pair[1]) for frame_pair in ds_preprocessed][0]
        ds_detected = detector.detect(ds_gray_frame_pair)
        
        # feature_extractor = optical_flow()
        # ds_features = feature_extractor.extract(ds_detected)

        # visualizer = visualization(ds_detected)
        # visualizer.produce_heatmap(ds_features)
        # visualizer.produce_trajectory(ds_features)
        # visualizer.produce_combined_visualization(ds_features)
        # visualizer.produce_bounding_boxes(ds_features)

if __name__ == "__main__":
    print("Starting main program...")
    Main()