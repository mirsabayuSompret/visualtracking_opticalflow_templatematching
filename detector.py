from ultralytics import YOLO

class yolo_detector:
    def __init__(self):
        print("Initializing YOLO detector...")
        
        self.model = YOLO('yolov8n.pt')
    
    def detect(self, image):
        """
        Detect kite in image using YOLO.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            list: Detected objects with bounding boxes and confidence scores
        """
        results = self.model(image)
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = result.names[class_id]
                if class_name.lower() == 'kite':
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'confidence': float(box.conf),
                        'class': class_name
                    }
                    detections.append(detection)
        return detections

class background_subtractor:
    def __init__(self):
        print("Initializing background subtractor...")
    
    def detect(self, images):

        print("Subtracting background from the image...")
        pass

