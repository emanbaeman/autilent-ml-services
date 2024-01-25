import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import normalize
from yolov8ObjectDetection.YOLOV8 import YOLOv8
from memory_profiler import profile

class EvalSingleImage:
    def __init__(self, model_interf, detector_path):
        if isinstance(model_interf, str) and model_interf.endswith(".h5"):
            self.model = tf.keras.models.load_model(model_interf, compile=False)
        else:
            self.model = model_interf
        self.face_index = None
        self.detector = YOLOv8(detector_path, conf_thres=0.3, iou_thres=0.5)  # Use the provided YOLOv8 model path
        self.dist_func = lambda a, b: np.dot(a, b)

    def prepare_image(self, image_path):
        img = load_img(image_path)
        img = img_to_array(img)
        img = self.crop_and_detect_face(img)
        if img is not None:
            img = (img - 127.5) * 0.0078125
            img = np.expand_dims(img, axis=0)
            return img
        else:
            return None

    def crop_and_detect_face(self, img):
        # Crop the left side of the image where the driver should be
        height, width, _ = img.shape
        crop_width_multiplier = 0.7
        crop_width = int(width * crop_width_multiplier)
        cropped_image = img[:, :crop_width, :]  # Cropping logic

       #  Detect Objects
        boxes, scores, class_ids = self.detector(cropped_image)
        try:
            for index, class_id in enumerate(class_ids):
                if class_id == 0:
                    self.face_index = index
                    break
            if self.face_index is not None:
                #print(f'boxes: {boxes}, class: {class_ids}, class_id type: {type(class_ids)} face_index: {self.face_index}, face_index type: {type(self.face_index)}')
                x1, y1, x2, y2 = boxes[self.face_index]
                cropped_face = cropped_image[int(y1):int(y2), int(x1):int(x2)]
                resized_face = cv2.resize(cropped_face, (112, 112))  # Resize to expected input size
                return resized_face
        except Exception as e:
            # print(e)
            return None

    
    def get_embeddings(self, img):
        emb = self.model.predict(img)
        emb = normalize(np.array(emb).astype("float32"))[0]
        return emb

    def compare_images(self, img_path1, img_path2):
        img1 = self.prepare_image(img_path1)
        img2 = self.prepare_image(img_path2)

        if img1 is not None or img2 is not None:
            emb1 = self.get_embeddings(img1)
            emb2 = self.get_embeddings(img2)
            return self.dist_func(emb1, emb2)
        else:
            return None
