import os
import numpy as np

def get_known_embeddings(upload_folder, evaluator):
    local_known_faces = {}

    for people in os.listdir(upload_folder):
        people_dir = os.path.join(upload_folder, people)
        encoding_list = []

        for filename in os.listdir(people_dir):
            encoding_dict = {}
            image_path = os.path.join(people_dir, filename)
            face = evaluator.prepare_image(image_path)
            if face is not None:
                face_embedding = evaluator.get_embeddings(face)

                encoding_dict['filename'] = filename
                encoding_dict['embedding'] = face_embedding
                encoding_list.append(encoding_dict)
            else:
                print(f"No faces found in {filename}")

        local_known_faces[people] = encoding_list
    return local_known_faces

def convert_numpy_float32(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError("Object of type {} is not JSON serializable".format(type(obj)))