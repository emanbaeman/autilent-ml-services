import os
import time
from dotenv import load_dotenv
from utils import get_known_embeddings
from create_face_embeddings import EvalSingleImage
from flask import Flask, request, jsonify
from memory_profiler import profile

evaluator = None
known_faces = None
model_loaded = False


# Load environment variables from .env file
load_dotenv()

# Access the variables
model_path = os.getenv("model_path")
upload_folder = os.getenv("upload_folder")
detector_path = os.getenv("detector_path")  # YOLOv8 detector model path

if evaluator is None:
    evaluator = EvalSingleImage(model_path, detector_path)
    model_loaded =True
    
if known_faces is None:
    known_faces = get_known_embeddings(upload_folder, evaluator)
    
    
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    global model_loaded
    if model_loaded:
        return jsonify(status='ok', message='Model is ready'), 200
    else:
        return jsonify(status='not ready', message='Model is still loading'), 503


@app.route('/update_embeddings', methods=['GET'])
def update_embeddings():
    try:
        global known_faces  # Reference the global variable to update it
        known_faces = get_known_embeddings(upload_folder, evaluator)
        return jsonify(status='success', message='Embeddings updated'), 200
    except Exception as e:
        return jsonify(status='error', message=str(e)), 500

@app.route('/process_frame', methods=['POST'])
def process_frame():
    start_time = time.time()
    global model_loaded

    if not model_loaded:
        return jsonify(status='not ready', message='Model is still loading'), 503

    if request.method == 'POST':
        # Assuming the frame data is sent in the request
        data = request.files['image']
        data.save('img.jpg')
        face = evaluator.prepare_image('img.jpg')

        # Calculate embeddings using your face recognition module
        face_embedding = evaluator.get_embeddings(face)

        result_list = []

        # Compare with known embeddings
        for person, file_and_embs_list in known_faces.items():
            for file_dict in file_and_embs_list:
                filename = file_dict['filename']
                embedding = file_dict['embedding']
                similarity = evaluator.dist_func(embedding, face_embedding)

                temp_dict = {'person': person, 'filename': filename, 'similarity': float(similarity)}
                result_list.append(temp_dict)
            
        end_time = time.time()
        print(f'Time taken: {end_time - start_time}')
        return jsonify(result_list)
    
if __name__ == "__main__":
    app.run(debug=False)