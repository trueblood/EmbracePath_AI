from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

model_name = "Emotion_Detection_AI_20240803_205050_v1.h5"
model_dir = "gs://storage_for_all/models"
full_model_path = f"{model_dir}/{model_name}"

# Initialize TensorFlow model
model = tf.keras.models.load_model(full_model_path)
label_encoder = LabelEncoder()

def predict_emotion(image_path):
    image_data = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.expand_dims(image, 0)  # Make batch of 1

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    emotion_label = label_encoder.inverse_transform([predicted_class])[0]
    emotion_probabilities = predictions[0].tolist()

    return emotion_label, emotion_probabilities

@app.route('/api/classify_image', methods=['POST'])
def classify_image():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    image_path = data.get('file')
    user_id = data.get('userID')

    if not image_path:
        return jsonify({'error': 'No file provided'}), 400
    if not user_id:
        return jsonify({'error': 'No userID provided'}), 400

    emotion_label, emotion_probabilities = predict_emotion(image_path)

    return jsonify({
        'emotion_label': emotion_label,
        'emotion_probabilities': emotion_probabilities,
        'userID': user_id
    })

@app.route('/api/health_check', methods=['GET'])
def health_check():
    # Assuming HealthCheckService is defined elsewhere
    service = HealthCheckService()
    api_status = service.check_api_status()
    db_status = service.check_database_status()
    return jsonify({"api_status": api_status, "db_status": db_status})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
