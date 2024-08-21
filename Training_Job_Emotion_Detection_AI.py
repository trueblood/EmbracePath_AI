import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
import datetime
import time
import psutil
from tensorflow.keras.callbacks import ModelCheckpoint
import sys

print(f"Python Version: {sys.version}")
print(f"TensorFlow Version: {tf.__version__}")

# Argument parsing
parser = argparse.ArgumentParser(description='Train a CNN on a facial emotion recognition dataset.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
parser.add_argument('--output_path', type=str, default='gs://storage_for_all/models', help='GCS path for saving the model.')
args = parser.parse_args()

# Load CSV data
csv_path = 'gs://storage_for_all/DataSets/FacialEmotionRecognitionImageDataset_v1/data.csv'
df = pd.read_csv(csv_path)
df = df[df['label'].isin(['Happy', 'Sad', 'Neutral'])]

# Display the total number of unique labels in the 'label' column
unique_labels = df['label'].unique()
total_unique_labels = len(unique_labels)

print("Unique labels:", unique_labels)
print("Total number of unique labels:", total_unique_labels)

# Print the original paths
print("Original Paths:")
#print(df['path'].head())

df = df.drop(df.columns[0], axis=1)

dataset_size = len(df)
df = df.sample(dataset_size).reset_index(drop=True) # limit number of values and shuffle

# Adjust paths
df['path'] = df['path'].apply(lambda x: f"gs://storage_for_all/DataSets/FacialEmotionRecognitionImageDataset_v1/dataset/{x.split('/')[-2]}/{x.split('/')[-1]}")
print(df.head)

# Print the updated paths
print("\nUpdated Paths:")

# Label encoding for ML processing
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['label'])
one_hot_encoded_labels = to_categorical(integer_encoded)

# Dataset preparation function
def load_image(file_path, label):
    """Load and preprocess images from file paths, handling different formats based on extensions."""
    try:
        image_data = tf.io.read_file(file_path)
        
        # Conditionally decode based on the file extension
        def decode_jpeg():
            return tf.image.decode_jpeg(image_data, channels=3)
        
        def decode_png():
            return tf.image.decode_png(image_data, channels=3)
        
        # Default to using decode_image which works for most formats but does not return a shape statically
        def decode_fallback():

            image = tf.image.decode_image(image_data, channels=3, expand_animations=False)
            print("Image shape:", image.shape)

            return image
        
        # Check the file extension and decode accordingly
        image = tf.cond(
            tf.strings.regex_full_match(file_path, ".*\.jpeg$|.*\.jpg$"),
            true_fn=decode_jpeg,
            false_fn=lambda: tf.cond(
                tf.strings.regex_full_match(file_path, ".*\.png$"),
                true_fn=decode_png,
                false_fn=decode_fallback
            )
        )
        
        image = tf.image.resize(image, [224, 224])
        return image, label
    except tf.errors.NotFoundError:
        print(f"Failed to load image at: {file_path}")
        return None, label
    except Exception as e:
        print(f"Error processing image at: {file_path}", str(e))
        return None, label

# Create TensorFlow datasets
full_dataset = tf.data.Dataset.from_tensor_slices((df['path'].tolist(), one_hot_encoded_labels))
full_dataset = full_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) # This line of code applies the load_image function to each 
full_dataset = full_dataset.filter(lambda x, y: x is not None and y is not None) #clean dataset with clean data
full_dataset = full_dataset.batch(args.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

# Split dataset into training and validation
dataset_size = len(df)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset = full_dataset.take(train_size)
val_dataset = full_dataset.take(train_size)

def tf_model(num_classes):
    print("num_classes value is:", num_classes)
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), padding='same', input_shape=(224, 224, 3)),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64),
        layers.LeakyReLU(alpha=0.1),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the model name and local path
model_name = f"Emotion_Detection_AI_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_v1.h5"

##model_path = os.path.join(args.output_path, model_name)


#model_name = "Emotion_Detection_AI_20240803_205050_v1.h5"

local_model_path = f"/tmp/{model_name}"

# Create a new directory for models if it doesn't exist
#model_dir = "gs://storage_for_all/models"
model_dir = args.output_path
full_model_path = f"{model_dir}/{model_name}"

num_classes = len(label_encoder.classes_)  # Number of unique classes
model = tf_model(num_classes)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    local_model_path,
    save_weights_only=False,
    save_best_only=False,
    verbose=1  # Logs output whenever the model is saved.
)

# Define a custom callback to copy the model to GCS
class CopyModelToGCS(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Save the model locally
        self.model.save(local_model_path, save_format='h5')

        # Copy the model to Google Cloud Storage
        gcs_model_path = f"gs://storage_for_all/models/{model_name}"
        os.system(f"gsutil cp {local_model_path} {gcs_model_path}")

        # Verify the model was copied
        if tf.io.gfile.exists(gcs_model_path):
            print(f"Model saved successfully to {gcs_model_path}")
        else:
            print("Failed to save the model to GCS")

# Create a list of callbacks
callbacks = [checkpoint_callback, CopyModelToGCS()]

# Train the model with the callbacks
history = model.fit(
    train_dataset,
    epochs=args.epochs,
    validation_data=val_dataset,
    callbacks=callbacks,
    use_multiprocessing=True,
    workers=3,
    verbose=1
)
   
# Save the model locally
model.save(local_model_path, save_format='h5')

# Copy the model to Google Cloud Storage
gcs_model_path = f"gs://storage_for_all/models/{model_name}"
os.system(f"gsutil cp {local_model_path} {gcs_model_path}")

# Verify the model was copied
if tf.io.gfile.exists(gcs_model_path):
    print(f"Model saved successfully to {gcs_model_path}")
else:
    print("Failed to save the model to GCS")
  