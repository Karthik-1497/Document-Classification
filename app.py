import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, flash, redirect, url_for, render_template

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('document_classifier_model.h5')

# Define function to classify and store images
def classify_and_store_image(uploaded_file):
    # Define target directories based on class labels
    target_directories = {
        'aadhar': 'root_dataset_folder/aadhar',
        'pan': 'root_dataset_folder/pan',
        'voter': 'root_dataset_folder/voter',
        'driving': 'root_dataset_folder/driving'
    }

    # Ensure the target directories exist
    for label, directory in target_directories.items():
        os.makedirs(directory, exist_ok=True)

    # Load and preprocess the uploaded image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0  # Normalize the image

    # Make a prediction using the model
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    predicted_class = tf.argmax(prediction, axis=1)
    predicted_label = list(target_directories.keys())[predicted_class[0]]

    # Move the uploaded image to the target directory
    image_filename = os.path.basename(uploaded_file)
    target_directory = target_directories.get(predicted_label)
    os.rename(uploaded_file, os.path.join(target_directory, image_filename))

    return predicted_label

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the 'file' field is in the POST request
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']

        # Check if the file is not empty
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file:
            # Save the uploaded file temporarily
            uploaded_file = os.path.join('uploads', file.filename)
            file.save(uploaded_file)

            # Classify and store the image
            predicted_label = classify_and_store_image(uploaded_file)

            flash(f'Image classified as: {predicted_label}', 'success')
            return redirect(request.url)

    return render_template('upload.html')

if __name__ == '__main__':
    app.secret_key = 'your_secret_key_here'
    app.run(debug=True)
