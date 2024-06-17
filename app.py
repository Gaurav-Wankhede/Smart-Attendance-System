import datetime
import os
import random
import re
import time
import zipfile
import cv2
import face_recognition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from mlxtend.plotting import plot_decision_regions
from skimage import exposure
from skimage import io
from skimage.filters import laplace
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TF_ENABLE_ONEDNN_OPTS = 0
class Attendance_Model:
    # ================================= Declaration ========================================
    # Load your trained model
    model = load_model('model.h5')
    extraction_path = 'nisha'

    SCALE_FACTOR = 1.3
    MIN_NEIGHBORS = 5
    MIN_SIZE = (30, 30)
    MAX_FRAMES = 100
    label_to_int_map = {}

    def __init__(self):
        self.images = []
        self.student_image_count = []
        self.train_path = []
        self.extracted_files = []

    # ================================= Extracting Pre-Tested Dataset ========================================
    def extract_zip(self):
        # Define the path to the zip file and the extraction path
        zip_path = 'nisha.zip'

        # Unzipping the dataset
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.extraction_path)

        # Listing the contents of the unzipped folder
        self.extracted_files = os.listdir(self.extraction_path)
        st.write(f"Extracted files and directories: {self.extracted_files}")
        return self.extracted_files

    # ================================= Add Student ========================================

    def capture_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            return None
        return frame

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(gray, scaleFactor=self.SCALE_FACTOR, minNeighbors=self.MIN_NEIGHBORS,
                                             minSize=self.MIN_SIZE)
        return faces

    def save_frame(self, original_frame, faces, student_name, roll_number, random_number, train_dir, frame_count,
                   progress_bar):

        for (x, y, w, h) in faces:
            face_image = original_frame[y:y + h, x:x + w]
            frame_path = os.path.join(train_dir, f'{student_name}.{roll_number}.{random_number}.{frame_count}.jpg')
            cv2.imwrite(frame_path, face_image)

            # Update the progress bar
            progress_bar.progress(frame_count / self.MAX_FRAMES)
            frame_count += 1
        return frame_count

    def capture_video(self, student_name, roll_number, random_number, train_dir, frame_count):
        cap = cv2.VideoCapture(0)
        start_time = time.time()

        # Create a progress bar
        progress_bar = st.progress(0)

        while True:
            frame = self.capture_frame(cap)
            if frame is None:
                break

            faces = self.detect_faces(frame)

            original_frame = frame.copy()

            # Draw a green rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Video', frame)

            # Save a frame every second
            if time.time() - start_time > 1:
                frame_count = self.save_frame(original_frame, faces, student_name, roll_number, random_number,
                                              train_dir,
                                              frame_count, progress_bar)
                start_time = time.time()
            if frame_count >= self.MAX_FRAMES:
                break

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Show the messsage after complete
        st.write(f"Your data has been captured for {frame_count} frames.")

        # Release the capture and destroy all windows
        cap.release()
        cv2.destroyAllWindows()

    #================================= Training Model ========================================

    def exploring(self):
        # Check if extracted_files is empty
        if not self.extracted_files:
            # Listing the contents of the unzipped folder
            self.extracted_files = os.listdir(self.extraction_path)

        # Check again if extracted_files is still empty
        if not self.extracted_files:
            st.write("No files found in extraction path")
            return
        dataset_path = os.path.join(self.extraction_path, self.extracted_files[0])

        # Adjust the path to include the 'train' directory
        train_path = os.path.join(dataset_path, 'train')

        # Exploring the dataset structure
        images = os.listdir(train_path)
        st.write(f"Total images: {len(images)}")

        # Visualizing a few sample images
        sample_images = [os.path.join(train_path, img) for img in images[:5]]

        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for ax, img_path in zip(axes, sample_images):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
        st.pyplot(fig)  # Display the figure with Streamlit

        # Analyzing image dimensions and counting occurrences
        dimension_counts = {}
        for img_name in images[:100]:  # Analyze a subset of images
            img_path = os.path.join(train_path, img_name)
            with Image.open(img_path) as img:
                dimension = img.size
                if dimension in dimension_counts:
                    dimension_counts[dimension] += 1
                else:
                    dimension_counts[dimension] = 1

        # Create a DataFrame from the dimension counts dictionary
        dimensions_df = pd.DataFrame(list(dimension_counts.items()), columns=['Dimension', 'Count'])

        # Display the DataFrame as a table
        st.dataframe(dimensions_df)  # Display the DataFrame with Streamlit

        # Calculate and visualize the histogram of pixel values for the first image
        with Image.open(sample_images[0]) as img:
            histogram = img.histogram()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(histogram[:256], color='r')  # Red channel for RGB image, or grayscale level for L image
        ax.set_title('Histogram of Pixel Values')
        ax.set_xlabel('Pixel value')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)  # Display the figure with Streamlit

        # Checking the distribution of images per student
        student_image_count = {}
        for img_name in images:  # Iterate over the list of image filenames
            student_id = img_name.split('.')[0]
            if student_id in student_image_count:
                student_image_count[student_id] += 1
            else:
                student_image_count[student_id] = 1

        # Display the distribution of images per student
        for student_id, count in sorted(student_image_count.items()):
            st.write(f"Student ID: {student_id}, Image Count: {count}")

        # ================= Pre-Processing of the Dataset ==============================

        # Function to assess the blurriness of an image
        def calculate_blurriness_score(img_path):
            img = io.imread(img_path, as_gray=True)
            return np.var(laplace(img))

        # Calculate the blurriness score for each image
        blurriness_scores = [calculate_blurriness_score(os.path.join(train_path, img_name)) for img_name in
                             images[:100]]
        st.write(f"Blurriness score images: {np.sum(blurriness_scores)}")

        # Function to assess the contrast of an image
        def calculate_contrast_score(img_path):
            img = io.imread(img_path, as_gray=True)
            return exposure.is_low_contrast(img)

        # Assess the contrast for each image
        contrast_scores = [calculate_contrast_score(os.path.join(train_path, img_name)) for img_name in images[:100]]
        st.write(f"Low contrast images: {np.sum(contrast_scores)}")

        # Add the contrast check to the blurriness_scores loop to save computation
        for img_name in images[:100]:
            img_path = os.path.join(train_path, img_name)
            blurriness_score = calculate_blurriness_score(img_path)
            is_low_contrast = calculate_contrast_score(img_path)
            st.write(f"Image: {img_name}, Blurriness: {blurriness_score}, Low Contrast: {is_low_contrast}")

        # ================= Distribution of Images per Student ==============================

        # Analyze the distribution balance
        student_counts = pd.Series(student_image_count)
        st.write("Statistics of images per student:")
        st.write(student_counts.describe())

        # Check if student_counts is empty or contains non-numeric data
        if student_counts.empty or not np.issubdtype(student_counts.dtypes, np.number):
            st.write("No numeric data to plot")
        else:
            # Visual representation of balance
            fig, ax = plt.subplots()
            student_counts.plot(kind='bar', ax=ax)
            ax.set_title('Distribution of Images per Student')
            ax.set_xlabel('Student ID')
            ax.set_ylabel('Number of Images')
            st.pyplot(fig)  # Display the figure with Streamlit

        # ================= Image Resizing and Processing/Splitting ==============================

        def preprocess_image(image_path, target_size=(128, 128)):
            # Load image
            image = Image.open(image_path)
            # Check if image is not grayscale
            if image.mode != 'L':
                image = image.convert('L')
            # Resize image
            image = image.resize(target_size)
            # Convert image to numpy array and normalize to range [0, 1]
            image_array = np.array(image) / 255.0
            return image_array

        # `train_path` is the directory containing the images
        train_contents = os.listdir(train_path)

        # Initialize lists for storing image data and labels
        data = []
        labels = []

        # Loop over all images and preprocess them
        for img_name in train_contents:
            img_path = os.path.join(train_path, img_name)
            img_array = preprocess_image(img_path)
            data.append(img_array)
            # Extracting labels from file names
            labels.append(img_name.split('.')[0])

        # Convert to numpy arrays
        data = np.array(data)
        labels = np.array(labels)

        # Reshape data to add channel dimension if model expects
        data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,  # Rotation range in degrees
            width_shift_range=0.15,  # Fraction of total width for horizontal shift
            height_shift_range=0.2,  # Fraction of total height for vertical shift
            shear_range=0.15,  # Shear intensity
            zoom_range=0.1,  # Zoom range
            horizontal_flip=True,  # Enable horizontal flipping of images
            fill_mode='nearest'
        )

        unique_labels = np.unique(labels)
        self.label_to_int_map = {label: index for index, label in enumerate(unique_labels)}
        encoded_labels = np.array([self.label_to_int_map[label] for label in labels])

        # Split the dataset into training and validation sets
        train_data, val_data, train_labels, val_labels = train_test_split(
            data, encoded_labels, test_size=0.2, random_state=42
        )

        # Display the shapes of the training and validation sets
        st.write(f"Train data shape: {train_data.shape}")
        st.write(f"Validation data shape: {val_data.shape}")

        # visualize some preprocessed images
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for ax, img_array in zip(axes, train_data[:5]):
            ax.imshow(img_array.squeeze(), cmap='gray')  # Use squeeze() to remove single-dimensional entries
            ax.axis('off')
        st.pyplot(fig)  # Display the figure with Streamlit

        # ================= Generating the Training Data ==============================

        # Training data generator with augmentation
        train_generator = datagen.flow(train_data, train_labels, batch_size=32)
        st.write(f"Created train_generator with {len(train_generator)} batches of size 32")

        # Validation data generator without augmentation
        validation_datagen = ImageDataGenerator()
        validation_generator = validation_datagen.flow(val_data, val_labels, batch_size=32)
        st.write(f"Created validation_generator with {len(validation_generator)} batches of size 32")

        # ================= Defining and Loading the CNN Model ==============================

        # Define the CNN model
        model = Sequential([
            Input(shape=train_data.shape[1:]),  # Use an Input layer as the first layer
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(len(np.unique(labels)), activation='softmax')
        ])

        # Output layer with a node for each class
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Summary of the CNN model
        st.text(model.summary())

        # ================= Defining and Loading the CNN Model ==============================

        # Calculate the number of steps per epoch for training and validation
        train_steps_per_epoch = len(train_data) // 32
        val_steps_per_epoch = len(val_data) // 32

        # Data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,  # Rotation range in degrees
            width_shift_range=0.2,  # Fraction of total width for horizontal shift
            height_shift_range=0.2,  # Fraction of total height for vertical shift
            shear_range=0.15,  # Shear intensity
            zoom_range=0.1,  # Zoom range
            horizontal_flip=True,  # Enable horizontal flipping of images
            fill_mode='nearest'
        )

        # No data augmentation for validation set
        val_datagen = ImageDataGenerator()

        # Create generators
        train_generator = train_datagen.flow(train_data, train_labels, batch_size=32)
        validation_generator = val_datagen.flow(val_data, val_labels, batch_size=32)

        # Train the model
        history = model.fit(
            train_generator,
            steps_per_epoch=train_steps_per_epoch,  # Number of steps per epoch
            epochs=50,  # Number of epochs
            validation_data=validation_generator,
            validation_steps=val_steps_per_epoch,  # Number of steps per validation epoch
        )

        # Display the training history
        st.write("Training history:")
        st.write(history.history)

        # Save the Model
        model.save('model.h5')
        st.write('Model saved as model.h5')

        # ================= Plotting the Training Curves ==============================

        # Plot training & validation accuracy values
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history.history['accuracy'], label='Train accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation accuracy')
        ax.set_title('Model accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper left')
        st.pyplot(fig)

        # Plot training & validation loss values
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history.history['loss'], label='Train loss')
        ax.plot(history.history['val_loss'], label='Validation loss')
        ax.set_title('Model loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper left')
        st.pyplot(fig)

        # Evaluate the model on the validation set
        val_loss, val_accuracy = model.evaluate(validation_generator, steps=len(val_data) // 32)
        st.write(f'Validation loss: {val_loss}')
        st.write(f'Validation accuracy: {val_accuracy}')

        # Save the Model
        model.save('model.h5')
        st.write('Model saved as model.h5')

        # ================= Implementing SVM Model ==============================

        # Load the CNN model
        model = load_model('model.h5')

        # Create a new model that outputs the activations of the second Dense layer (third from last layer) of the original model
        feature_extractor = Model(inputs=model.layers[0].input, outputs=model.layers[-3].output)

        # Extract features using the feature extractor
        # Replace 'data' with your actual data and 'encoded_labels' with your actual labels
        features = feature_extractor.predict(data)

        # Split features into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

        # Train SVM
        svm_classifier = SVC(kernel='linear')
        svm_classifier.fit(X_train, y_train)

        # Evaluate SVM
        y_pred = svm_classifier.predict(X_test)
        st.write(f"SVM Accuracy: {accuracy_score(y_test, y_pred)}")

        # ===============================================================================================

        # Project features into 2D space for visualization
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)

        # Train SVM on the 2D features
        svm_classifier_2d = SVC(kernel='linear')
        svm_classifier_2d.fit(X_train_pca, y_train)

        # Plot decision regions
        fig, ax = plt.subplots()
        plot_decision_regions(X_train_pca, y_train.astype(np.integer), clf=svm_classifier_2d, legend=2, ax=ax)

        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_title('SVM Decision Boundary in 2D Feature Space')
        st.pyplot(fig)

    # ================================= Prediction ========================================

    def prediction(self):

        # Listing the contents of the unzipped folder
        global predicted_label, unknown
        message = ''
        color = (255, 255, 255)
        extracted_files = os.listdir(self.extraction_path)
        dataset_path = os.path.join(self.extraction_path, extracted_files[0])

        # Adjust the path to include the 'train' directory
        train_path = os.path.join(dataset_path, 'train')

        def preprocess_image(image_path, target_size=(128, 128)):
            # Load image
            image = Image.open(image_path)
            # Check if image is not grayscale
            if image.mode != 'L':
                image = image.convert('L')
            # Resize image
            image = image.resize(target_size)
            # Convert image to numpy array and normalize to range [0, 1]
            image_array = np.array(image) / 255.0
            return image_array

        # `train_path` is the directory containing the images
        train_contents = os.listdir(train_path)

        # Initialize lists for storing image data and labels
        data = []
        labels = []
        roll_number = []

        # Loop over all images and preprocess them
        for img_name in train_contents:
            img_path = os.path.join(train_path, img_name)
            img_array = preprocess_image(img_path)
            data.append(img_array)
            # Extracting labels from file names
            labels.append(img_name.split('.')[0])
            roll_number.append(img_name.split('.')[1])


        # Convert to numpy arrays
        data = np.array(data)
        labels = np.array(labels)
        roll_number = np.array(roll_number)
        st.write(roll_number)

        # Reshape data to add channel dimension if model expects
        data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,  # Rotation range in degrees
            width_shift_range=0.1,  # Fraction of total width for horizontal shift
            height_shift_range=0.1,  # Fraction of total height for vertical shift
            shear_range=0.1,  # Shear intensity
            zoom_range=0.1,  # Zoom range
            horizontal_flip=True,  # Enable horizontal flipping of images
            fill_mode='nearest'
        )

        unique_labels = np.unique(labels)
        label_to_int_map = {label: index for index, label in enumerate(unique_labels)}

        label_to_int_roll = {name: roll for name, roll in zip(labels, roll_number)}

        # Load the trained model
        model = load_model('model.h5')

        # Assuming self.label_to_int_map is a list of labels
        labels = label_to_int_map
        st.write(labels)

        # Ensure label_to_int_map is initialized and has values
        if not label_to_int_map:
            print("Error: label_to_int_map is not initialized or has no values.")
            return

        # Now you can safely access self.label_to_int_map in this method
        int_to_label_map = {v: k for k, v in labels.items()}

        name_to_roll_number_map = {v: k for k, v in label_to_int_roll.items()}
        # interchange key and value
        name_to_roll_number_map = {v: k for k, v in name_to_roll_number_map.items()}

        st.write(name_to_roll_number_map)

        # Initialize the camera
        cap = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame")
                break

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = frame[:, :, ::-1]

            # Find all the faces in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame)

            # If faces were found, we will mark the image
            for top, right, bottom, left in face_locations:
                # You can access the actual face itself like this:
                face_image = frame[top:bottom, left:right]
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                face_image = cv2.resize(face_image, (128, 128))  # Resize to the size expected by the model
                face_image = face_image / 255.0  # Normalize pixel values
                face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension
                face_image = np.expand_dims(face_image, axis=-1)  # Add channel dimension

                # Predict the label of the face
                predictions = model.predict(face_image)
                predicted_class = np.argmax(predictions)
                max_prediction_prob = np.max(predictions)

                threshold = 0.99999999

                if max_prediction_prob > threshold and predicted_class in int_to_label_map:
                    predicted_label = int_to_label_map[predicted_class]
                    predicted_roll_number = name_to_roll_number_map[predicted_label]
                    color = (0, 255, 0)  # BGR color for green
                else:
                    predicted_label = "Unknown"
                    color = (0, 0, 255)  # BGR color for red

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, predicted_label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Store the key press event in a variable
            key = cv2.waitKey(1) & 0xFF

            # Break the loop on pressing 'q'
            if key == ord('q'):
                break

            # Take attendance on pressing 'a'
            if key == ord('a'):
                # Check if the predicted label is "Unknown"
                if predicted_label == "Unknown":
                    message = "Unknown face detected. Please add the new student."
                    color = (0, 0, 255)  # BGR color for red
                    st.write(message)

                else:
                    # Check if the CSV file exists and if not, create it with column names
                    if not os.path.isfile('attendance.csv'):
                        df = pd.DataFrame(columns=['Student_Name', 'Roll_Number' , 'Date', 'Time'])
                    else:
                        # If the CSV file exists, read it into a DataFrame
                        df = pd.read_csv('attendance.csv')

                    # If the predicted label is not "Unknown", append the new entry to the DataFrame
                    if predicted_label != "Unknown":
                        # Get today's date and current time
                        today = datetime.datetime.now().date()
                        now = datetime.datetime.now().time()
                        # Convert the date and time to strings
                        today = today.strftime('%Y-%m-%d')
                        now = now.strftime('%H:%M:%S')

                        # Check if the student has already been marked present today
                        if not ((df['Student_Name'] == predicted_label) & (df['Date'] == str(today))).any():
                            # Concatenate the new entry to the DataFrame
                            new_row = pd.DataFrame(
                                {'Student_Name': [predicted_label], 'Roll_Number': [predicted_roll_number], 'Date': [str(today)], 'Time': [str(now)]})
                            df = pd.concat([df, new_row], ignore_index=True)

                        # Remove duplicate entries for the same student on the same day
                        df = df.drop_duplicates(subset=['Student_Name', 'Date'])

                        # Write the updated DataFrame back to the CSV file
                        df.to_csv('attendance.csv', index=False)

                    message = f"Attendance taken for {predicted_label}"
                    color = (0, 255, 0)  # BGR color for green

                    st.write(f"Attendance taken for {predicted_label}")
                    # Wait for 5 seconds
                    if cv2.waitKey(5000): break

            # Display the message at the bottom of the frame
            cv2.putText(frame, message, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Display the resulting image
            cv2.imshow('Video', frame)

        # Release handle to the webcam
        cap.release()
        cv2.destroyAllWindows()

    # ================================= Calling Training Model ========================================
    def train_model(self):
        st.write("Train your model here")
        self.exploring()

    # ================================= Marking Attendance ========================================

    def mark_attendance(self):
        st.title("Marking Attendance")
        self.prediction()

    # ================================= Checking Attendance ========================================

    def check_attendance(self):
        st.title("Your Attendance Dataset")
        # Check if attendance.csv file exists
        if not os.path.isfile('attendance.csv'):
            st.subheader("No attendance data available")
            return
        else:
            st.subheader("Attendance data available")
            df = pd.read_csv('attendance.csv')
        st.write(df)


# =================================Streamlit App========================================
def main():
    # Create an instance of Attendance_Model
    attendance_model = Attendance_Model()

    # Create a sidebar for navigation
    st.sidebar.title('Navigation')
    options = ['Extract Zip', 'Add Student', 'Train Model', 'Mark Attendance', 'Check Attendance']
    choice = st.sidebar.selectbox("Choose an option", options)

    if choice == 'Extract Zip':
        st.title('Extract Pretested Dataset')
        if st.button('Click to Extract'):
            attendance_model.extract_zip()

    elif choice == 'Add Student':

        st.title('Please add new Student record')
        student_name = st.text_input("Enter the student's name: ")
        roll_number = st.text_input("Enter the student's roll number: ")

        # Check if the inputs are valid
        if student_name and re.match("^[A-Za-z ]*$", student_name) and roll_number and roll_number.isdigit():
            roll_number = int(roll_number)

            random_number = random.randint(1000, 9999)
            train_dir = os.path.join(attendance_model.extraction_path, 'Smart attendance dataset', 'train')
            frame_count = 0

            if st.button('Capture Frame'):
                attendance_model.capture_video(student_name, roll_number, random_number, train_dir, frame_count)
        else:
            if not student_name or not re.match("^[A-Za-z ]*$", student_name):
                st.write(
                    "**Please enter a valid string for the student's name. It should not contain numbers or symbols.**")
            if not roll_number or not roll_number.isdigit():
                st.write("**Please enter a valid integer roll number.**")

    elif choice == 'Train Model':
        st.title("Training the Model")
        st.button('Train the Model', on_click=attendance_model.train_model)

    elif choice == 'Mark Attendance':
        st.title("Mark attendance using OpenCV and your trained model here")
        st.button('Mark your attendance', on_click=attendance_model.mark_attendance)

    elif choice == 'Check Attendance':
        attendance_model.check_attendance()


if __name__ == '__main__':
    main()
