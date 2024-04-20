## main.py - Application Entry Point

### Overview
This Python script uses a graphical user interface (GUI) based on Tkinter to display live camera feed for real-time drowsy detection. It leverages the CNN model trained to detect sleepiness by analyzing facial features in video frames. The application alerts the user with a sound if drowsiness is detected.

### Dependencies
- Tkinter
- OpenCV
- PIL (Python Imaging Library)
- NumPy
- TensorFlow
- Keras
- pygame
- logging

### Features
- **Real-time Sleep Detection:** Utilizes a pre-trained CNN model to detect drowsiness in real-time using video feed from the webcam.
- **Alert System:** Plays a beep sound using pygame when drowsiness is detected.
- **GUI Feedback:** Displays a "WAKE UP" message on the GUI if the user is detected as sleepy.

### Implementation Details
- The script starts by importing necessary libraries and setting up the logging level to suppress verbose output from TensorFlow.
- It initializes the Tkinter window and sets up the webcam feed.
- The `detect_sleep` function reads frames from the webcam, applies face detection, and then predicts sleepiness using the CNN model.
- If sleepiness is detected, `play_beep` function is triggered to sound an alarm.
- The GUI is updated in real-time with the video feed and alerts.

### Usage
To run the application:
1. Ensure all dependencies are installed.
2. Run the script using Python:
   ```bash
   python main.py


## CNN Model

### Overview
This section of the project involves a Convolutional Neural Network (CNN) designed to classify images into two categories: 'notsleep' and 'sleep'. The model is built and trained using TensorFlow and Keras, utilizing various layers including Conv2D, MaxPooling2D, Flatten, and Dense.

### Dependencies
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- scikit-learn
- OpenCV

### Data Preparation
Images are loaded from a Google Drive directory, split into 'notsleep' and 'sleep' categories, and preprocessed into NumPy arrays. The dataset is visualized using Matplotlib to ensure proper loading and preprocessing. (To use for local ensure to change the pathing from drive location to local file)

### Model Architecture
The model uses multiple convolutional layers with different filter sizes and neuron counts, tested across various hyperparameters:
- Learning rates: 0.0001, 0.0005
- Number of filters: 8, 16
- Filter sizes: 3x3, 5x5
- Neurons in dense layers: 32, 64

### Training
Training involves K-Fold cross-validation with 5 folds to optimize the generalization of the model. The best model is selected based on the highest validation accuracy. The model is then saved for further use or deployment.

### Evaluation
After training, the model's performance is evaluated on a test set. Metrics calculated include accuracy, recall, precision, F1-score, and ROC AUC score.

### Final Model
A final model is created using the best parameters from cross-validation. This model is trained on the entire dataset and saved for deployment in the system.

```python
final_model.save("final_model.h5")
```

## SVM Model

### Overview
This section details the implementation of a Support Vector Machine (SVM) used for classifying images as 'notsleep' or 'sleep'. The SVM model is trained using scikit-learn's SVM classifier with a linear kernel.

### Dependencies
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- OpenCV
- dlib

### Data Preparation
Images are preprocessed and faces are detected using OpenCV's deep learning face detector based on the SSD framework with a ResNet base network. Detected faces are cropped and resized for uniformity. The resulting data is converted into a flattened array for SVM training.

### Model Training
The data is split into training and test datasets with an 80-20 ratio, maintaining equal distribution among classes using stratified sampling. The SVM model is trained with a linear kernel, and hyperparameters are optimized using grid search.

### Model Evaluation
The model's performance is evaluated on the test set using accuracy, precision, recall, and F1-score. Additionally, ROC AUC score is calculated to assess the model's ability to distinguish between the two classes.

