# Sign-Language-Recognition-Model



**Sign Language Recognition with Text-to-Speech**

This project implements a real-time Sign Language Recognition (SLR) system that combines computer vision, machine learning, and text-to-speech functionalities. It allows users to communicate using hand gestures, with the system translating signs into spoken words or displayed text.

**Project Goals**

* Bridge the communication gap between deaf and hearing individuals through intuitive sign language recognition.
* Foster inclusivity by providing a user-friendly interface for interacting with technology through hand gestures.
* Contribute to the field of SLR research by providing a versatile tool for further development and evaluation.

**System Functionality**

1. **Data Preprocessing:**
   - A dataset of hand gesture images for various signs (letters, words, etc.) is prepared.
   - Images are preprocessed to extract key features, such as hand landmark coordinates.

3. **Model Training:**
   - A machine learning model, specifically a Random Forest Classifier, is trained on the labeled dataset.
   - The model learns to associate hand landmark patterns with their corresponding signs.

4. **Real-time Recognition:**
   - The webcam captures video frames.
   - Hand landmarks are detected and extracted from each frame using MediaPipe, a powerful computer vision library.
   - The trained model predicts the sign based on the extracted hand landmark features.

5. **Text-to-Speech (Optional):**
   - The predicted sign is translated into spoken language using a text-to-speech engine (pyttsx3).
   - This provides an audio representation of the recognized sign, enhancing user experience.

6. **Visual Output:**
   - The video frame with the detected hand and the predicted sign is displayed.

**Execution Instructions**

1. **Prerequisites:**

   - Python 3.x ([https://www.python.org/](https://www.python.org/))
   - Necessary libraries:
     - OpenCV (`pip install opencv-python`)
     - MediaPipe (`pip install mediapipe`)
     - NumPy (`pip install numpy`)
     - Pickle (`pip install pickle`)
     - Scikit-learn (`pip install scikit-learn`)
     - Matplotlib (`pip install matplotlib`)
     - Seaborn (`pip install seaborn`)
     - pyttsx3 (`pip install pyttsx3`)
     - Pillow (`pip install Pillow`) (optional, for image display in GUI)
     - tkinter (`pip install tkinter`) (optional, for the basic GUI)

2. **Data Preparation:**

   - If you want to train your own model, you'll need to collect and label a dataset of hand gesture images.
   - Consider using the provided code (`data_creation.py`) as a starting point for data collection, or explore existing SLR datasets.

3. **Model Training :**

   - If you have a dataset, run the training script (`train_model.py`) to train the Random Forest Classifier.
   - This script will create a pickle file (`model.p`) containing the trained model.

4. **Real-time Recognition:**

   - Run the main script (`sign_language_recognition.py`).
   - This script loads the trained model and uses your webcam to capture video frames for real-time sign recognition.

**Additional Notes**

* This project uses a basic GUI (tkinter) for illustration purposes. You can customize the interface or integrate it into more advanced applications.
* The model accuracy can be further improved by using more sophisticated machine learning models or techniques.
* Consider exploring additional features like sentence recognition or sign disambiguation to enhance the system's capabilities.

