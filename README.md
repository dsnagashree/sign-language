
# Sign Language Detection System

A real-time sign language detection system using computer vision and machine learning. This application captures hand gestures through your webcam, processes them using MediaPipe, and predicts sign language gestures using a trained machine learning model with audio feedback.

## Features

- **Real-time Hand Detection**: Uses MediaPipe to detect and track hand landmarks
- **Gesture Recognition**: Predicts sign language gestures using a pre-trained machine learning model
- **Audio Feedback**: Announces predictions using text-to-speech
- **Visual Feedback**: Displays hand landmarks and predictions on screen
- **Sequence-based Prediction**: Analyzes sequences of 30 frames for more accurate predictions
- **Adjustable Prediction Rate**: Configurable interval between predictions to prevent audio spam

## Requirements

### Python Dependencies

```bash
pip install opencv-python
pip install mediapipe
pip install numpy
pip install scikit-learn  # If your model was trained with sklearn
pip install pyttsx3
```

### Additional Requirements

- **Webcam**: A working camera connected to your computer
- **Trained Model Files**:
  - `sign_language_model.pkl`: Your trained machine learning model
  - `labels_dict.pkl`: Label mapping dictionary
- **Audio System**: Working speakers or headphones for text-to-speech feedback

## File Structure

```
project/
│
├── sign_language_detection.py    # Main application file
├── sign_language_model.pkl       # Trained ML model (required)
├── labels_dict.pkl              # Label mappings (required)
└── README.md                    # This file
```

## Installation

1. **Clone or download** the project files
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Or install individually:
   ```bash
   pip install opencv-python mediapipe numpy pyttsx3
   ```
3. **Ensure model files exist**: Make sure you have both `sign_language_model.pkl` and `labels_dict.pkl` in the same directory

## Usage

1. **Run the application**:
   ```bash
   python sign_language_detection.py
   ```

2. **Position yourself** in front of the camera with good lighting

3. **Perform sign language gestures** - the system will:
   - Display your camera feed with hand landmarks
   - Show predictions on screen
   - Announce predictions via text-to-speech every 5 seconds

4. **Exit** by pressing 'q' or closing the window

## Configuration Options

### Adjustable Parameters

In the code, you can modify these settings:

```python
# MediaPipe confidence thresholds
min_detection_confidence=0.5    # Hand detection sensitivity
min_tracking_confidence=0.5     # Hand tracking sensitivity

# Prediction settings
SEQUENCE_LENGTH = 30           # Number of frames to analyze
prediction_interval = 5        # Seconds between predictions
max_num_hands = 2             # Maximum hands to detect
```

### Text-to-Speech Settings

The TTS engine can be customized:

```python
# Adjust speech rate
engine.setProperty('rate', 150)

# Change voice (if multiple voices available)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # Use first available voice

# Adjust volume
engine.setProperty('volume', 0.8)
```

## How It Works

1. **Hand Detection**: MediaPipe detects hand landmarks in real-time
2. **Feature Extraction**: Extracts 126 features (63 per hand × 2 hands max)
3. **Sequence Building**: Collects 30 consecutive frames of features
4. **Prediction**: Feeds the sequence to the trained model
5. **Output**: Displays prediction and announces it via TTS

## Troubleshooting

### Common Issues

**Camera not opening:**
- Check if another application is using the camera
- Try changing camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`

**Model files not found:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'sign_language_model.pkl'
```
- Ensure model files are in the same directory as the script
- Check file names match exactly

**Poor prediction accuracy:**
- Ensure good lighting conditions
- Keep hands clearly visible to the camera
- Make sure gestures match your training data
- Adjust MediaPipe confidence thresholds

**No audio output:**
- Check system volume and speakers
- Verify pyttsx3 installation: `pip install --upgrade pyttsx3`
- Test TTS separately:
  ```python
  import pyttsx3
  engine = pyttsx3.init()
  engine.say("Test")
  engine.runAndWait()
  ```

**High CPU usage:**
- Increase `prediction_interval` to reduce processing frequency
- Lower camera resolution if needed

## Model Requirements

Your trained model should:
- Accept input shape: `(1, 3780)` - flattened sequence of 30 frames × 126 features
- Be saved as a pickle file compatible with your ML library
- Have corresponding labels dictionary mapping class indices to gesture names

## Performance Tips

- **Lighting**: Ensure good, even lighting for better hand detection
- **Background**: Use a plain background to improve detection accuracy
- **Distance**: Stay 2-3 feet from the camera for optimal hand tracking
- **Gestures**: Make clear, distinct gestures that match your training data

## Contributing

To improve this system:
1. Add more gesture classes to your training data
2. Experiment with different sequence lengths
3. Implement gesture smoothing algorithms
4. Add confidence scoring for predictions
5. Create a GUI interface

## License

This project is open source. Please ensure you comply with the licenses of all dependencies (OpenCV, MediaPipe, etc.).

## Acknowledgments

- **MediaPipe**: Google's framework for building perception pipelines
- **OpenCV**: Computer vision library
- **pyttsx3**: Text-to-speech conversion library

---

**Note**: This README assumes you have already trained a sign language recognition model. If you need help with model training, consider using datasets like ASL Alphabet or creating your own gesture dataset.
