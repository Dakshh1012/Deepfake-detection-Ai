
Deepfake Detection AI System
Live Demo: https://youtu.be/GUIBKSDPuLY

This project presents a modular, multi-modal deepfake detection pipeline designed to identify synthetic video manipulations with high precision. It incorporates state-of-the-art techniques in facial analysis, audio-visual synchronization, frame consistency evaluation, and image manipulation detection. The backend is implemented using Flask, and the system integrates models developed in PyTorch, TensorFlow, and other advanced frameworks to perform real-time detection.

1. Detection Modules and Frameworks
1.1 Face Distortion Detection – MobileNetV2
Model: MobileNetV2 (fine-tuned for binary classification: Real vs Fake)

Purpose: Detect unnatural distortions in facial features commonly introduced by deepfake generation pipelines.

Key Details:

Model loaded via torch.hub, pre-trained on ImageNet.

Final classification head modified for binary output.

Input image resolution: 224×224.

Normalization:

normalized_pixel = (pixel_value - mean) / std

 
1.2 Frame Anomaly Detection – Meso4
Model: Meso4 CNN architecture

Purpose: Detect spatial inconsistencies across video frames.

Key Details:

Input shape: 112×112×3.

Network consists of 4 convolutional layers with Batch Normalization and LeakyReLU.

Binary classification head.

Normalization: Pixel values scaled to [0, 1].

Frame anomaly score calculated via cosine similarity between feature vectors of consecutive frames:

cosine_similarity=1−cos(v1,v2)
Anomaly Threshold: 0.85

1.3 Facial Landmark Analysis – MediaPipe FaceMesh
Framework: MediaPipe

Purpose: Identify distortions in facial proportions using landmark-based facial geometry.

Key Details:

Configured with:

static_image_mode = True

max_num_faces = 1

min_detection_confidence = 0.5

min_tracking_confidence = 0.5

Extracts 468 facial landmarks per frame.

Used to compute geometric ratios and detect unnatural facial warping.

1.4 Audio-Visual Inconsistency Detection – Wav2Vec2
Model: Wav2Vec2 (self-supervised speech model)

Purpose: Detect temporal inconsistencies between lip movements and speech patterns.

Key Details:

Audio extracted using librosa, converted into Mel spectrograms.

Audio embeddings extracted via Wav2Vec2.

Visual embeddings obtained from video frames.

Cosine similarity between modalities used to quantify mismatch.

1.5 Expression & Manipulation Analysis – DeepFace
Framework: DeepFace

Purpose: Analyze facial expressions and detect synthetic facial attributes.

Key Details:

Emotion detection used to identify unnatural or inconsistent expressions.

Manipulation detection based on inconsistencies in pixel-level facial features.

2. Preprocessing Pipelines
2.1 Frame Processing
Video decomposed into individual frames using OpenCV.

Frames resized and normalized per model-specific requirements.

2.2 Audio Processing
Audio extracted from video via librosa.

Transformed into spectrograms for embedding extraction (Wav2Vec2).

2.3 Feature Vector Construction
Feature vectors obtained from each model (MobileNetV2, Wav2Vec2, etc.).

Standardized normalization applied.

Cosine similarity used for cross-modal and temporal comparisons.

3. Evaluation Metrics and Thresholds
3.1 Face Distortion Detection
Detection Rate: Percentage of frames with successfully detected faces.

Distortion Rate: Percentage of frames with abnormal facial proportions.

3.2 Frame Anomaly Detection
Anomaly Rate: Proportion of frames with low inter-frame similarity.

Threshold: Cosine similarity < 0.85 is considered anomalous.

3.3 Audio-Video Synchronization
Mismatch Score: Derived from cosine distance between audio and video embeddings.

Confidence Score: Likelihood estimate of a synchronization anomaly.

4. Technology Stack
Backend: Flask (Python)

ML Frameworks: PyTorch, TensorFlow, MediaPipe

Supporting Libraries: OpenCV, librosa, torchvision, facenet-pytorch, transformers

Models Used:

MobileNetV2

Meso4

Wav2Vec2

MediaPipe FaceMesh

DeepFace

