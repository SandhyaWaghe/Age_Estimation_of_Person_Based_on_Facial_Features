
# Age and Gender Estimation from Facial Features

This project estimates a person's age and gender based on facial features using deep learning models and OpenCV.

## 📁 Project Structure

```
AgeGenderEstimation/
│
├── model_gender.h5         # Trained Keras model for gender detection
├── model_age.h5            # Trained Keras model for age estimation
├── predictgender.py        # Script to detect gender from an image
├── predictage.py           # Script to detect age from an image
├── demoagegnder.py         # Combined age & gender detection from a static image
├── videostream.py          # Real-time gender detection from webcam
├── videostreamage.py       # Real-time age detection from webcam
├── environment.yml         # Conda environment file with dependencies
└── README.md               # This file
```

## ⚙️ Setup

1. **Create a Conda Environment (recommended):**
```bash
conda env create -f environment.yml
conda activate age-gender-env
```

2. **If using pip (alternative):**
```bash
pip install tensorflow keras opencv-python numpy
```

## 🖼️ Running Static Image Detection

- **Gender only:**
```bash
python predictgender.py
```

- **Age only:**
```bash
python predictage.py
```

- **Age + Gender:**
```bash
python demoagegnder.py
```

Make sure to place your test image as `image2.jpg`, `image2.png`, or `kat2.jpg` in the same directory.

## 🎥 Real-time Detection via Webcam

- **Gender:**
```bash
python videostream.py
```

- **Age:**
```bash
python videostreamage.py
```

Press `q` to quit the webcam stream.

## 📌 Notes

- This project uses Haar cascades for face detection.
- Models were trained separately and must be in the same directory as the scripts.
- Input image files (`kat2.jpg`, `image2.jpg`, etc.) are expected to be in the root folder.

## 📬 Contact

For issues or improvements, feel free to contribute or raise an issue.

---

Enjoy experimenting with AI-powered age and gender estimation! 🧠📷
