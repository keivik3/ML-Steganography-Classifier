# ML-Steganography-Classifier
A Python-based project for detecting **stego images** (images with hidden data) versus **clean images** using machine learning.   The program extracts multiple image features (color histograms, HOG features, brightness, and saturation) and trains a **K-Nearest Neighbors (KNN)** classifier to classify new images as either **clean** or **stego**.

---

## 📂 Repository Structure
```
├── clear/ # Directory containing clean images
├── stego_us/ # Directory containing stego images
├── yo/ # Directory for test images
├── main.py # Main script (this code)
└── README.md # Documentation
```

---

## 🚀 Features
- **Feature Extraction**:
  - Hue histogram from HSV color space  
  - Histogram of Oriented Gradients (HOG)  
  - Brightness mean & standard deviation  
  - Saturation mean & standard deviation  

- **Classification**:  
  Trains a **KNN classifier (k=3)** using extracted features.  

- **Prediction**:  
  Given a new image path, predicts whether it is **clean** or **stego**.  

---

## ⚙️ Installation

Clone the repository and install dependencies:

```
git clone https://github.com/keivik3/ML-Steganography-Classifier.git
cd Image-Steganography-Classifier
pip install -r requirements.txt
```
requirements.txt : 

numpy
opencv-python
scikit-learn
scikit-image

## ▶️ Usage

Place your clean images inside the clear/ folder.

Place your stego images inside the stego_us/ folder.

(Optional) Add test images inside the yo/ folder.

Run the script:
```
python main.py
```

Enter the path to an image when prompted:
```
Enter the path to the image: path/to/image.jpg
Predicted class: clean
```

## 🧪 Example

```
Enter the path to the image: yo/test1.png
Predicted class: stego
```
## 📜 License

This project is open-source under the MIT License.
