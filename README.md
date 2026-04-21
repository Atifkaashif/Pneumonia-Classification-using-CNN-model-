 Pneumonia X-Ray Classification using CNN

 📌 Overview
This project uses a Convolutional Neural Network (CNN) to classify chest X-ray images as either **Normal** or **Pneumonia**. It helps in early detection of pneumonia by automating the diagnostic process using deep learning.

🎯 Objectives
- Build a CNN model for medical image classification
- Detect pneumonia from chest X-rays
- Achieve high accuracy and reliable predictions

 🧠 Model Architecture
- Convolutional Layers (Feature Extraction)
- Max Pooling Layers
- Fully Connected (Dense) Layers
- Activation Function: ReLU, Softmax/Sigmoid
- Loss Function: Binary Crossentropy

📂 Dataset
- Chest X-ray images (Normal & Pneumonia)
- Dataset split:
  - Training set
  - Validation set
  - Test set

⚙️ Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV (optional)

🚀 How to Run
1. Clone the repository:
   
   git clone https://github.com/Atifkaashif/Pneumonia-Classification-using-CNN-model-
   cd pneumonia-cnn

Install dependencies:

pip install -r requirements.txt

Run the model:

python train.py
📊 Results
Accuracy: ~XX%
Loss: ~XX
Confusion Matrix & Graphs included
📷 Sample Output

The model predicts whether the X-ray image shows:

Normal
Pneumonia
🔮 Future Improvements
Use Transfer Learning (ResNet, VGG)
Improve dataset size and quality
Deploy as a web or mobile app
👨‍💻 Author

M.Atif

📜 License

This project is open-source and available under the MIT License.
