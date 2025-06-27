# 🌸 Flower Image Classifier – Deep Learning with PyTorch

This project implements a deep learning image classifier that can identify 102 species of flowers. It uses transfer learning (VGG16 architecture) and is built in PyTorch. The classifier can be trained, saved, and used for inference via command-line scripts.

> 🎯 Imagine a mobile app that tells you what flower you're looking at. This project is the foundation for that!

---

## 📁 Project Structure

image-classification/
├── Completed part 1 notebook/
│ └── Flower Classifier.html # Interactive notebook exported to HTML
│
├── Part 2 files/
│ ├── model.py # Model architecture and checkpoint save/load
│ ├── train.py # Command-line training script
│ ├── util.py # Data transforms and helper functions
│ └── predict.py # Command-line prediction script
│
├── cat_to_name.json # Mapping from category index to flower name
├── README.md # This file
└── .gitignore # Files/directories to ignore in Git

yaml
Copy
Edit

---

## 🧠 Model Overview

- 🔧 **Base model**: VGG16 (pre-trained on ImageNet)
- 🔀 **Transfer Learning**: Freeze convolution layers, train custom classifier
- 🧱 **Classifier**: Fully connected layers + ReLU + Dropout + LogSoftmax
- 🧮 **Loss**: Negative Log Likelihood (`nn.NLLLoss`)
- ⚙️ **Optimizer**: Adam
- 📏 **Image Size**: 224 x 224 RGB

---

## ⚙️ Setup Instructions

### ✅ 1. Clone the repository

```bash
git clone https://github.com/your-username/image-classification.git
cd image-classification
✅ 2. Install dependencies
Install required libraries (preferably in a virtual environment):

bash
Copy
Edit
pip install torch torchvision numpy matplotlib pillow argparse
✅ 3. Download the dataset
If the flowers/ directory is missing, download and extract it:

bash
Copy
Edit
# For Windows (Git Bash or PowerShell)
curl -O https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
mkdir flowers
tar -xzf flower_data.tar.gz -C flowers
🏋️‍♀️ Training the Model
Run the train.py script from inside the Part 2 files/ directory:

bash
Copy
Edit
cd "Part 2 files"
python train.py --data_dir ../flowers --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 5 --gpu
🔧 Parameters
Argument	Description	Example
--data_dir	Path to the dataset folder	../flowers
--arch	Pretrained model architecture	vgg16
--learning_rate	Learning rate	0.001
--hidden_units	Hidden units in the classifier	512
--epochs	Training epochs	5
--gpu	Use GPU if available	flag only (no value)

🔍 Predicting Flower Species
Use the trained model to predict flower names from images:

bash
Copy
Edit
python predict.py ../flowers/test/1/image_06743.jpg checkpoint.pth --top_k 5 --category_names ../cat_to_name.json --gpu
🔧 Prediction Options
Argument	Description
image_path	Path to input image
checkpoint	Trained model checkpoint file
--top_k	Top K most probable classes (default: 5)
--category_names	JSON file mapping class idx to name
--gpu	Use GPU if available

🧪 Example Output
matlab
Copy
Edit
Image: image_06743.jpg

Top 5 Predictions:
1. Daffodil – 92.4%
2. Buttercup – 4.1%
3. Sunflower – 1.7%
...
📊 Performance
Metric	Value
Training Accuracy	~98%
Validation Accuracy	~87%
Test Accuracy	~85%

The model generalizes well on unseen flower images thanks to transfer learning and data augmentation.

🧼 .gitignore Setup
Ensure your .gitignore file excludes unnecessary files:

plaintext
Copy
Edit
# Ignore dataset and temp files
flowers/
flower_data.tar.gz

# Checkpoints and logs
*.pth
*.log

# Python junk
__pycache__/
*.pyc

# Jupyter artifacts
.ipynb_checkpoints/
*.html

# OS files
.DS_Store
🧠 Learnings & Extensions
✅ Gained experience with transfer learning using PyTorch

✅ Learned to modularize model code and build reusable CLI tools

📱 Future plan: Convert this into a mobile or web-based flower recognition app using Flask or Streamlit

🙌 Acknowledgements
Oxford 102 Flower Dataset

Udacity AI Nanodegree

PyTorch Docs

👩‍💻 Author
Devanshi Nikam
Final Year B.Tech | AI/ML Enthusiast | Passionate about solving real-world problems through intelligent systems

⭐ Like this project?
If this helped or inspired you, consider giving it a ⭐ and sharing your feedback!