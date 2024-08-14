Here's the updated `README.md` file incorporating the additional information about the Cat and Dog Image Classification using CNN, alongside the existing details:

```markdown
# Image Classification with VGG16 and EfficientNetB0

This project is a web application that classifies images of cats and dogs using a combination of the VGG16 model (served via FastAPI) and the EfficientNetB0 model (for local validation). The app is built using Streamlit and integrates various features to ensure accurate and user-friendly image classification.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Important Notes](#important-notes)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Overview

This repository contains a notebook that investigates the use of a Convolutional Neural Network (CNN) built with Keras for classifying images of cats and dogs. The notebook covers the following steps:
- Data preprocessing
- Model construction
- Training
- Evaluation
- Visualization of results

Significant overfitting is noted due to the large gap between training and validation accuracy, prompting the creation of a new notebook to tackle this problem.

## Libraries Used

- **Numpy**
- **TensorFlow**
- **Keras**
- **Matplotlib**
- **scikit-learn**

## Data

The datasets for training and testing contain images of cats and dogs.

## Model

A straightforward CNN architecture is employed for the image classification task.

## Results

While the model achieves high accuracy on the training set, it suffers from overfitting, highlighting the need for further refinement.

## Features

- **Image Upload**: Users can upload images in `jpg`, `jpeg`, or `png` formats.
- **File Name Validation**: The app checks if the uploaded image file name contains "cat", "ca", "dog", or "do" to determine if it can be processed.
- **Content Validation**: The app uses the `EfficientNetB0` model to check if the uploaded image is likely to contain a cat or dog.
- **Exception Handling**: If the content validation fails but the file name suggests it's a dog, the image is still processed.
- **FastAPI Integration**: The image is sent to a FastAPI endpoint that uses the VGG16 model to predict whether the image is a cat or dog.
- **User-Friendly Warnings**: The app prominently warns users to correctly label their images for successful processing.

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

### Clone the Repository

```bash
git clone https://github.com:ashedrack/Image-Classification-with-VGG16-and-EfficientNetB0.git
cd Image-Classification-with-VGG16-and-EfficientNetB0
```

### Install Required Packages

Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Running the FastAPI Server

1. Start the FastAPI server that serves the VGG16 model:

   ```bash
   uvicorn main:app --reload
   ```

   This will run the FastAPI server locally on `http://127.0.0.1:8000`.

### Running the Streamlit App

2. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

   The app will open in your default web browser, accessible at `http://localhost:8501`.

### Uploading and Classifying Images

- Upload an image of a cat or dog.
- Ensure the filename contains either "cat", "ca", "dog", or "do".
- The app will validate the content using `EfficientNetB0` and make a prediction using the FastAPI server.

## Project Structure

```plaintext
image-classification-app/
│
├── app.py               # Main Streamlit application script
├── main.py              # FastAPI application script
├── requirements.txt     # Python dependencies
└── README.md            # This README file
```

## Model Details

### EfficientNetB0

- **Purpose**: Used locally for validating the content of uploaded images to check if they contain a cat or dog.
- **Training**: Pre-trained on ImageNet, allowing it to identify a wide range of objects.

### VGG16 (Served via FastAPI)

- **Purpose**: Predicts whether the uploaded image is of a cat or a dog.
- **Training**: The VGG16 model is also pre-trained on ImageNet and is well-known for its deep architecture, making it effective for image classification tasks.

## Important Notes

- **File Naming**: Ensure your image filenames include "cat", "ca", "dog", or "do" for them to be accepted by the application.
- **Dog Image Exception**: If an image fails content validation but the filename suggests it is a dog, the image will still be processed.
- **Content Validation**: EfficientNetB0 provides initial validation to ensure the image is suitable for prediction, minimizing the chances of incorrect classifications.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## Acknowledgements

- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

Thank you for using the Image Classification App! We hope it helps you with your projects. Feel free to explore the notebook and provide any feedback or suggestions for improvement.
```

### Explanation:

- **Overview**: Added a new section detailing the investigation and findings from the CNN notebook.
- **Libraries Used**: Lists the libraries used in the notebook for clarity.
- **Data and Model**: Provides information on the data and model used in the CNN notebook.
- **Results**: Highlights the issue of overfitting observed with the CNN model.

This `README.md` now includes comprehensive information about both the web application and the CNN notebook, giving users and contributors a complete view of the project.