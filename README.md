# Husky vs. Wolf Image Classifier

This project is a binary image classification pipeline designed to distinguish between images of huskies and wolves. Using machine learning techniques, the system processes images of varying sizes, applies preprocessing steps, and predicts the class (husky or wolf) with high accuracy. The implementation leverages a Logistic Regression model, augmented data, and a streamlined workflow suitable for real-world applications.

## Project Overview

The goal of this project is to build an efficient and accurate classifier to identify whether an image depicts a husky or a wolf. The pipeline includes data loading, preprocessing, model training, and evaluation, making it a robust example of image classification.

### Features
- **Data Loading**: Loads images from a structured dataset and verifies them with labels.
- **Preprocessing**: Resizes, normalizes, and augments images to enhance model performance.
- **Model**: Trains a Logistic Regression classifier with hyperparameter tuning.
- **Evaluation**: Provides detailed metrics including accuracy, classification report, and confusion matrix.
- **Scalability**: Includes a placeholder for testing on new datasets.

## Implementation Details

### Dataset
- **Structure**: Organized as `data/split/class/images` (e.g., `train/husky`, `train/wolf`, `test/husky`, `test/wolf`).
- **Loading**: Images are loaded using OpenCV, resized to 256x256 pixels, and normalized to a [0, 1] range.
- **Splitting**: An 80/20 train-test split is applied with stratification to ensure balanced classes.

### Preprocessing
- **Resizing**: Uniformly scales all images to 256x256 pixels.
- **Normalization**: Adjusts pixel values from [0, 255] to [0, 1].
- **Augmentation**: Uses the `albumentations` library to apply:
  - Rotation (up to 20 degrees, 50% probability)
  - Horizontal flipping (50% probability)
  - Random resized cropping (scale 0.8-1.0, 50% probability)

### Classifier
- **Model**: Logistic Regression, chosen for its simplicity and effectiveness.
- **Hyperparameter Tuning**: Grid search with 5-fold cross-validation over:
  - `C`: [0.01, 0.1, 1, 10, 100] (regularization strength)
  - `penalty`: ['l1', 'l2'] (regularization type)
  - `max_iter`: [100, 500, 1000] (maximum iterations)
- **Best Parameters**: `{'C': 0.01, 'max_iter': 100, 'penalty': 'l2'}`.
- **Performance**: Achieves 95% accuracy on the test set.

### Evaluation
- **Metrics**:
  - Accuracy: 0.95
  - Classification Report:
    ```
                precision    recall  f1-score   support
    husky       1.00      0.90      0.95        10
    wolf        0.91      1.00      0.95        10
    accuracy                        0.95        20
    ```
  - Confusion Matrix:
    ```
    [[ 9  1]
     [ 0 10]]
    ```
- **Test Placeholder**: A function to evaluate new test data with a customizable path.

### Tools and Libraries
- **Python**: 3.x
- **Dependencies**:
  - `opencv-python` (cv2): Image loading and processing
  - `numpy`: Array operations
  - `matplotlib`: Visualization
  - `scikit-learn`: Machine learning and evaluation
  - `albumentations`: Image augmentation
  - `jupyter`: Interactive notebook environment

## Setup Instructions

### Prerequisites
- Python 3.x
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AhmedSamir423/husky-vs-wolf-classifier.git
   cd husky-vs-wolf-classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset in the following structure:
   ```
   data/
   ├── train/
   │   ├── husky/
   │   └── wolf/
   └── test/
       ├── husky/
       └── wolf/
   ```

### Running the Project
- Launch the Jupyter notebook:
  ```bash
  jupyter notebook Husky_vs_Wolf_Classifier.ipynb
  ```
- Execute all cells to run the pipeline from data loading to evaluation.

## Files
- `Husky_vs_Wolf_Classifier.ipynb`: Main notebook with the complete pipeline.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file.

## Usage
1. Place your husky and wolf images in the `data` directory as described above.
2. Run the notebook to train the model and evaluate its performance.
3. Use the `evaluate_test` function with a custom test path to classify new images.

## Results
The classifier achieves a 95% accuracy on the test set, with strong precision and recall for both husky and wolf classes. The confusion matrix shows minimal misclassifications, indicating robust performance.

## Future Enhancements
- Add **Grad-CAM** to visualize the model's decision-making process.
- Incorporate a convolutional neural network (CNN) for potentially higher accuracy.
- Expand the dataset with more diverse images to improve generalization.

## Contributing
Feel free to fork this repository, submit pull requests, or open issues for suggestions and improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

