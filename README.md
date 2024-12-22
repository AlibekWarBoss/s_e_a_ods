# README.md

```markdown
# Project: Sentiment Classification using Bi-LSTM+CNN with Attention

This project involves building a sentiment classification model using a hybrid architecture that combines Bi-LSTM, CNN, and an Attention mechanism. The model is trained and evaluated on a combined dataset consisting of IMDb movie reviews and Amazon customer reviews.

## Overview

The goal of this project is to classify text data into positive or negative sentiment categories. To address challenges such as class imbalance and dataset diversity, the project employs:

1. **Hybrid Architecture**: Combines Bi-LSTM for capturing sequential dependencies, CNN for local feature extraction, and an Attention mechanism for focusing on relevant parts of the text.
2. **Data Augmentation**: Uses the SMOTE technique to address class imbalance.
3. **Comprehensive Evaluation**: Evaluates the model using metrics such as accuracy, precision, recall, and F1-score.

## Features

- **Dataset Combination**: Merges IMDb movie reviews and Amazon customer reviews.
- **Text Preprocessing**: Implements stemming, lemmatization, and stopword removal.
- **SMOTE**: Balances the dataset by generating synthetic samples for underrepresented classes.
- **Attention Mechanism**: Enhances model interpretability by focusing on the most relevant parts of the input sequence.

## Installation

To run this project locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/username/repository-name.git
   ```
2. Navigate to the project directory:
   ```bash
   cd repository-name
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the Dataset**:
   - Ensure the files `train_data.csv`, `test_data.csv`, and `Amazon Review Data Web Scrapping.csv` are in the root directory.
   - Run the `dataset.py` script to preprocess and combine datasets:
     ```bash
     python dataset.py
     ```

2. **Train the Model**:
   - Execute the `main.py` script to train the Bi-LSTM+CNN model:
     ```bash
     python main.py
     ```

3. **Evaluate the Model**:
   - Use the `metrics.py` script to evaluate the model performance:
     ```bash
     python metrics.py
     ```

## Results

- **F1-Score**: Achieved 91.88% on the test dataset.
- **Other Metrics**:
  - Accuracy: 91.88%
  - Precision: 91.70%
  - Recall: 91.95%

## File Structure

- `dataset.py`: Prepares and combines datasets, applies text preprocessing.
- `main.py`: Contains the Bi-LSTM+CNN model with an Attention mechanism and training logic.
- `metrics.py`: Includes functions to evaluate the model’s performance and display metrics.
- `requirements.txt`: Lists all the dependencies required for the project.
- `train_data.csv` and `test_data.csv`: IMDb dataset files for training and testing.
- `Amazon Review Data Web Scrapping.csv`: Additional Amazon review data for diversity.

## Technologies Used

- **Python**: Programming language.
- **TensorFlow/Keras**: For implementing the Bi-LSTM+CNN model.
- **NLTK**: For text preprocessing (lemmatization, stemming, stopword removal).
- **SMOTE**: For balancing the dataset.
- **Scikit-learn**: For evaluation metrics and data splitting.

## References

1. Beakcheol Jang, et al. (2024). Bi-LSTM model to increase accuracy in text classification. *Some Journal*, pages 45–56.
2. Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
3. Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.
4. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
5. Kim, Y. (2014). Convolutional neural networks for sentence classification. *arXiv preprint arXiv:1408.5882*.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- The creators of the IMDb and Amazon review datasets.
- Open-source libraries and tools, including TensorFlow, Keras, and NLTK.
```