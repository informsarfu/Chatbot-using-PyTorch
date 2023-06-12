# Chatbot using PyTorch

This project is focused on developing a chatbot using PyTorch, along with essential libraries such as NumPy, NLTK, and PyTorch. The chatbot is designed by implementing natural language processing (NLP) concepts, including tokenization, stemming, and the bag-of-words model. The chatbot model architecture includes two hidden layers and applies softmax activation after the final layer for training. The training data for the chatbot is scraped from a JSON intents file.

## Features

- **Natural Language Processing:** The chatbot utilizes NLP techniques to process user input, understand the context, and generate meaningful responses.
- **Tokenization:** The input text is tokenized to break it down into individual words or tokens, enabling the chatbot to understand the input more effectively.
- **Stemming:** Stemming is implemented to reduce words to their root form, allowing the chatbot to handle variations of words more accurately.
- **Bag-of-Words Model:** The bag-of-words model is used to represent text data, creating a numerical representation of words in the chatbot's vocabulary.
- **Machine Learning with PyTorch:** The chatbot leverages the power of PyTorch, a popular deep learning framework, to train and implement machine learning models for natural language understanding and response generation. The model architecture includes two hidden layers and applies softmax activation after the final layer for training.
- **Interactive Chat Interface:** The chatbot provides an interactive chat interface, allowing users to engage in conversations and receive responses in real-time.
- **Data Preprocessing:** The input data is preprocessed by applying cleaning techniques, removing noise, and transforming it into a format suitable for training the chatbot model.
- **Response Generation:** The chatbot generates responses by applying machine learning algorithms and using the bag-of-words model to identify the most appropriate response based on the user's input.
- **Data Scraping:** The training data for the chatbot is scraped from a JSON intents file, ensuring a diverse range of conversation examples for training.

## Installation

1. Clone the repository:

```
git clone https://github.com/informsarfu/Chatbot-using-PyTorch.git
```

2. Create and activate a new conda environment:

```
conda create --name chatbot-env python=3.8
conda activate chatbot-env
```

3. Install the required dependencies:

```
conda install numpy nltk pytorch
```

4. Download NLTK data for tokenization and stemming:

```python
import nltk
nltk.download('punkt')
```

## Usage

1. Run the chatbot:

```
python chatbot.py
```

2. Interact with the chatbot through the command line interface.
3. Enter your queries or statements and observe the chatbot's responses.

## Customization

Feel free to customize the chatbot according to your needs:

- **Training Data:** Modify the training data to enhance the chatbot's knowledge and response accuracy.
- **Model Architecture:** Experiment with different neural network architectures or pre-trained language models to improve the chatbot's performance.
- **Additional Features:** Extend the chatbot's capabilities by incorporating other NLP techniques, such as sentiment analysis or intent recognition.

## Contributing

Contributions to this project are welcome. Feel free to open issues or submit pull requests for any enhancements or bug fixes.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgements

- The project utilizes the power of PyTorch, NumPy, and NLTK libraries.
- The inspiration and initial code structure were based on the tutorial by [Tim](https://www.youtube.com/channel/UC4JX40jDee_tINbkjycV4Sg).
