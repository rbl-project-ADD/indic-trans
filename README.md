# Indic Language Translation

This project leverages the `transformers` library from Hugging Face to perform translations between English and various Indian languages.

## Description

The project utilizes `indictrans` models from the AI4Bharat project, which are capable of translating between English and several Indian languages. The repository includes a script, `indic-trans.py`, demonstrating how to use these models for translation tasks.

## Installation

First, clone the repository to your local machine:

```sh
git clone https://github.com/rbl-project-ADD/indic-trans.git
```

Navigate to the project directory:

```sh
cd indic-trans
```

## Activate the venv

On macOS and Linux:

```sh
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```sh
python -m venv venv
.\venv\Scripts\activate
```

Install the required dependencies:

```sh
pip install -r requirements.txt
```

This will install the `transformers` library, among other dependencies.

## Usage

To use the translation models, you first need to create a tokenizer and a model:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "ai4bharat/indictrans2-indic-indic-1B"  # for Indian languages translation
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
```

You can then use the tokenizer and model to translate sentences:

```python
from IndicTransTokenizer import IndicProcessor

ip = IndicProcessor(inference=True)
input_sentences = ["जब मैं छोटा था, मैं हर रोज़ पार्क जाता था।"]

# Tokenize the input sentences
inputs = tokenizer(input_sentences, return_tensors="pt", padding=True, truncation=True)

# Perform translation
outputs = model.generate(**inputs)

# Decode the translated sentences
translated_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Print the translated sentences
for sentence in translated_sentences:
    print(sentence)
```

This example demonstrates how to tokenize input sentences, perform the translation, and decode the translated sentences.

## To run the program (make sure u have activated the venv)

```sh
python indic-trans.py
```

