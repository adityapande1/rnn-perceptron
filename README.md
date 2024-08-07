# Noun Chunking Using RNN Perceptron

#### NOTE
This project is done as a part of course assignment for<br>
 Course : __CS772: Deep Learning for Natural Language Processing__ <br>
 Instructor :  __Prof. Pushpak Bhattacharyya__ at __IIT Bombay__

## Description
The project aims at performing noun chunking in a text sequence given its Part of Speech (POS) tags. The task is achieved using RNN Perceptron and Sigmoid models that are implemented from scratch.

#### 1. Noun Chunking Using POS Tags
Chunking is a crucial process in Natural Language Processing (NLP) for extracting meaningful units from text that involves identifying and segmenting multi-token sequences, such as noun phrases, within sentences using Part-Of-Speech (POS) tags. 
It involves labeling each word in a sentence with its corresponding part of speech, such as noun (NN), verb (VB), adjective (JJ), determiner (DT), etc.<br><br>
Once the words are tagged, chunking algorithms group sequences of words into chunks based on their POS tags. Noun phrase chunking specifically targets sequences of words that form noun phrases. For example sentence, a noun phrase chunker would identify and extract "The quick brown fox" and "the lazy dog" as noun phrases. This process helps in isolating and understanding the structure of noun phrases within sentences, facilitating further analysis and processing of the text.


#### 2. Using RNNs for Noun Chunking
Recurrent Neural Networks (RNNs) are well-suited for sequential data and can be effectively used for tasks like POS tagging and chunking.__In this project the user is provided with POS tagged data and the model. The RNN takes the sequence of POS tags (and potentially the original words) as input and outputs chunk tags that indicate the boundaries of noun phrases.__

## Try the model here [huggingface demo](https://huggingface.co/spaces/vivek9/CS772_Assignment2)
__NOTE__: Please restart the space if prompted

## Some Cool Examples
### Example 1: Noun Chunking using model
![Cool Demo](https://github.com/adityapande1/rnn-perceptron/blob/main/media/gifs/one.gif)

### Example 2 : Noun Chunking using model
![Cool Demo](https://github.com/adityapande1/rnn-perceptron/blob/main/media/gifs/two.gif)
 

## File Desciption
#### ./models
1. `PRNN.py` file contains the Peceptron RNN implementation from scratch
2. `PRNNSigmoid.py` file contains the Sigmoid RNN implementation from scratch

#### ./utils
1. `PRNN_utils.py` file contains the Peceptron RNN utils
2. `PRNNSigmoid.py` file contains the Sigmoid RNN utils

#### ./notebooks
1. `mainPerceptron.ipynb` file contains all the step by step training and testing of the Perceptron RNN model.
2. `mainSigmoid.ipynb` file contains all the step by step training and testing of the Sigmoid RNN model.
3. `test.ipynb` file contains the inference procedure for testing the model.

#### ./presentation.pptx
Containes the __presentation__ of the assignment used during the evaluation.

#### ./data
Containes the __cross validation__ , __hyperparameters__ and conditional data __json__ used in different parts of the model training and inference.
#### ./media
Contains some of the __videos__ and __gifs__ obtained from model inference.<br>
The __images__ folder contains the _filter tables_  and _architecture_ of the net

#### ./derivations
Contains the __handwritten derivations__ of the model from scratch, it forms the basis of the model implementation.

# My Project
## Overview

<p align="left">
  <img src="https://github.com/adityapande1/rnn-perceptron/blob/main/media/images/architecture.png" alt="Project Logo" style="float: left; margin-right: 10px;" />
  This is the architecture of the recurrent perceptron. A tag can be one of `[OT, NN, JJ, DT]` This is fed into the network as a 1 hot vector.
  The detailed inplementation can be found in `models` folder. The idea is to design an RNN classifier using the architure above.
</p>

<p align="left">
  <img src="https://github.com/adityapande1/rnn-perceptron/blob/main/media/images/der1.png" alt="Project Logo" style="float: left; margin-right: 10px;" />
  This is the derivation that forms the theoritical foundation of __BACK PROPOGATION THROUGH TIME__ algorith implemented in this project.
</p>

<p align="left">
  <img src="https://github.com/adityapande1/rnn-perceptron/blob/main/media/images/green_conditions.png" alt="Project Logo" style="float: left; margin-right: 10px;" />
  Once the architecture is setup we analyse different conditions on the data to see how close a neural net model is to a deterministic model. The table shows how samples can be filtered on basis of human rules. The analysis is properly elaborated in the notebooks provided
</p>

