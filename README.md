# Word Embeddings for Entity-annotated Texts

This repository provides a reference implementation of the paper [Word Embeddings for Entity-annotated Texts](https://arxiv.org/pdf/1902.02078.pdf) as well as links to the data.

A long-standing challenge for research in computer science is the understanding of written text and extraction of useful information from it. These distributed representations or so-called word embeddings, map words of a vocabulary to a dense vector, such that words with closer meanings are mapped to the nearby points and the similarity between them is computed based on their distance in the embedding space. Traditional word embeddings, despite being good at capturing semantics, have some drawbacks. They treat all words equally as terms and cannot be directly used to represent named entities. Disregarding the named entities while generating a word embeddings creates several challenges for downstream tasks that use them as input.
In this work, we address the problems of term-based models by generating embeddings for named entities as well as terms using an annotated corpus using two approaches: 

To naively include entities in our models, we train the well-established word embedding models on a corpus, annotated with named entities. 
To better capture the entity-entity relations, we take advantage of the graph representation of the corpus, and embed the nodes of co-occurrence graphs extracted from the annotated text. 
To enhance the performance of our models, we try a wide range of word and graph embedding techniques and compare them against word embedding models trained on raw text.

## Datasets

### Training corpus 
For training, we use 209,023 news articles from English-speaking news outlets,
collected from June to November 2016 by [Spitz and Gertz](https://dbs.ifi.uni-heidelberg.de/files/Team/aspitz/publications/Spitz_Gertz_2018_Entity-centric_Topic_Extraction.pdf). For entity embeddings we annotated the corpus using [ Ambiverse](https://github.com/ambiverse-nlu). Although the full dataset is not available in this repository, we provided a small example corpus in "test_corpus" folder. The `"corpus_raw.txt"` file the first 100 line of our raw corpus, containing the news articles without the annotation. The annotated version can be found under `"corpus_annotated.txt"` in the same folder, where each line is pre-processed and entities are replaced with their unique identifier as described in the paper. 

### Graph 
For the graph-based methods the LOAD network was extracted from the same corpus, the edge list can be found [here](https://dbs.ifi.uni-heidelberg.de/resources/entity-embeddings/). The `"node_list.txt"` file contains the label for each node along with the unique identifier. `"edge_list_raw.txt"` contains the edge list where each node is the unique identifier from the LOAD network. Since embeddings need an index file that maps the words to their row in the embedding matrix, we created a second edge list with the indexes as nodes in `"edge_list.txt"`.

### Test data
The relevant test dataset for the tasks of Word Similarity, Analogy and Clustering can be found under the "test_data" folder There exist two versions of each dataset, one which is the original version and the second one which was tailored to be used by the models. Specifically, the words that do not exist in our corpus were removed and the unique identifier for each word was added. 

## Pre-trained Models
The results presented in the paper are the average result between 10 embedding model trained using the same hyperparameters on the test datasets. We provide one pre-trained model per method [here](https://dbs.ifi.uni-heidelberg.de/resources/entity-embeddings/), for the exact replication of the results in the paper however, all 10 models are required. 

## Settings File 

To train your own models or run test the settings file should be edited. Below there is a description of each field and its role: 

- `MODE`: (Train/Test/Batch_Test) Shows the current mode of the program could be "Train" to train the models,"Test" tests the model, "Batch_Test" is when multiple models with the same parameters have been trained the average test result is required, all the models should be in the same folder and the folder path should be given as 'SAVE_FOLDERPATH' 
- `SAVE_FOLDERPATH`: The path to a folder to save the current model files or to read the model in Test mode 
- `EDGELIST_PATH`: Path to the file for the edge list of a co-occurrence network 
- `NODELIST_PATH`: Path to the file for the node list of a co-occurrence network 
- `EMBEDDING_TYPE`: (GloVe/Word2Vec/DeepWalk/VERSE) The type of the embedding to be trained or tested "GloVe", "Word2Vec", "DeepWalk", "VERSE"
- `CORPUS_PATH`: Path to the file for the textual corpus to create glove or word2vec

- `TEST_BATCH_NUMBER`: If we are doing batch test how many models are in the batch
- `TEST_DATA_PATH`: The location of the test dataset in the form of CSV files, separated by a tab. The dataset should be relevant for the test type, for the required columns please look at the test_data folder for examples 
- `TEST_MODE`: (Clustering\Analogy\WordSimilarity) Type of test to be performed. 
- `TEST_ON_RAW_TEXT`: If set to true all tests will not use the unique ids to find embeddings (no entity embedding) but the raw form of the text. should be used for models trained on the raw data without annotations. 
- `ENITY_CENTRIC_TEST`: If the set is True in the test case of analogies we limit the results to only the entities of the same type as the question. 
- `EMBEDDING_SIZE`: Embedding dimensions
- `MODEL_NUMBER`: To use batch test we number each model, The name of the saved models is the combination of their parameters, the model Number allows us to save multiple models with the same parameter in the same folder 
- `NUM_EPOCH`: Number of epochs for training 
- `LEARNING_RATE`: Learning rate
- `BATCH_SIZE`: Batch size for batch gradient descend
- `NUM_BATCH`: Number of examples in a batch 
- `PROXIMATY`: The proximity for the DeepWalk it defines which function of the edge weights to use and can take values "Plain"-> no change to weights, "log"-> log(weight) and "sqrt"-> sqrt(weight)
- `NUM_NEGATIVE_SAMPLES`: Number of negative examples to be considered for negative sampling
- `WINDOW_SIZE`: Window size for the word2vec model 
- `MAX_WEIGHT_CAP`: Maximum weight cap parameter for the GloVe model 
- `POWER_SCALING`: The power scaling for the weighing function of glove 
- `MIN_COUNT`: Minimum number of occurrence for a word in the corpus to be included in the model 
- `NUM_THREAD`: Number of threads for multi-threading 
- `NUM_WALKS`: Number of random walks in the deep walk based model
- `LENGTH_WALK`: Length of random walks in the deep walk based model


An example settings file can be found in 'settings/settings.ini'

## Usage 

After changing the settings file, run the following command: 

```
python main.py 
```

To train the verse model on the data, please refer to their GitHub repository [VERSE](https://github.com/xgfs/verse) and use the C++ code to train the model using the edge list of a co-occurrence network. To evaluate their model using our code, use the convertor.py in the verse package to convert the embeddings into numpy. Rename the embedding to 'emb.bin' and place them along with the dictionary 'dicts.pickle' in a folder. The folder path should be given as 'SAVE_FOLDERPATH' in the settings file. 

## Required packages  

The word2vec model uses the `gensim` package and the GloVe model uses the `glove-python` package. 
The code for the verse model can be obtained from [VERSE](https://github.com/xgfs/verse) and the original DeepWalk implementation is available in [DeepWalk](https://github.com/phanein/deepwalk), for our model we modified the code to meet our needs.

A full list of required python packages is provided in the `requirements.txt` file.

To install the requirements for pip run:
```
# Using pip
pip install -r requirements.txt

# Using conda
conda install --file requirements.txt
```


## Citation 
If you use the code or the datasets, please consider citing the paper:
```
@inproceedings{DBLP:conf/ecir/AlmasianSG19,
  author    = {Satya Almasian and
               Andreas Spitz and
               Michael Gertz},
  title     = {Word Embeddings for Entity-Annotated Texts},
  booktitle = {Advances in Information Retrieval - 41st European Conference on {IR}
               Research, {ECIR} 2019, Cologne, Germany, April 14-18, 2019, Proceedings,
               Part {I}},
  pages     = {307--322},
  year      = {2019},
  url       = {https://doi.org/10.1007/978-3-030-15712-8\_20},
  doi       = {10.1007/978-3-030-15712-8\_20},
}
```


## MIT License

Copyright (c) 2019 Satya Almasian

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

