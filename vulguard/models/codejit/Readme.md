# CodeJIT
This is the replication of CodeJIT.

## Training and Evaluating

## Requirements
In order to train GNN models, you need to install these required libraries.
```
# Install required packages.
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
```

## Instruction to train word2vec model
```
python Main_Word2Vec.py --vocab_file_path="vocab/word2vec.vocabulary.txt" --vector_length=32 --model_dir="model" --model_name="w2v"
```
The pretrained word2vec model is in the [Model](model/Model) folder.

## Instruction to embed features of nodes and edges of the graphs
```
python Main_Graph_Embedding.py --node_graph_dir="Data/Graph/node" --node_graph_dir="Data/Graph/edge" --label=1 --embedding_graph_dir="Data/embedding" 
```

## Instruction to train and test GNN models
```
python Main_VULJIT_Detection.py --graph_dir='Data/Embedding_CTG'  --train_file='Data/data_split/train_time_id.txt' --test_file='Data/data_split/test_time_id.txt'  --model_dir='Model'  --model_name="rgcn" 
```




