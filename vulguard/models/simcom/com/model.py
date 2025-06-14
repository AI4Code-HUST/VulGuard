import torch.nn as nn
import torch
import torch.nn.functional as F

class DeepJITModel(nn.Module):
    def __init__(self, params):
        super(DeepJITModel, self).__init__()
        self.params = params

        V_msg = params["vocab_msg"]
        V_code = params["vocab_code"]
        Dim = params["embedding_size"]
        Class = params["class_num"]

        Ci = 1  # input of convolutional layer
        Co = params["num_filters"]  # output of convolutional layer
        Ks = params["filter_sizes"]  # kernel sizes

        # CNN-2D for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])

        # CNN-2D for commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_file = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co * len(Ks))) for K in Ks])

        # other information
        self.dropout = nn.Dropout(params["dropout_rate"])
        self.fc1 = nn.Linear(2 * len(Ks) * Co, params["hidden_size"])  # hidden units
        self.fc2 = nn.Linear(params["hidden_size"], Class)
        self.sigmoid = nn.Sigmoid()

    def forward_msg(self, x, convs):
        # note that we can use this function for commit code line to get the information of the line
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x

    def forward_code(self, x, convs_line, convs_hunks):
        n_batch, n_file = x.shape[0], x.shape[1]
        x = x.reshape(n_batch * n_file, x.shape[2], x.shape[3])

        # apply cnn 2d for each line in a commit code
        x = self.forward_msg(x=x, convs=convs_line)

        # apply cnn 2d for each file in a commit code
        x = x.reshape(n_batch, n_file, self.params["num_filters"] * len(self.params["filter_sizes"]))
        x = self.forward_msg(x=x, convs=convs_hunks)
        return x

    def forward(self, msg, code):
        x_msg = self.embed_msg(msg)
        x_msg = self.forward_msg(x_msg, self.convs_msg)

        x_code = self.embed_code(code)
        x_code = self.forward_code(x_code, self.convs_code_line, self.convs_code_file)

        x_commit = torch.cat((x_msg, x_code), 1)
        x_commit = self.dropout(x_commit)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out