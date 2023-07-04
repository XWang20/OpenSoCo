import os
import torch

import bmtrain as bmt

from model_center.model import Bert, Roberta
from model_center.layer import Linear

class RobertaModel(torch.nn.Module):
    def __init__(self, config, model_path, label_num):
        super().__init__()
        bmt.print_rank(f"Loading config...")
        self.model = Roberta(config)
        bmt.print_rank(f"Loading roberta from model path {model_path}...")
        bmt.load(self.model, model_path)
        bmt.print_rank(f"Initializing dense layer...")
        self.dense = Linear(config.dim_model, label_num)
        bmt.print_rank(f"Initializing parameters...")
        bmt.init_parameters(self.dense) # init dense layer

    def forward(self, *args, **kwargs):
        pooler_output = self.model(*args, **kwargs, output_pooler_output=True).pooler_output
        x = self.dense(pooler_output)
        return x

class BertModel(torch.nn.Module):
    def __init__(self, config, model_path, label_num):
        super().__init__()
        self.model = Bert(config)
        bmt.load(self.model, os.path.join(model_path, "model.pt"))
        self.dense = Linear(config.dim_model, label_num)
        bmt.init_parameters(self.dense) # init dense layer

    def forward(self, *args, **kwargs):
        pooler_output = self.model(*args, **kwargs, output_pooler_output=True).pooler_output
        x = self.dense(pooler_output)
        return x
