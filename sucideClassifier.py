
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel



class TextClassificationModel(nn.Module):
    def __init__(self,  embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.embed_dim = self.bert.config.hidden_size
        self.config = self.bert.config
        # self.device = device

        print(f"embedding dimension {embed_dim}")
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.02
        nn.init.normal_(self.fc1.weight, std=initrange)
        nn.init.normal_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=initrange)
        nn.init.normal_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=initrange)
        nn.init.normal_(self.fc3.bias, 0)
        nn.init.normal_(self.fc4.weight, std=initrange)
        nn.init.normal_(self.fc4.bias, 0)



    def forward(self, input_ids=None, attention_masks=None):

        vec=self.bert(input_ids = input_ids, attention_mask = attention_masks, return_dict=True, output_hidden_states=True, output_attentions=False)
        vec = vec['last_hidden_state']
        vec = vec[:, 0, :]

        vec = vec.view(-1, self.embed_dim)

        x = self.fc1(vec)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.softmax(x, dim=1)
