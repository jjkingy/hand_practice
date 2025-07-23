import torch
from torch import nn

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dk = 1.0 / (self.head_dim ** 0.5)

        #初始化Q K V矩阵
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        #输出线性层
        self.out_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, hidden_state, attention_mask=None):
        #hidden_state是嵌入层传进来的
        batch_size = hidden_state.size()[0]

        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)

        q = self.split_head(query)
        k = self.split_head(key)
        v = self.split_head(value)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.dk
        
        if(attention_mask != None):
            attention_scores = attention_scores.mask_fill(attention_mask==0, float('-inf'))

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        output = torch.matmul(attention_probs, v)

        ##拼接输出的注意力
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.head_dim*self.num_heads)
        output = self.out_linear(output)
        return output
        
    #分头函数
    def split_head(self, x):
        batch_size = x.size()[0]
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2) # (batch_size, num_heads, seq_len, head_dim)