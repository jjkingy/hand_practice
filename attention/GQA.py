import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, groups=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % groups == 0, "num_heads must be divisible by groups"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.groups = groups
        self.head_dim = embed_dim // num_heads
        self.groups_heads = num_heads // groups #每个group的头数

        #创建QKV的投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.groups * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.groups * self.head_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        #x (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.size()

        #投影QKV
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        #调整维度，为注意力计算做准备
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.groups, self.head_dim).permute(0, 2, 3, 1)
        v = v.view(batch_size, seq_len, self.groups, self.head_dim).transpose(1, 2)

        #扩展kv以匹配组的头数
        k = k.unsqueeze(2).expand(-1, -1, self.groups_heads, -1, -1).contiguous()
        k = k.view(batch_size, self.num_heads, self.head_dim, seq_len)
        v = v.unsqueeze(2).expand(-1, -1, self.groups_heads, -1, -1).contiguous()
        v = v.view(batch_size, self.num_heads, seq_len, self.head_dim)

        #计算注意力分数
        atten_scores = torch.matmul(q, k) #(batch_size, num_heads, seq_len, seq_len)
        atten_scores = atten_scores / (self.head_dim ** 0.5)

        if key_padding_mask is not None:
            mask = key_padding_mask.view(batch_size, 1, 1, seq_len)
            atten_scores = atten_scores.masked_fill(mask == 0, float('-inf'))
            
        atten_weights = F.softmax(atten_scores, dim=-1)
        atten_weights = self.dropout(atten_weights)

        #加权求和
        output = torch.matmul(atten_weights, v) #(batch, num_heads, seq_len, head_dim)
        output = output.transpose(1,2).contiguous().view(batch_size, seq_len, -1)

        return self.out_proj(output)

#测试用例
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    groups = 2

    gqa = GroupQueryAttention(d_model, num_heads, groups)
    x = torch.randn(batch_size, seq_len, d_model)
    output = gqa(x)
    print(output.shape) #应该保持(2, 10, 512)
