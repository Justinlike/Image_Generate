import torch
import math

class Embed(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.embed = torch.nn.Embedding(49408, 768)
        self.pos_embed = torch.nn.Embedding(77, 768)

        self.register_buffer("pos_ids", torch.arange(77).unsqueeze(dim=0))

    def forward(self, input_ids):
        # input_ids -> [b, 77]

        # [b, 77] -> [b, 77, 768]
        embed = self.embed(input_ids)

        # [1, 77] -> [b, 77, 768]
        pos_embed = self.pos_embed(self.pos_ids)

        # [b, 77, 768]
        return embed + pos_embed


class Atten(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(768, 768)
        self.k = torch.nn.Linear(768, 768)
        self.v = torch.nn.Linear(768, 768)
        self.out = torch.nn.Linear(768, 768)

    def forward(self, x):
        # x -> [b, 77, 768]
        b = x.shape[0]

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # 拆分注意力头
        q = q.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)
        k = k.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)
        v = v.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)

        # 计算qk乘积
        # [b*12, 77, 64] x [b*12, 64, 77] -> [b*12, 77, 77]
        attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(64)


        attn = attn.reshape(b, 12, 77, 77)

        def get_mask(b):
            mask = torch.empty(b, 77, 77)

            # 上三角设为负无穷
            mask.fill_(-float('inf'))

            # 对角线和以下位置为0
            mask.triu_(1)

            return mask.unsqueeze(1)

        attn = attn + get_mask(b)

        attn = attn.reshape(b*12, 77, 77)

        # compute softmax
        attn = attn.softmax(dim=-1)

        # 计算和v的乘积
        attn = torch.bmm(attn, v)

        attn = attn.reshape(b, 12, 77, 64).transpose(1, 2).reshape(b, 77, 768)

        # 线性输出，维度不变
        return self.out(attn)


class ClipEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.s1 = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            Atten(),
        )

        self.s2 = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Linear(768, 3072),
        )

        self.s3 = torch.nn.Linear(3072, 768)

    def forward(self, x):
        # x -> [2, 77, 768]
        x = x + self.s1(x)

        res = x

        x = self.s2(x)

        # 某种新的激活函数
        x = x * (x * 1.702).sigmoid()

        return res + self.s3(x)

encoder = torch.nn.Sequential(
    Embed(),
    ClipEncoder(),
    ClipEncoder(),
    ClipEncoder(),
    ClipEncoder(),
    ClipEncoder(),
    ClipEncoder(),
    ClipEncoder(),
    ClipEncoder(),
    ClipEncoder(),
    ClipEncoder(),
    ClipEncoder(),
    ClipEncoder(),
    torch.nn.LayerNorm(768),
)

# print(Embed()(torch.ones(2, 77).long()).shape)
# print(Atten()(torch.randn(2, 77, 768)).shape)
# print(ClipEncoder()(torch.randn(2, 77, 768)).shape)
# print(encoder(torch.ones(2, 77).long()).shape)

# 加载预训练模型
from transformers import CLIPTextModel

# 加载预训练模型参数
params = CLIPTextModel.from_pretrained(
    'lansinuote/diffsion_from_scratch.params', subfolder="text_encoder"
)

# 词编码
encoder[0].embed.load_state_dict(
    params.text_model.embeddings.token_embedding.state_dict()
)

# 位置编码
encoder[0].pos_embed.load_state_dict(
    params.text_model.embeddings.position_embedding.state_dict()
)

# 12层编码层
for i in range(12):
    # 第一层Norm
    encoder[i+1].s1[0].load_state_dict(
        params.text_model.encoder.layers[i].layer_norm1.state_dict()
    )
    # 注意力矩阵q
    encoder[i+1].s1[1].q.load_state_dict(
        params.text_model.encoder.layers[i].self_attn.q_proj.state_dict()
    )
    # 注意力矩阵k
    encoder[i + 1].s1[1].k.load_state_dict(
        params.text_model.encoder.layers[i].self_attn.k_proj.state_dict()
    )
    # 注意力矩阵v
    encoder[i + 1].s1[1].v.load_state_dict(
        params.text_model.encoder.layers[i].self_attn.v_proj.state_dict()
    )
    # 注意力out
    encoder[i+1].s1[1].out.load_state_dict(
        params.text_model.encoder.layers[i].self_attn.out_proj.state_dict()
    )
    # 第二层norm
    encoder[i + 1].s2[0].load_state_dict(
        params.text_model.encoder.layers[i].layer_norm2.state_dict()
    )
    # mlp第一层fc
    encoder[i + 1].s2[1].load_state_dict(
        params.text_model.encoder.layers[i].mlp.fc1.state_dict()
    )
    # mlp第二层fc
    encoder[i + 1].s3.load_state_dict(
        params.text_model.encoder.layers[i].mlp.fc2.state_dict()
    )

# 输出norm
encoder[13].load_state_dict(params.text_model.final_layer_norm.state_dict())