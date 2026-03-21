import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from efficientnet_pytorch import EfficientNet 
import math
import numpy as np 

# 1. Validate lại kỹ xem batch là bao nhiêu cho chắc
# 2. 
#

def cal_edge_emb(x):
    x = F.normalize(x, dim=-1, p=2)
    sim = torch.bmm(x, x.transpose(1, 2))
    return F.relu(sim)

# Validate kỹ phần này, build dựa trên cái gì
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features)) # (512, 512)
        self.bias = nn.Parameter(torch.Tensor(out_features)) # (512,) 
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 
        nn.init.zeros_(self.bias) 
        # Batch 
    def forward(self, x, adj):
        # x (Node feature) => (B, 4, 512)
        I = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0) # (1,4,4) 
        adj = F.normalize(adj + I, p=1, dim=-1) # Adjacentcy Matrix  (B, 4, 4) + (1, 4, 4) --> (B, 4, 4)
        return torch.matmul(adj, torch.matmul(x, self.weight)) + self.bias  # 1: (B, 4, 512) x (512, 512) => (B, 4, 512) + (512,) 

# Build feature node 
# 10 nodes, nodes biểu trưng cho cái gì
class DualGraphAdapter(nn.Module):
    def __init__(self, text_features): # iNPUT: image Feature (4, 512)
        super().__init__()
        self.num_classes, self.dim = text_features.shape
        self.register_buffer("base_text_features", text_features) 
        self.visual_context = nn.Parameter(text_features.clone()) 
        
        # Hai mạng Graph Convolution độc lập, có số chiều là 512 x 512
        self.gcn_sem = GraphConvolution(self.dim, self.dim) # (512, 512)
        self.gcn_str = GraphConvolution(self.dim, self.dim) # (512, 512)
        
        self.gate_net = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.GELU(),
            nn.Dropout(0.5), 
            nn.Linear(self.dim // 2, 1)
        )
        
        # Tham số alpha học được (learnable parameter) cho Weighted Residual Connection
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, img_feat): 
        B = img_feat.size(0)   
        # Z_t (Đặc trưng văn bản gốc của 4 subtype) => (B, 4, 512)
        x = self.base_text_features.unsqueeze(0).expand(B, -1, -1)  
            # base_txxt_features (1,4,512)

        # Structure Graph (Kiến thức y văn tĩnh)
        adj_str = cal_edge_emb(self.base_text_features.unsqueeze(0))  # Adjacency Matrix for Structure (1,4,4)
            # base_text_features x base_text_features.transpose()
            # (1,4, 512) x (1, 512, 4) => (1,4,4)
        out_str = F.relu(self.gcn_str(x, adj_str)) # Z_tt => (B, 4, 512)
            # (1,4,4) x (B, 4, 512)  => (1, 4, 512)
        
        # Semantic Graph (Cá nhân hóa theo ảnh)
        # dynamic context => (1, 4, 512)
        # Dùng phép nhân (*) thay vì cộng (+) để tránh Oversmoothing
        dynamic_context = self.visual_context.unsqueeze(0) * img_feat.unsqueeze(1) # Phải nhân 2 features với nhau
                # img_feat (B, 1 , 512) 
                # visual context (1, 4, 512)
                # => (B, 4, 512)
        adj_sem = cal_edge_emb(dynamic_context) 
          # dynamic_context x dynamic_context Transpose
          # (B, 4, 512) x (B, 512, 4) => (B, 4, 4)
        out_sem = F.relu(self.gcn_sem(x, adj_sem)) # Z_vt => (B, 4, 512) 
        
        # Graph Gating (Phương trình 1: Z_t' = beta * Z_tt + (1 - beta) * Z_vt)
        beta = torch.sigmoid(self.gate_net(img_feat)).unsqueeze(1) # (B, 1, 1)
        z_t_prime = beta * out_sem + (1 - beta) * out_str # => (B, 4, 512)
          # gate x out_sem  | (B, 1, 1) nhân tensor (B, 4, 512) => (B, 4, 512)
          # gate oC x out_str | shape(B, 4, 512)
          # => (B, 4, 512)
        
        # 4. Weighted Residual Connection (Phương trình 2: Z_t^* = alpha * Z_t + (1 - alpha) * Z_t')
        z_t_star = self.alpha * x + (1 - self.alpha) * z_t_prime # => (B, 4, 512)
        
        return z_t_star 

class MultiModalAdapter(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.5): 
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), # (B, 1, 2048)
            nn.GELU(),  
            nn.Dropout(dropout), # 50% neural để tránh overfitting
            nn.Linear(dim * 4, dim) # Chuyển về 512 chiêuy
        )
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, image_embeds, text_embeds, key_padding_mask=None): # [SỬA LỖI] Thêm mask
        # Image_embed = Query Q => img_feat qua unsqueeze(1) => (B, 1, 512)
        # Text_embed = Value V & Key K Đặc trưng văn bản (text_feat) lấy từ CLinicalBERT shape (B, L, 512)
        
        attn_out, _ = self.attn(query=image_embeds, key=text_embeds, value=text_embeds, key_padding_mask=key_padding_mask)
         # Query (B, 1, 512) x Key.transpose() (B, 512, L) => (B, 1 , L ) | Softmax(Q.K.transpose/sqrt(dk)) x V
         # Cho qua hàm softmax để quy chuyển về giá  trị %
         # (B, 1, L) x (B, L, 512) => (B, 1, 512)
        x = self.norm1(image_embeds + attn_out)
         # (B, 1, 512 ) + (B, 1, 512) => (B, 1, 512)
        return self.norm2(x + self.ffn(x)).squeeze(1) 
          # Feed foorwaed network 
           # => (B, 512)

# --- [MỚI] LATE FUSION MODULE ---
class PatientLateFusion(nn.Module):
    """
    Thực hiện phương trình cá nhân hóa Prototypes: Z_t^{**} = gamma * Z_t^* + (1 - gamma) * Z_text
    """
    def __init__(self, dim):
        super().__init__()
        # Mạng MLP gắn với ClinicalBERT (đầu vào là Z_text)
        # Dánh giá bệnh án này để tự động tính ra hệ số lai tạp gamma
        self.mlp_gamma = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid() # Ép giá trị gamma về khoảng [0, 1]
        )
        
        # Bias = 2.0 giúp gamma ban đầu khoangr 0.88 
        nn.init.xavier_uniform_(self.mlp_gamma[2].weight)
        nn.init.constant_(self.mlp_gamma[2].bias, 2.0)

    def forward(self, z_t_star, z_text):
        """
        Input:
            z_t_star: Prototypes chung từ GCN. Shape: (B, 4, 512)
            z_text: Đặc trưng bệnh án từ ClinicalBERT + Attention. Shape: (B, 512)
        Output:
            z_t_star_star: Prototypes đã được cá nhân hóa. Shape: (B, 4, 512)
        """
        # Dùng MLP tính gamma từ bệnh án
        gamma = self.mlp_gamma(z_text) # => (B, 1)
        gamma = gamma.unsqueeze(1)     # => (B, 1, 1) 
        
        # Mở rộng chiều của Z_text để cộng được với 4 Prototypes
        z_text_expanded = z_text.unsqueeze(1) # (B, 512) => (B, 1, 512)
        
        # Phương trình Late Fusion (Trộn 4 node đồ thị chung với 1 vector bệnh án cá nhân)
        z_t_star_star = gamma * z_t_star + (1 - gamma) * z_text_expanded # => (B, 4, 512)
        
        # Chuẩn hóa L2 để chuẩn bị cho phép tính Cosine Similarity ở cuối
        return F.normalize(z_t_star_star, dim=-1)

class MammoAnalyzeModel(nn.Module):
    def __init__(self, task_id, 
                 model_name="/home/minhntn24/nhatminh/M2G-BRCA/data/huggingface_models/clip-vit-base-patch32", 
                 mammoclip_path=None, 
                 text_model_name="/home/minhntn24/nhatminh/M2G-BRCA/data/huggingface_models/Bio_ClinicalBERT"):
        super().__init__()
        self.task_id = task_id
        
        self.vision_encoder = EfficientNet.from_name('efficientnet-b5')
        
        self.visual_projection = nn.Sequential(
            nn.Linear(2048, 512, bias=False), 
            nn.LayerNorm(512),
            nn.Dropout(0.1) 
        )
        nn.init.kaiming_normal_(self.visual_projection[0].weight, mode='fan_out', nonlinearity='relu')

        if mammoclip_path and os.path.exists(mammoclip_path):
            ckpt = torch.load(mammoclip_path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
            model_dict = self.vision_encoder.state_dict()
            p_dict = {k.replace("image_encoder.", "").replace("module.", ""): v 
                      for k, v in state_dict.items() if k.replace("image_encoder.", "").replace("module.", "") in model_dict}
            self.vision_encoder.load_state_dict(p_dict, strict=False)
            print(f"🟢 [VISION] Successfully loaded MammoCLIP backbone")

        # FREEZE VISION BACKBONE
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        if self.task_id == 1: 
            self.classifier = nn.Linear(512, 4)
            
        if self.task_id in [2, 3]:
            # Đọc CLIP Offline từ Local
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            clip_model = AutoModel.from_pretrained(model_name)
            classes = ["Luminal A", "Luminal B", "HER2", "Triple Negative"]
            with torch.no_grad():
                t_emb = clip_model.text_projection(clip_model.text_model(**tokenizer(classes, padding=True, return_tensors="pt")).pooler_output)
            
            self.graph_learner = DualGraphAdapter(t_emb / t_emb.norm(dim=-1, keepdim=True))
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            
        if self.task_id == 3:
            # Đọc Bio_ClinicalBERT Offline từ Local
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            
            # FREEZE TEXT BACKBONE
            for param in self.text_encoder.parameters():
                param.requires_grad = False
                
            self.text_proj = nn.Linear(768, 512)
            nn.init.kaiming_normal_(self.text_proj.weight, mode='fan_out', nonlinearity='relu')
            #self.adapter = MultiModalAdapter(512, dropout=0.5) #Remove attention mechanism
            
            self.late_fusion = PatientLateFusion(dim=512)

    def forward(self, image, input_ids=None, attention_mask=None):
         # image (B, 3, H, W)
         # input_id, attn_mask => (B, L)
         # = > (B, 2048, H', W') => (B, 2048, 1, 1) 
         
        with torch.no_grad():
            v_feat = F.adaptive_avg_pool2d(self.vision_encoder.extract_features(image), (1, 1)).flatten(1)
             # v_feat => (B, 2048)
             
        # Z_image (Đặc trưng hình ảnh khối u)
        img_feat = F.normalize(self.visual_projection(v_feat), dim=-1) # => (B, 512)
        
        if self.task_id == 1: 
            return self.classifier(img_feat)
    
        # Z_t^* (Prototypes từ đồ thị kép)
        # Trộn img_feat với Text Embedding của 4 loại ung thư dể tạo ra đồ thị => L2 Normalization
        prototypes = F.normalize(self.graph_learner(img_feat), dim=-1) # => (B, 4, 512)

        if self.task_id == 3:
            with torch.no_grad():
                t_feat_raw = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # (B, L)
            t_feat = self.text_proj(t_feat_raw) # => (B, L, 512)
            # input ids => Clinical BERT (B, L, 512)
            # img (B, 1, 512) 
            # text (B, L, 512)  
            
            mask = attention_mask.unsqueeze(-1).expand(t_feat.size()).float() # (B, L, 512)
            sum_embeddings = torch.sum(t_feat * mask, 1) # Sum features of non-pad tokens
            sum_mask = torch.clamp(mask.sum(1), min=1e-9) # Count non-pad tokens
            patient_feat = F.normalize(sum_embeddings / sum_mask, dim=-1) # (B, 512)
            
            # Áp dụng Patient-level Late Fusion
            # Z_t^** = gamma * Z_t^* + (1 - gamma) * Z_text
            final_prototypes = self.late_fusion(prototypes, patient_feat) # => (B, 4, 512)
        else: 
            # Nếu không có ClinicalBERT (Task 2), giữ nguyên Prototypes từ Graph
            final_prototypes = prototypes # => (B, 4, 512)

        # (Logits = Z_image * (Z_t^**)^T )
        # Validate để chác chắn bmm (Nhân ma trận) nó phải đúng / Nhân tay xem sao
        # img_feat.unsqueeze(1) => (B, 1, 512)
        # final_prototypes.transpose(1, 2) => (B, 512, 4)
        # bmm nhân lại => (B, 1, 4) -> squeeze(1) => (B, 4)
        logits = torch.bmm(img_feat.unsqueeze(1), final_prototypes.transpose(1, 2)).squeeze(1) 
         # => (B, 1, 4)
         # => (B, 4)
        
        return self.logit_scale.exp() * logits # => (B, 4)