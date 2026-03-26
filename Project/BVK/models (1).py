import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from efficientnet_pytorch import EfficientNet 
import math
import numpy as np 

# =====================================================================
# BÊ NGUYÊN BLOCK CODE TỪ BASECLIP_GRAPH_V1.PY
# =====================================================================

def graph_norm_ours(A, batch=False, self_loop=True, symmetric=True):
    # A = A + I    A: (bs, num_nodes, num_nodes)
    # Degree
    d = A.sum(-1) # (bs, num_nodes) #[1000, m+1]
    if symmetric:
        # D = D^-1/2
        d = torch.pow(d, -0.5)
        # Fix lỗi chia cho 0 sinh ra vô cực (Inf)
        d[torch.isinf(d)] = 0.0 
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A).bmm(D)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A).mm(D)
    else:
        # D=D^-1
        d = torch.pow(d,-1)
        d[torch.isinf(d)] = 0.0
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A)

    return norm_A

def cal_similarity(x, p=2, dim=1):
    '''
    x: (n,K)
    return: (n,n)
    '''
    x = F.normalize(x, p=p, dim=dim)
    return torch.mm(x, x.transpose(0, 1))

def cal_edge_emb(x, p=2, dim=1):   # v1_graph---taking the similairty by 
    ''' 
    x: (n,K)   [m+1, 1000, 1024]
    return: (n^2, K)
    '''
    x = F.normalize(x, p=p, dim=dim)    #[m+1, 1000, 1024], [100, 1024, 101]
    x_c = x
    x = x.transpose(1, 2)  #[1000, m+1, 1024]  [100, 101, 1024]
    x_r = x  # (K, n, 1) #[1000, m+1, 1024]
    A = torch.bmm(x_r, x_c)     # [1000, m+1, m+1]
    return A


class GraphConvolution(nn.Module):
    def __init__(self, hidden_dim, name=None, device=None, class_num=None, sparse_inputs=False, act=nn.Tanh, bias=True, dropout=0.0):
        super().__init__()
        self.act = nn.Tanh()
        self.device = device
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.hidden_dim = 512 # Cứng theo baseclip
        self.class_num = class_num
        self.gcn_weights = nn.Parameter(torch.ones(self.hidden_dim, self.hidden_dim))
        if self.bias:
            self.gcn_bias = nn.Parameter(torch.zeros(class_num, self.hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.gcn_weights.size(1))
        self.gcn_weights.data.uniform_(-stdv, stdv)

    def forward(self, feat, adj):
        x = feat        #[100, 1024, 101]
        node_size = adj.size()[1]  
        adj = torch.clip(adj, min=0.0)
        I = torch.eye(node_size, device=adj.device).unsqueeze(dim=0).to(adj.device)
        adj = adj + I      # [1000, m+1, m+1]
        adj = graph_norm_ours(adj, batch=True, self_loop=True, symmetric=True)  #[1000, m+1, m+1]
        x = x.transpose(1, 2)
        pre_sup = torch.matmul(x, self.gcn_weights)  # [m+1, 1000, 1024]
        output = torch.matmul(adj, pre_sup) #[1000, m+1, 1024]

        if self.bias:
            # SỬA LỖI Ở ĐÂY: Dùng unsqueeze(1) để cộng bias của từng class [4, 1, 512] vào 5 nodes [4, 5, 512]
            output += self.gcn_bias.unsqueeze(1)
            
        if self.act is not None:
            return self.act(output[:, 0, :])
        else:
            return output[:, 0, :]


class GraphLearner(nn.Module):
    def __init__(self, base_text_features, base_img_features, beta_it=0.5):
        super().__init__()
        self.device = base_text_features.device 
        self.alpha = 0.1
        print(">> DCT scale factor: ", self.alpha)
        
        # Đưa thẳng 4 nodes (4 subtypes) vào buffer
        self.register_buffer("base_text_features", base_text_features) #[4, 512]
        self.register_buffer("base_img_features", base_img_features) #[4, 512]
        
        self.alpha_it = 0.7
        self.beta_it = beta_it 
        self.hidden_dim = 1 # Giữ nguyên biến để pass vào hàm init của GCN
        
        # Khởi tạo GCN với số node chính xác bằng số class (4)
        self.GCN_tt = GraphConvolution(self.hidden_dim, name='metagraph', device=self.device, class_num=base_text_features.size()[0])
        self.GCN_it = GraphConvolution(self.hidden_dim, name='metagraph', device=self.device, class_num=base_text_features.size()[0])

    def forward(self, img_feature):
        # Không dùng biến B (Batch Size) ở đây để tính Graph
        with torch.no_grad():
            # feat_tt: [4, 1, 512] (Target nodes)
            inputs_text = self.base_text_features.unsqueeze(1)
            # node_cluster: [4, 4, 512] (Context nodes)
            node_cluster_tt = self.base_text_features.unsqueeze(0).expand(4, -1, -1)
            node_cluster_it = self.base_img_features.unsqueeze(0).expand(4, -1, -1)
            
            # Tạo đồ thị 5 nodes (1 target + 4 neighbors) cho mỗi class
            feat_tt = torch.cat([inputs_text, node_cluster_tt], dim=1).transpose(1, 2).detach() # [4, 512, 5]
            feat_it = torch.cat([inputs_text, node_cluster_it], dim=1).transpose(1, 2).detach() # [4, 512, 5]
            
            edge_tt = cal_edge_emb(feat_tt).detach() # [4, 5, 5]
            edge_it = cal_edge_emb(feat_it).detach() # [4, 5, 5]
            
        # GCN chạy trên "batch" 4 class -> Kết quả trả về [4, 512]
        graph_o_tt = self.GCN_tt(feat_tt, edge_tt) 
        graph_o_it = self.GCN_it(feat_it, edge_it)
        
        graph_o_t = (graph_o_tt) * self.alpha_it + (1 - self.alpha_it) * graph_o_it
        
        # Kết quả là 4 Prototypes đã được tinh chỉnh
        refined_prompts = self.beta_it * self.base_text_features + (1 - self.beta_it) * graph_o_t
    
        return refined_prompts, img_feature

#Our SOTA
class PatientLateFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_gamma = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid() 
        )
        nn.init.xavier_uniform_(self.mlp_gamma[2].weight)
        nn.init.constant_(self.mlp_gamma[2].bias, 0.0)

    def forward(self, z_t_star, z_text): #Input as prototypes (B, 4, 512) and patient text feature (B, 512)
        gamma = self.mlp_gamma(z_text) 
        gamma = gamma.unsqueeze(1)     
        z_text_expanded = z_text.unsqueeze(1) 
        z_t_star_star = gamma * z_t_star + (1 - gamma) * z_text_expanded #z_text_expanded = z_Bio_ClinicalBERT, z_t_star = prototypes từ GraphLearner
        return F.normalize(z_t_star_star, dim=-1)


class MammoAnalyzeModel(nn.Module):
    def __init__(self, task_id, 
                 base_img_features=None, 
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
                t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
            
            # Khởi tạo chuẩn xác bằng GraphLearner gốc
            if base_img_features is None:
                base_img_features = torch.randn_like(t_emb) 
                
            self.graph_learner = GraphLearner(base_text_features=t_emb, 
                                              base_img_features=base_img_features, 
                                              beta_it=0.5)
                                              
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0))
            
        if self.task_id == 3:
            # Đọc Bio_ClinicalBERT Offline từ Local
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
                
            self.text_proj = nn.Linear(768, 512)
            nn.init.kaiming_normal_(self.text_proj.weight, mode='fan_out', nonlinearity='relu')
            self.late_fusion = PatientLateFusion(dim=512)

    def forward(self, image, input_ids=None, attention_mask=None):
        with torch.no_grad():
            v_feat = F.adaptive_avg_pool2d(self.vision_encoder.extract_features(image), (1, 1)).flatten(1)
             
        img_feat = F.normalize(self.visual_projection(v_feat), dim=-1) # => (B, 512)
        
        if self.task_id == 1: 
            return self.classifier(img_feat)
    
        # GraphLearner gốc trả về tuple: (text_features, image_features)
        text_features, _ = self.graph_learner(img_feat) 
        
        # SỬA LỖI Ở ĐÂY: Khai báo rõ prototypes_global trước khi expand
        prototypes_global = F.normalize(text_features, dim=-1) 
        
        # Ta cần expand ra theo size của Batch
        B = img_feat.size(0)
        # Expand Prototypes từ [4, 512] thành [B, 4, 512]
        prototypes = prototypes_global.unsqueeze(0).expand(B, -1, -1)

        if self.task_id == 3:
            with torch.no_grad():
                t_feat_raw = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state 
            t_feat = self.text_proj(t_feat_raw) 
            
            mask = attention_mask.unsqueeze(-1).expand(t_feat.size()).float() 
            sum_embeddings = torch.sum(t_feat * mask, 1) 
            sum_mask = torch.clamp(mask.sum(1), min=1e-9) 
            patient_feat = F.normalize(sum_embeddings / sum_mask, dim=-1) 
            
            patient_feat = F.dropout(patient_feat, p=0.3, training=self.training)
            
            final_prototypes = self.late_fusion(prototypes, patient_feat) # => (B, 4, 512)
        else: 
            final_prototypes = prototypes # => (B, 4, 512)

        logits = torch.bmm(img_feat.unsqueeze(1), final_prototypes.transpose(1, 2)).squeeze(1) 
        
        return self.logit_scale.exp() * logits # => (B, 4)