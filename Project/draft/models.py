import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import math
import logging
import numpy as np 

# ==========================================
# 1. GRAPH UTILS
# ==========================================
def cal_edge_emb(x, p=2, dim=1):
    x = F.normalize(x, p=p, dim=dim)
    return torch.bmm(x, x.transpose(1, 2))

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        I = torch.eye(adj.size(1), device=adj.device).unsqueeze(0)
        adj = adj + I 
        d = adj.sum(-1, keepdim=True).clamp(min=1e-6)
        adj = adj / d
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output

# ==========================================
# 2. CLASS GRAPH LEARNER
# ==========================================
class ClassGraphLearner(nn.Module):
    def __init__(self, text_features):
        super().__init__()
        self.register_buffer("base_text_features", text_features)
        dim = text_features.size(1)
        self.visual_context = nn.Parameter(text_features.clone())
        self.gcn = GraphConvolution(dim, dim)
        self.alpha = 0.5 
        self.relu = nn.ReLU()

    def forward(self):
        inputs_text = self.base_text_features.unsqueeze(0)
        inputs_vis = self.visual_context.unsqueeze(0)
        adj = cal_edge_emb(inputs_vis)
        out = self.gcn(inputs_text, adj)
        out = self.relu(out)
        final = self.alpha * inputs_text + (1 - self.alpha) * out
        return final.squeeze(0)

# ==========================================
# 3. MAIN MODEL
# ==========================================
class MammoAnalyzeModel(nn.Module):
    def __init__(self, task_id, model_name="openai/clip-vit-base-patch32", num_classes=4, mammoclip_path=None):
        super().__init__()
        self.task_id = task_id
        
        # --- VISION ENCODER ---
        self.clip_model = AutoModel.from_pretrained(model_name)
        self.vision_encoder = self.clip_model.vision_model
        self.visual_projection = self.clip_model.visual_projection
        vision_dim = self.clip_model.config.projection_dim 
        
        # Load Weights
        if mammoclip_path and mammoclip_path != "None":
            print(f"[INFO] Loading MammoCLIP weights from: {mammoclip_path}")
            try:
                state_dict = torch.load(mammoclip_path, map_location='cpu', weights_only=False)
                if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
                elif 'model' in state_dict: state_dict = state_dict['model']
                
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'): k = k[7:]
                    new_state_dict[k] = v
                self.clip_model.load_state_dict(new_state_dict, strict=False)
                print(f"[INFO] MammoCLIP weights loaded.")
            except Exception as e:
                print(f"[WARNING] Load failed: {e}. Using OpenAI weights.")

        # --- TASKS ---
        if task_id == 1:
            self.classifier = nn.Linear(vision_dim, num_classes)
            
        elif task_id in [2, 3]:
            print(f"[INFO] Task {task_id}: Initializing Class-Knowledge Graph...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            class_names = ["Luminal A breast cancer", "Luminal B breast cancer", "HER2-enriched breast cancer", "Triple-negative breast cancer"]
            with torch.no_grad():
                inputs = tokenizer(class_names, padding=True, return_tensors="pt")
                text_outputs = self.clip_model.text_model(**inputs)
                text_embeds = self.clip_model.text_projection(text_outputs.pooler_output)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            self.graph_learner = ClassGraphLearner(text_embeds)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            
            if task_id == 3:
                print("[INFO] Task 3: Adding BioClinicalBERT...")
                self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                text_dim = 768 
                self.text_proj = nn.Linear(text_dim, vision_dim)
                self.fusion_gate = nn.Parameter(torch.tensor(0.5)) 

    def forward(self, image, input_ids=None, attention_mask=None):
        vision_out = self.vision_encoder(image)
        img_emb = self.visual_projection(vision_out.pooler_output)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        
        if self.task_id == 1:
            return self.classifier(img_emb)
            
        # GRAPH ADAPTER LOGIC
        refined_classifier = self.graph_learner()
        refined_classifier = refined_classifier / refined_classifier.norm(dim=-1, keepdim=True)
        
        if self.task_id == 2:
            patient_feature = img_emb
        elif self.task_id == 3:
            txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            txt_emb = txt_out.pooler_output
            txt_emb = self.text_proj(txt_emb)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            patient_feature = (1 - self.fusion_gate) * img_emb + self.fusion_gate * txt_emb
            patient_feature = patient_feature / patient_feature.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * patient_feature @ refined_classifier.t()
        return logits