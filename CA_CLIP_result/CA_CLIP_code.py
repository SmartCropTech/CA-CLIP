import math
import torch
import torch.nn as nn

# =========================
# class-aware (CA)
# =========================
class ClassAwareResidualGate(nn.Module):

    def __init__(self, dim, num_classes, dropout=0.1):
        super().__init__()

        self.class_queries = nn.Parameter(torch.randn(num_classes, dim) * 0.02)
        self.class_bias = nn.Parameter(torch.zeros(num_classes))

        self.img_reliability = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        self.txt_reliability = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        self.class_gate = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, img_proj, txt_proj):

        diff = torch.abs(img_proj - txt_proj)

        img_rel = torch.sigmoid(self.img_reliability(torch.cat([img_proj, diff], dim=-1)))
        txt_rel = torch.sigmoid(self.txt_reliability(torch.cat([txt_proj, diff], dim=-1)))

        q = self.class_queries.unsqueeze(0).expand(img_proj.size(0), -1, -1)

        img_expand = img_proj.unsqueeze(1).expand(-1, q.size(1), -1)
        txt_expand = txt_proj.unsqueeze(1).expand(-1, q.size(1), -1)
        diff_expand = diff.unsqueeze(1).expand(-1, q.size(1), -1)


        gate_input = torch.cat([
            img_expand * q,
            txt_expand * q,
            diff_expand * q,
            q
        ], dim=-1)

        gate = torch.sigmoid(self.class_gate(gate_input))  


        effective_txt_gate = gate * txt_rel.unsqueeze(1)


        fused = self.out_norm(
            img_expand + effective_txt_gate * (txt_expand - img_expand)
        )

        logits = (fused * q).sum(dim=-1) / math.sqrt(img_proj.size(-1)) + self.class_bias

        aux = {
            "gate": gate,
            "effective_txt_gate": effective_txt_gate,
            "img_rel": img_rel,
            "txt_rel": txt_rel
        }
        return logits, aux

class BalancedCLIPTeacher(nn.Module):

    def __init__(self, clip_model, num_classes, projection_dim=256, freeze_clip=True):
        super().__init__()
        self.clip_model = clip_model
        self.freeze_clip = freeze_clip

        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        img_feat_dim = clip_model.visual.output_dim
        txt_feat_dim = clip_model.text_projection.shape[1]


        self.image_projection = nn.Sequential(
            nn.Linear(img_feat_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(txt_feat_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.fusion_head = ClassAwareResidualGate(
            dim=projection_dim,
            num_classes=num_classes
        )

    def forward(self, images, text_tokens):
        if self.freeze_clip:
            with torch.no_grad():
                img_feat = self.clip_model.encode_image(images)
                txt_feat = self.clip_model.encode_text(text_tokens)
        else:
            img_feat = self.clip_model.encode_image(images)
            txt_feat = self.clip_model.encode_text(text_tokens)

        img_proj = self.image_projection(img_feat.float())
        txt_proj = self.text_projection(txt_feat.float())

        logits, aux = self.fusion_head(img_proj, txt_proj)

        return logits, aux