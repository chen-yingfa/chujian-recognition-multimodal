from pathlib import Path

from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertOnlyMLMHead,
    BertEmbeddings,
)
from torch import nn, Tensor
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer


class VitBert(nn.Module):
    def __init__(
        self,
        bert_config_file: Path,
        num_classes: int,
        img_size: tuple,
        patch_size: int = 16,
        in_chans: int = 3,
        vit_embed_dim: int = 768,
        vit_depth: int = 12,
        vit_num_heads: int = 12,
        vit_ckpt: Path | None = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.vit = VisionTransformer(
            img_size=img_size[0],  # Somehow pylance is expecting int here.
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
        )
        if vit_ckpt is not None:
            self.vit.load_state_dict(torch.load(vit_ckpt))

        self.bert_config = BertConfig.from_json_file(bert_config_file)
        # self.bert_mlm = BertForMaskedLM(self.bert_config)

        # Custom embeddings
        self.bert_embeddings = BertEmbeddings(self.bert_config)
        self.bert_encoder = BertEncoder(self.bert_config)
        self.bert_mlm_head = BertOnlyMLMHead(self.bert_config)

        self.embed_weight = nn.parameter.Parameter(Tensor(0.5))
        self.logits_weight = nn.parameter.Parameter(Tensor(0.5))

    def forward(
        self,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ):
        """
        images: (b, n, c, h, w)
        input_ids: (b, n)
        attention_mask: (b, n)
        labels: (b, n)
        """
        batch_size, seq_len, chans, height, weight = images.shape
        images = images.view(-1, chans, height, weight)  # (b*n, c, h, w)
        hidden_vit = self.vit(images, pre_logits=True)  # (b*n, d)
        hidden_vit = hidden_vit.view(batch_size, seq_len, -1)  # (b, n, d)
        logits_vit = self.vit.head(hidden_vit)  # (b, n, v)

        # Pass to hidden states encoder
        text_embeds = self.bert_embeddings(input_ids)  # (b, n, d)
        text_embeds = (
            self.embed_weight * text_embeds
            + (1 - self.embed_weight) * hidden_vit
        )
        logits_text = self.bert_encoder(
            hidden_vit, attention_mask
        ).logits  # (b, n, v)
        logits_text = self.bert_mlm_head(logits_text)  # (b, n, v)

        # Compute logits
        logits_vit = logits_vit[attention_mask == 1]  # (num_tokens, v)
        logits_text = logits_text[attention_mask == 1]  # (num_tokens, v)
        labels_vit = labels[attention_mask == 1]  # (num_tokens)
        logits = (
            self.logits_weight * logits_vit
            + (1 - self.logits_weight) * logits_text
        )

        # Compute loss
        # NOTE: Only the masked tokens are used for computing these loss.
        loss_vit = F.cross_entropy(logits_vit, labels_vit)
        loss_text = F.cross_entropy(logits_text, labels_vit)
        loss_multimodal = F.cross_entropy(logits, labels_vit)
        loss = loss_vit + loss_text + loss_multimodal
        return loss, loss_multimodal, loss_vit, loss_text
