from pathlib import Path
from typing import Union

from transformers import BertConfig
from transformers.utils import ModelOutput
from transformers.models.bert.modeling_bert import (
    BertEncoder,
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
        vit_ckpt: Union[Path, None] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.in_chans = in_chans

        # Vit
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
            print(f'Loading vit checkpoint from {vit_ckpt}')
            state_dict = torch.load(vit_ckpt)
            renamed = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    k = k[6:]
                renamed[k] = v
            self.vit.load_state_dict(renamed)

        self.bert_config = BertConfig.from_json_file(bert_config_file)
        # self.bert_mlm = BertForMaskedLM(self.bert_config)

        # Custom embeddings
        self.vocab_size = num_classes + 5  # 5 special tokens
        self.bert_embeddings = BertEmbeddings(self.bert_config)
        self.bert_encoder = BertEncoder(self.bert_config)
        self.bert_mlm_head = nn.Linear(
            self.bert_config.hidden_size, self.vocab_size
        )

        self.embed_weight = nn.Parameter(torch.tensor(0.5))
        self.logits_weight = nn.Parameter(torch.tensor(1.0))

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
        # (b, n - 2, v), (b, n - 2, d)
        logits_vit, hidden_vit = self.forward_vit(images)
        # (b, n, v)
        logits_text = self.forward_text(input_ids, attention_mask, hidden_vit)
        # (b, n - 2, v)
        logits = (
            self.logits_weight * logits_vit
            + (1 - self.logits_weight) * logits_text[:, 1:-1]
        )
        logits = F.pad(logits, (0, 0, 1, 1), value=0)  # (b, n, v)

        # Compute loss
        # NOTE: Only the masked tokens are used for computing these loss.
        loss, loss_multimodal, loss_vit, loss_text = self.loss_fn(
            logits, logits_vit, logits_text, labels
        )
        return ModelOutput(
            loss=loss_vit,
            loss_multimodal=loss_multimodal,
            loss_vit=loss_vit,
            loss_text=loss_text,
            logits=logits,
        )

    def forward_vit(self, images: Tensor):
        """
        images: (b, n - 2, c, h, w)
        Return (logits, hidden)
        logits: (b, n - 2, v)
        hidden: (b, n - 2, d)
        """
        batch_size, seq_len, chans, height, weight = images.shape
        images = images.view(-1, chans, height, weight)  # (b*n, c, h, w)
        hidden_states = self.vit.forward_features(images)  # (b*n, p, d)
        # (b*n, d)
        cls_embed: Tensor = self.vit.forward_head(
            hidden_states, pre_logits=True
        )
        logits: Tensor = self.vit.head(cls_embed)  # (b*n, v-5)
        # Pad 5 zeros for special tokens.
        logits = F.pad(logits, (0, 5), value=0)  # (b*n, v)
        return (
            logits.view(batch_size, seq_len, -1),
            cls_embed.view(batch_size, seq_len, -1),
        )

    def forward_text(
        self, input_ids: Tensor, attention_mask: Tensor, img_embeds: Tensor
    ) -> Tensor:
        """
        input_ids: (b, n)
        attention_mask: (b, n)
        img_embeds: (b, n - 2, d), -2 because [CLS] and [SEP] has no image.

        Return logits
        logits: (b, n, v)
        """
        padded_img_embeds = F.pad(
            img_embeds, (0, 0, 1, 1), value=0
        )  # (b, n, d)
        text_embeds = self.bert_embeddings(input_ids)
        text_embeds = (
            self.embed_weight * text_embeds
            + (1 - self.embed_weight) * padded_img_embeds
        )  # (b, n, d)
        logits = self.bert_encoder(
            text_embeds, attention_mask[:, None, None, :]
        ).last_hidden_state  # (b, n, v)
        logits = self.bert_mlm_head(logits)  # (b, n, v)
        return logits

    def loss_fn(
        self,
        logits: Tensor,
        logits_vit: Tensor,
        logits_text: Tensor,
        labels: Tensor,
    ):
        '''
        logits: (b, n, v)
        logits_vit: (b, n - 2, v)
        logits_text: (b, n, v)
        labels: (b, n)
        '''
        assert logits.shape == logits_text.shape
        # NOTE: Only the masked tokens are used for computing these loss.
        #  Remove the [CLS] and [SEP] tokens.
        labels = labels[:, 1:-1]
        logits_text = logits_text[:, 1:-1]
        logits = logits[:, 1:-1]
        # Now they are all (b, n - 2, v)
        mask_indices = labels != -100
        logits = logits[mask_indices]
        logits_vit = logits_vit[mask_indices]  # (num_tokens, v)
        logits_text = logits_text[mask_indices]  # (num_tokens, v)
        labels = labels[mask_indices]  # (num_tokens)

        loss_vit = F.cross_entropy(logits_vit, labels)
        loss_text = F.cross_entropy(logits_text, labels)
        loss_multimodal = F.cross_entropy(logits, labels)
        loss = loss_multimodal + loss_vit + loss_text
        return loss, loss_multimodal, loss_vit, loss_text
