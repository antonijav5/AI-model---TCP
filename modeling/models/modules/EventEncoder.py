from typing import List

import torch
from torch import nn

from modeling.models.modules.Transformer.PositionalEncoding import PositionalEncoding


def self_attention_mask(seq_len):
    mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)
    mask_2 = torch.rot90( mask, k = 2, dims = [ 0, 1 ] )
    mask = mask + mask_2
    return mask


class EventEncoder(nn.Module):

    def __init__(
        self,
        d_categories: List[ int ],  # Each category, with how many possible values there are.
        d_model
    ):
        super(EventEncoder, self).__init__()

        self.pe = PositionalEncoding( d_model, 100000 )
        max_size = max( [ v for i, v in enumerate(d_categories) ])
        embedding_size = d_model
        self.category_embedding = nn.Embedding(
            max_size, embedding_size
        )
        self.categories = d_categories
        self.down_projection = nn.Sequential(
            nn.Linear( embedding_size * len( d_categories ), embedding_size * len( d_categories ) * 10 ),
            nn.Mish(),
            nn.Linear( embedding_size * len( d_categories) * 10, d_model )
        )

    def forward(self, events) -> torch.Tensor:
        """
        Args:
            events: [batch_size, seq_len] or [batch_size, seq_len, d_categories]
        """
        events = events.to(self.category_embedding.weight.device)

        embedded_events = self.category_embedding(events)
        # Check dimensions of input
        if events.dim() == 2:
            # Handle 2D input - assuming it's a single category
            embedded_events = self.category_embedding(events)
            # No need for down_projection since we only have one category
            return self.pe(embedded_events)

        # Original code for 3D input
        output_tensors = []
        for i, category in enumerate(self.categories):
            output_tensors.append(self.category_embedding(events[:, :, i]))
        embedded_categories = torch.cat(output_tensors, dim=2)
        embedded_events = self.down_projection(embedded_categories)

        return self.pe(embedded_events)
