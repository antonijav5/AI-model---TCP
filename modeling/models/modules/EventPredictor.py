from typing import List

import torch
from torch import nn


def self_attention_mask(seq_len):
    mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)
    mask_2 = torch.rot90( mask, k = 2, dims = [ 0, 1 ] )
    mask = mask + mask_2
    return mask


class EventPredictor(nn.Module):
    def __init__(
        self,
        d_categories: List[ int ],
        d_input,
        d_facts,
        d_model,
        num_heads,
        decoder_layers
    ):
        super(EventPredictor, self).__init__()

        self.predictor = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model = d_model,
                nhead = num_heads,
                batch_first = True,
                norm_first = True,
                activation = nn.Mish()
            ) for _ in range( decoder_layers )
        ])

        self.fact_projection = nn.Sequential(
            nn.Linear( d_model, d_facts )
        )
        modules = []
        for d_category in d_categories:
            modules.append( nn.Linear( d_model, d_category ) )
        self.categorical_projections = nn.ModuleList( modules )
        self.output_projection = nn.Sequential(
            nn.Linear( d_model, d_input )
        )

    def forward(
        self,
        tgt,
        memory,
        tgt_mask = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: [batch_size, seq_len, d_model]
            memory: [batch_size, seq_len, d_model]
                This is the "real" input sequence, that we will predict the next value for.
            tgt_mask: [batch_size]
                This is the mask for the target so that it cannot look back.

        Returns:
            predictions: [batch_size, seq_len, d_categories] of set [0,1]
        """
        y_e = tgt
        for layer in self.predictor:
            y_e = layer(
                tgt = y_e,
                memory = memory,
                tgt_mask = tgt_mask,
                tgt_is_causal = True
            )

        category_predictions = [ ]
        for output in self.categorical_projections:
            category_predictions.append( output( y_e ) )

        return category_predictions