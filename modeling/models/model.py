
from typing import List, Dict

import torch
from torch import nn

from modeling.models.modules.EventEncoder import EventEncoder
from modeling.models.modules.EventPredictor import EventPredictor


def causal_attention_mask(seq_len):
    mask = torch.triu( torch.ones( (seq_len, seq_len) ), diagonal = 1 )
    return mask


class Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_model,
        n_heads,
        encoder_layers,
        decoder_layers,
        d_categories: List[ int ],
        encoders: List[ str ],
        d_output
    ):
        super( Model, self ).__init__()
        self.event_embedder = EventEncoder( d_categories, d_model )

        encoder_modules = { }
        for encoder in encoders:
            encoder_modules[ encoder ] = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model = d_model,
                    nhead = n_heads,
                    batch_first = True,
                    activation = nn.Mish()
                ),
                num_layers = encoder_layers
            )
        self.encoders_dict = nn.ModuleDict( encoder_modules )

        self.event_predictor = EventPredictor(
            d_categories = d_categories,
            d_facts = d_output,
            d_input = d_input,
            d_model = d_model,
            num_heads = n_heads,
            decoder_layers = decoder_layers
        )

    def forward(
        self,
        src_events: Dict[ str, List[ torch.Tensor ] ],
        tgt_events: torch.Tensor,
        encodings_to_carry: int
    ) -> List[ torch.Tensor ]:
        """
        Args:
            src_events: {
                    event_stream_name: [
                        [ 0, sequence_length, categories]
                        ...
                    ],
                    ...
                }
            tgt_events: List[ Tensor[ 0, sequence_length, categories] ]
        """
        '''
        1. Encode the current state based on events at each time step.
            Concept:
                events --> 'global' employee states
        '''
        event_stream_encodings = [ ]
        # :TODO: encode src streams

        event_stream_encodings = torch.cat( event_stream_encodings, dim = 0 )
        event_stream_encodings = event_stream_encodings.reshape( ( 1, -1, event_stream_encodings.shape[2] ))

        '''
        2. Predict next event based on 'global' employee states
            Concept:
                'global' employee states --> events + 1
        '''
        tgt_output = []
        embedded_target = self.event_embedder( tgt_events )
        causal_mask = causal_attention_mask( embedded_target.shape[ 1 ] ).to( device = embedded_target.device, non_blocking=True )
        # :TODO: predict Output

        return tgt_output


class ModelTrainer(nn.Module):
    def __init__(
        self,
        d_input,
        d_model,
        n_heads,
        encoder_layers,
        decoder_layers,
        d_categories: List[ int ],
        encoders: List[str],
        d_output
    ):
        super( ModelTrainer, self ).__init__()

        self.model = Model(
            d_input = d_input,
            d_model = d_model,
            n_heads = n_heads,
            encoder_layers = encoder_layers,
            decoder_layers = decoder_layers,
            d_categories = d_categories,
            encoders = encoders,
            d_output = d_output
        )

    def forward(
        self,
        batch_src_events: List[ Dict[ str, List[ torch.Tensor ] ] ],
        batch_tgt_events: List[ torch.Tensor ],
        batch_masks: List[ List[ torch.Tensor ]],
        cross_encoder_token_length: int = 1,
        run_backward = False,
    ):
        loss_sum = 0.0
        per_category_loss = { }
        for mini_batch in range( len( batch_tgt_events ) ):
            category_losses = [ ]
            tgt_events = batch_tgt_events[ mini_batch ]
            src_events = batch_src_events[ mini_batch ]
            masks = batch_masks[ mini_batch ]
            for event_stream_name in src_events:
                for i, event_stream in enumerate( src_events[ event_stream_name ] ):
                    src_events[ event_stream_name ][ i ] = event_stream.to( dtype = torch.int32, device = torch.default_device, non_blocking = True )
            for i, mask in enumerate( masks ):
                masks[ i ] = mask.to( dtype = torch.get_default_dtype(), device = torch.default_device, non_blocking = True )
            tgt_events = tgt_events.to( dtype = torch.int32, device = torch.default_device, non_blocking = True ).unsqueeze( 0 )

            # :TODO: describe src_events
            predictions = self.midas(
                src_events,
                tgt_events[ :, :-1, : ],
                cross_encoder_token_length
            )
            for i, prediction in enumerate( predictions ):
                # -inf for logits that we know cannot be chosen
                prediction = prediction + masks[ i ]

                category_expected = tgt_events[ :, 1:, i ].reshape( -1 ).to( dtype = torch.long )
                category_loss = torch.nn.functional.cross_entropy(
                    prediction.reshape( -1, prediction.shape[ -1 ] ),
                    category_expected
                )
                category_losses.append( category_loss )
                if i not in per_category_loss:
                    per_category_loss[ i ] = 0.0
                per_category_loss[ i ] += category_loss.item()

            mini_batch_loss = torch.mean( torch.stack( category_losses ) )
            if run_backward:
                mini_batch_loss.backward()
            loss_sum += mini_batch_loss.item()

        return (
            loss_sum / len( batch_tgt_events ),
            [ per_category_loss[ i ] / len( batch_tgt_events ) for i in per_category_loss ]
        )
