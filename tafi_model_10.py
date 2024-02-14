import torch
import torch.nn as nn
import math

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoderWithConv(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=128, dropout=0.5):
        super(TransformerEncoderWithConv, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=mask)
        src = src.permute(0, 2, 1)  # Transpose for Conv1d (input shape: B x C x L)
        src = self.conv(src)
        src = src.permute(0, 2, 1)  # Transpose back to B x L x C
        return self.norm(src)

class TAFIWithFCOutput(nn.Module):
    def __init__(self, context_size=1, num_heads=8, num_layers=2, hidden_size=256, dropout=0.1):
        super(TAFIWithFCOutput, self).__init__()

        self.context_size = context_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Spatial Encoder with Convolutional Layer
        self.spatial_embedding = nn.Conv1d(6, hidden_size, kernel_size=3, padding=1)
        self.spatial_attention = TransformerEncoderWithConv(hidden_size, num_heads, num_layers)

        # Contextual Component with Convolutional Layer
        self.context_embedding = nn.Conv1d(context_size, 2*hidden_size, kernel_size=3, padding=1)
        self.context_attention = TransformerEncoderWithConv(2*hidden_size, num_heads, num_layers)
        self.context_fc = nn.Linear(2*hidden_size, hidden_size)

        # Temporal Decoder
        self.temporal_embedding = nn.Conv1d(6, hidden_size, kernel_size=3, padding=1)
        self.temporal_attention = nn.TransformerDecoderLayer(hidden_size, num_heads)

        # # Positional Encoder
        self.positional_encoder = nn.Linear(hidden_size,hidden_size)

        # Output layer
        self.output_layer = FCOutputModule(hidden_size, num_outputs=2)  # Output vx, vy

    def forward(self, x, context):
        # Spatial Encoder with Convolutional Layer
        spatial_embedding = self.spatial_embedding(x)
        spatial_attention = self.spatial_attention(spatial_embedding.permute(0, 2, 1))

        # Contextual Component with Convolutional Layer
        context = context.unsqueeze(2)
        context_embedding = self.context_embedding(context)
        context_attention = self.context_attention(context_embedding.permute(0, 2, 1))
        context_attention = self.context_fc(context_attention)

        # Concatenate Spatial and Contextual features
        combined_features = torch.cat((spatial_attention, context_attention), dim=1)#spatial_attention
        combined_features = combined_features.permute(1, 0, 2)

        # Positional Encoding [sequence length, batch size, embed dim]


        # Temporal Decoder with Convolutional Layer
        temporal_embedding = self.temporal_embedding(x)
        target_mask = self.generate_square_subsequent_mask(200).to(temporal_embedding.device)
        temporal_embedding = temporal_embedding.permute(2, 0, 1)

        # Positional Encoding
        positional_encoded_temporal = self.positional_encoder(temporal_embedding)

        temporal_attention = self.temporal_attention(positional_encoded_temporal, combined_features, tgt_mask=target_mask)

        # Apply the output layer
        predictions = self.output_layer(temporal_attention)

        return predictions

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class FCOutputModule(nn.Module):
    def __init__(self, in_planes, num_outputs, **kwargs):
        super(FCOutputModule, self).__init__()
        hidden_size = in_planes
        dropout = kwargs.get('dropout', 0.1)
        sequence_length = kwargs.get('sequence_length', 200)  # Assuming sequence length is 200

        self.fc = nn.Sequential(
            nn.Linear(sequence_length * hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_outputs)
        )

    def forward(self, x):
        batch_size = x.size(1)  # Swap batch_size and sequence_length dimensions
        x = x.permute(1, 0, 2).reshape(batch_size, -1)
        y = self.fc(x)
        return y
