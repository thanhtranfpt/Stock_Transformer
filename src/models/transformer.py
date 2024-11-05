import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, cfg: dict) -> None:
        """
            Args:
                cfg = {
                    'd_model': 10,
                    'max_len': 5000
                }

        """

        super().__init__()

        self.config = cfg

        # Create a tensor of shape (max_len, d_model)
        pe = torch.zeros(self.config['max_len'], self.config['d_model'])
        position = torch.arange(start=0, end=self.config['max_len'], dtype=torch.float).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(start=0, end=self.config['d_model'], step=2).float() * (-math.log(10000.0) / self.config['d_model']))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(dim=0)  # Shape (1, max_len, d_model)
        self.register_buffer(name='pe', tensor=pe)


        print('PositionalEncoding initialized.')


    def forward(self, x: torch.Tensor):
        """
            *   INPUT:
                        x --> shape: (batch_size, seq_len, d_model)
            *   OUTPUT:
                        x --> shape: (batch_size, seq_len, d_model)
        """

        # Clone to avoid modifying original tensors
        x = x.clone()

        x = x + self.pe[:, :x.size(dim=1), :]

        return x


class FeedForward(nn.Module):
    def __init__(self, cfg: dict) -> None:
        """
            Args:
                cfg = {
                    'd_model': 10,
                    'd_ff': 2048,
                    'dropout': 0.1
                }

        """

        super().__init__()

        self.config = cfg

        self.linear_1 = nn.Linear(in_features=self.config['d_model'],
                                  out_features=self.config['d_ff'])
        self.dropout = nn.Dropout(p=self.config['dropout'])
        self.linear_2 = nn.Linear(in_features=self.config['d_ff'],
                                  out_features=self.config['d_model'])


        print('FeedForward initialized.')


    def forward(self, x: torch.Tensor):
        """
            *   INPUT:
                        x --> shape: (batch_size, seq_len, d_model)
            *   OUTPUT:
                        x --> shape: (batch_size, seq_len, d_model)
        """

        # Clone to avoid modifying original tensors
        x = x.clone()

        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.linear_2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, cfg: dict):
        """
            Args:
                cfg = {
                    'd_model': 10,
                    'attention_heads': 4,
                    'feedforward': {
                        'd_model': 10,
                        'd_ff': 2048,
                        'dropout': 0.1
                    }
                }

        """

        super().__init__()

        self.config = cfg

        self.check_config()

        self.feedforward = FeedForward(cfg=self.config['feedforward'])
        self.attention = nn.MultiheadAttention(embed_dim=self.config['d_model'],
                                               num_heads=self.config['attention_heads'])
        self.layer_norm = nn.LayerNorm(normalized_shape=self.config['d_model'])


        print('Encoder initialized.')


    def forward(self, x: torch.Tensor):
        """
            *   INPUT:
                        x --> shape: (batch_size, seq_len, d_model)
            *   OUTPUT:
                        x --> shape: (batch_size, seq_len, d_model)
        """

        # Clone to avoid modifying original tensors
        x = x.clone()

        # Multi-Head Attention
        attention_output, _ = self.attention(query = x,
                                             key = x,
                                             value = x,
                                             attn_mask = None)

        x = attention_output + x

        x = self.layer_norm(x)

        # Feedforward
        feedforward_output = self.feedforward(x)

        x = feedforward_output + x

        x = self.layer_norm(x)

        return x


    def check_config(self):
        if self.config['d_model'] != self.config['feedforward']['d_model']:
            raise ValueError('d_model must be equal to d_model in feedforward')

        if self.config['d_model'] % self.config['attention_heads'] != 0:
            raise ValueError('d_model must be divisible by attention_heads')
        

class Decoder(nn.Module):
    def __init__(self, cfg: dict):
        """
            Args:
                cfg = {
                'd_model': 10,
                'attention_heads': 4,
                'feedforward': {
                    'd_model': 10,
                    'd_ff': 2048,
                    'dropout': 0.1
                }
            }

        """

        super().__init__()

        self.config = cfg

        self.check_config()

        self.feedforward = FeedForward(cfg = self.config['feedforward'])
        self.masked_attention = nn.MultiheadAttention(embed_dim=self.config['d_model'],
                                                      num_heads=self.config['attention_heads'])
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.config['d_model'],
                                                     num_heads=self.config['attention_heads'])  # Cross-Attention (Encoder-Decoder Attention)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.config['d_model'])


        print('Decoder initialized.')


    def forward(self, decoder_input: torch.Tensor, encoder_output: torch.Tensor):
        """
            *   INPUT:
                        decoder_input --> shape: (batch_size, seq_len_decoder, d_model)
                        encoder_output --> shape: (batch_size, seq_len_encoder, d_model)
            *   OUTPUT:
                        decoder_output --> shape: (batch_size, seq_len_decoder, d_model)
        """

        # Clone to avoid modifying original tensors
        decoder_input = decoder_input.clone()
        encoder_output = encoder_output.clone()

        batch_size, seq_len_decoder, d_model = decoder_input.size()

        # ----------- Attention mechanism ------------
        # Re-Arrange Shape
        decoder_input = decoder_input.permute(1, 0, 2)  # shape: (seq_len_decoder, batch_size, d_model)
        encoder_output = encoder_output.permute(1, 0, 2)  # shape: (seq_len_encoder, batch_size, d_model)

        # Generate an attention mask
        attn_mask = torch.triu(torch.ones(seq_len_decoder, seq_len_decoder) * float('-inf'), diagonal=1).to(decoder_input.device)

        masked_attention_output, _ = self.masked_attention(query=decoder_input,
                                                           key=decoder_input,
                                                           value=decoder_input,
                                                           attn_mask=attn_mask)  # (seq_len_decoder, batch_size, d_model)

        # Re-Arrange Shape
        masked_attention_output = masked_attention_output.permute(1, 0, 2)  # shape: (batch_size, seq_len_decoder, d_model)
        decoder_input = decoder_input.permute(1, 0, 2)  # shape: (batch_size, seq_len_decoder, d_model)
        encoder_output = encoder_output.permute(1, 0, 2)  # shape: (batch_size, seq_len_encoder, d_model)
        # ----------- End: Attention mechanism ------------

        # Add and Normalize
        decoder_output = masked_attention_output + decoder_input  # (batch_size, seq_len_decoder, d_model)
        decoder_output = self.layer_norm(decoder_output)  # (batch_size, seq_len_decoder, d_model)

        # ----------- Cross-Attention -----------
        # Re-Arrange Shape
        decoder_output = decoder_output.permute(1, 0, 2)  # shape: (seq_len_decoder, batch_size, d_model)
        encoder_output = encoder_output.permute(1, 0, 2)  # shape: (seq_len_encoder, batch_size, d_model)

        cross_attention_output, _ = self.cross_attention(query=decoder_output,
                                                         key=encoder_output,
                                                         value=encoder_output,
                                                         attn_mask = None)  # (seq_len_decoder, batch_size, d_model)

        # Re-Arrange Shape
        cross_attention_output = cross_attention_output.permute(1, 0, 2)  # shape: (batch_size, seq_len_decoder, d_model)
        decoder_output = decoder_output.permute(1, 0, 2)  # shape: (batch_size, seq_len_decoder, d_model)
        encoder_output = encoder_output.permute(1, 0, 2)  # shape: (batch_size, seq_len_encoder, d_model)
        # ----------- End: Cross-Attention -----------

        # Add and Normalize
        decoder_output = cross_attention_output + decoder_output  # shape: (batch_size, seq_len_decoder, d_model)
        decoder_output = self.layer_norm(decoder_output)  # shape: (batch_size, seq_len_decoder, d_model)

        # Feed Forward
        feedforward_output = self.feedforward(decoder_output)  # shape: (batch_size, seq_len_decoder, d_model)

        decoder_output = feedforward_output + decoder_output  # shape: (batch_size, seq_len_decoder, d_model)

        decoder_output = self.layer_norm(decoder_output)  # shape: (batch_size, seq_len_decoder, d_model)

        return decoder_output


    def check_config(self):
        if self.config['d_model'] != self.config['feedforward']['d_model']:
            raise ValueError('d_model must be equal to d_model in feedforward')

        if self.config['d_model'] % self.config['attention_heads'] != 0:
            raise ValueError('d_model must be divisible by attention_heads')


class Transformer(nn.Module):
    def __init__(self, cfg: dict):
        """
            Args:
                cfg = {
                    'pos_encoder': {
                        'd_model': 10,
                        'max_len': 5000
                    },
                    'num_encoders': 5,
                    'num_decoders': 5,
                    'encoder': {
                        'd_model': 10,
                        'attention_heads': 4,
                        'feedforward': {
                            'd_model': 10,
                            'd_ff': 2048,
                            'dropout': 0.1
                        }
                    },
                    'decoder': {
                        'd_model': 10,
                        'attention_heads': 4,
                        'feedforward': {
                            'd_model': 10,
                            'd_ff': 2048,
                            'dropout': 0.1
                        }
                    },
                    'final_linear': {
                        'in_features': 10,
                        'out_features': 10
                    }
                }
                
        """

        super().__init__()

        self.config = cfg

        self.check_config()

        self.pos_encoder = PositionalEncoding(cfg=self.config['pos_encoder'])
        self.encoder = Encoder(cfg=self.config['encoder'])
        self.decoder = Decoder(cfg=self.config['decoder'])
        self.final_linear = nn.Linear(in_features=self.config['final_linear']['in_features'],
                                       out_features=self.config['final_linear']['out_features'])


        print('Transformer initialized.')


    def check_config(self):
        if self.config['pos_encoder']['d_model'] != self.config['encoder']['d_model']:
            raise ValueError('d_model in encoder must be equal to d_model in pos_encoder')

        if self.config['pos_encoder']['d_model'] != self.config['decoder']['d_model']:
            raise ValueError('d_model in decoder must be equal to d_model in pos_encoder')

        if self.config['encoder']['d_model'] != self.config['decoder']['d_model']:
            raise ValueError('d_model in decoder must be equal to d_model in encoder')

        if self.config['final_linear']['in_features'] != self.config['decoder']['d_model']:
            raise ValueError('in_features in final_linear must be equal to d_model in decoder')


    def forward(self, input_embedding: torch.Tensor, output_embedding: torch.Tensor):
        """
            *   INPUT:
                        input_embedding --> shape: (batch_size, seq_len_inputs, d_model)
                        output_embedding --> shape: (batch_size, seq_len_outputs, d_model)
            *   OUTPUT:
                        model_output --> shape: (batch_size, final_linear['out_features'])
        """

        # Clone to avoid modifying original tensors
        input_embedding = input_embedding.clone()
        output_embedding = output_embedding.clone()

        batch_size, seq_len_inputs, d_model = input_embedding.size()
        _, seq_len_outputs, _ = output_embedding.size()

        # Positional Encoding
        input_embedding = self.pos_encoder(input_embedding)  # shape: (batch_size, seq_len_inputs, d_model)
        output_embedding = self.pos_encoder(output_embedding)  # shape: (batch_size, seq_len_outputs, d_model)

        # Encoder
        encoder_output = input_embedding
        for _ in range(self.config['num_encoders']):
            encoder_output = self.encoder(encoder_output)  # shape: (batch_size, seq_len_inputs, d_model)

        # Decoder
        decoder_output = output_embedding
        for _ in range(self.config['num_decoders']):
            decoder_output = self.decoder(decoder_input = decoder_output, encoder_output = encoder_output)  # shape: (batch_size, seq_len_outputs, d_model)

        # Final Linear Layer
        model_output = self.final_linear(decoder_output[:, -1, :])  # shape: (batch_size, final_linear['out_features'])

        return model_output