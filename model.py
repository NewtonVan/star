import torch
from typing import Optional, List, Tuple
from torch import nn


class FCWithActivation(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, activation: Optional[nn.Module]) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.activation = activation()

    def forward(self, inputs: torch.Tensor):
        return self.activation(self.linear(inputs))


class StarEncoderLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: int, num_layers: int, num_fc=3) -> None:
        super().__init__()
        # TODO figure out size
        self.fc = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(num_fc)])
        self.relu = nn.ReLU()
        self.lstm_encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for fc in self.fc:
            inputs = self.relu(fc(inputs))

        output, h_n, c_n = self.lstm_encoder(inputs)

        return output, h_n, c_n


class StarDecoderLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: int, num_layers: int, num_fc=3) -> None:
        super().__init__()
        self.lstm_decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, inputs: torch.Tensor, state: List[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, h_n, c_n = self.lstm_encoder(inputs)
        output = self.fc(output)

        return output, h_n, c_n


class StarEncoder(nn.Module):
    def __init__(self, dropout=0.0) -> None:
        super().__init__()
        input_size = 4
        embed_dim = 32
        # TODO
        hidden_size = 32
        num_layers = 1
        self.dropout = dropout

        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self):
        pass


# TODO
class StarAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> None:
        pass


# TODO
class Star(nn.Module):
    def __init__(self, input_shape: Tuple, num_heads=8, dropout=0., future_step=12) -> None:
        """

        @param input_shape: (T, N, F) T for frames, N for num of pedestrian, F for feature
        @param num_heads:
        @param dropout:
        @param future_step:
        """
        super().__init__()
        # parameter
        d_k = 16
        in_feat = input_shape[2]  # default 4
        embed_dim = 32
        out_feat = 2
        self.embed_dim = embed_dim
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.d_k = d_k
        self.future_step = future_step
        self.input_shape = input_shape
        # device
        self.device = torch.device('cuda')

        # map input to embedding
        self.input_embedding_layer = nn.Linear(in_feat, embed_dim)

        # todo struct reasonable?
        #  1: only 1 encoder divided into multi-parts
        #  2: `StarEncoder`
        # star encoder for every pedestrian
        self.star_encoder = torch.nn.ModuleList([StarEncoder(dropout) for _ in range(input_shape[1])])
        # world attention model
        # todo
        #   whether needed
        #   dim proper
        self.w_q = nn.Linear(embed_dim, d_k)
        self.w_k = nn.Linear(embed_dim, d_k)
        self.w_v = nn.Linear(embed_dim, d_k)
        # todo
        #  transformer better
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        # star decoder for every pedestrian
        self.star_decoder = torch.nn.ModuleList([StarEncoder(dropout) for _ in range(future_step)])

        # todo output & fusion
        self.output_layer = nn.Linear(embed_dim, out_feat)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        :param inputs: [B, T, N, F]
        """
        embed_inputs = self.input_embedding_layer(inputs)
        assert embed_inputs.shape[1] == self.input_shape[0], "inconsistency on frames"
        assert embed_inputs.shape[2] == self.input_shape[1], "inconsistency on number of pedestrian"
        assert embed_inputs.shape[3] == self.in_feat, "inconsistency on features"
        h_n: List[Optional[torch.Tensor]] = [None for _ in range(embed_inputs.shape[2])]
        c_n: List[Optional[torch.Tensor]] = [None for _ in range(embed_inputs.shape[2])]

        # todo more concurrency with reshape?
        attn_out_list: List[Optional[torch.Tensor]] = [None for _ in range(embed_inputs.shape[1])]
        for frame in range(embed_inputs.shape[1]):
            # query space attention
            # [B, T, N, F] -> [B, N, F]
            embed_inputs_frame = embed_inputs[:, frame, :, :]
            attn_outputs_frame = self.attn(self.w_q(embed_inputs_frame), self.w_k(embed_inputs_frame),
                                           self.w_v(embed_inputs_frame))
            # idea of residual
            attn_outputs_frame = attn_outputs_frame + embed_inputs_frame
            attn_outputs_frame = self.attn_norm(attn_outputs_frame)
            attn_out_list[frame] = attn_outputs_frame

        attn_outputs = torch.stack(attn_out_list, dim=1)

        # replace former loop
        # [B, T, N, F] -> [B*T, N, F]
        # attn_input = torch.reshape(embed_inputs, (-1, embed_inputs.shape[2], embed_inputs.shape[3]))
        # attn_outputs = self.attn(self.w_q(attn_input), self.w_k(attn_input), self.w_v(attn_input))
        # attn_outputs = torch.reshape(attn_outputs, (-1, embed_inputs.shape[1], embed_inputs.shape[2],
        #                                             embed_inputs.shape[3]))

        # contain all users encode result in current frame
        encoder_inputs: Optional[torch.Tensor] = None
        encoder_outputs: List[Optional[torch.Tensor]] = [None for _ in range(embed_inputs.shape[2])]
        # encoder: use query result from world model for every single frame
        # train time continuity
        for ped_id in range(embed_inputs.shape[2]):
            # [B, T, N, F] -> [B, T, F]
            encoder_inputs: torch.Tensor = attn_outputs[:, :, ped_id, :]
            outputs, h_n[ped_id], c_n[ped_id] = self.star_encoder[ped_id](encoder_inputs, h_n[ped_id], c_n[ped_id])
            # [B, T, F] -> [B, 1, F] we only care about last moment
            encoder_outputs[ped_id] = outputs[:, -1:, :]

        print("h_n size: ")
        print(h_n[0].size())
        assert encoder_inputs.shape == (embed_inputs.shape[0], embed_inputs.shape[1], embed_inputs.shape[3]), \
            "wrong shape: encoder_inputs"
        assert len(encoder_outputs) == embed_inputs.shape[2], "wrong lth: encoder outputs"
        assert encoder_outputs[0].shape == encoder_inputs.shape, "wrong shape encoder output"

        decoder_outputs = encoder_outputs
        future_outputs = list()
        for frame in range(self.future_step):
            # contain all users decode result in current frame
            for ped_id, decoder in enumerate(self.star_decoder):
                outputs, h_0, c_0 = decoder(decoder_outputs[ped_id], h_n[ped_id], c_n[ped_id])
                decoder_outputs[ped_id] = outputs

            # [B, 1, F] -> [B, N, F]
            future_outputs.append(torch.cat(decoder_outputs, dim=1))

        assert len(future_outputs) == self.future_step, "wrong lth: future outputs"
        assert future_outputs[0].shape == (embed_inputs.shape[0], embed_inputs.shape[2], self.in_feat), \
            "wrong shape: future output"

        # [B, N, F] -> [B, TF, N, F]
        future_outputs = torch.stack(future_outputs, dim=1)
        future_outputs = self.output_layer(future_outputs)

        return future_outputs

    def save(self) -> None:
        pass
