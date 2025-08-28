from pypots.nn.modules.timemixerpp.backbone import BackboneTimeMixerPP
import torch.nn as nn
import torch
import torch.nn.functional as F


class TimeMixerPPBackbone(BackboneTimeMixerPP):
    def __init__(
        self,
        n_features: int,
        n_layers: int,
        d_model: int,  # step embedding size
        d_ffn: int,
        n_heads: int,
        dropout: float,
        top_k: int,
        n_kernels: int,
        channel_mixing: bool,
        channel_independence: bool,
        downsampling_layers: int,
        downsampling_window: int,
        downsampling_method: str,
        use_future_temporal_feature: bool,
        use_norm: bool = False,
        embed="fixed",
        freq="h",
        n_classes=None,
    ):
        super(TimeMixerPPBackbone, self).__init__(
            task_name='short_term_forecast',
            n_steps=100,  # random number to avoid error in pypots
            n_features=n_features,
            n_pred_steps=10,  # random number to avoid error in pypots
            n_pred_features=10,  # random number to avoid error in pypots
            n_layers=n_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            n_heads=n_heads,
            dropout=dropout,
            top_k=top_k,
            n_kernels=n_kernels,
            channel_mixing=channel_mixing,
            channel_independence=channel_independence,
            downsampling_layers=downsampling_layers,
            downsampling_window=downsampling_window,
            downsampling_method=downsampling_method,
            use_future_temporal_feature=use_future_temporal_feature,
            use_norm=use_norm,
            embed=embed,
            freq=freq,
            n_classes=n_classes,
        )

        delattr(self, 'predict_layers')  # remove the original predict_layers
        delattr(self, 'projection_layer')  # remove the original projection_layer

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        # copy paste from pypots except for single change in the downsampling part
        # pypots implementation assumes the ts length is a power of 2, we fix that 
        # B,T,C -> B,C,T
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc
        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.downsampling_layers):
            if self.downsampling_method == "conv" and i == 0 and self.channel_independence:
                x_enc_ori = x_enc_ori.contiguous().reshape(B * N, T, 1).permute(0, 2, 1).contiguous()

            x_enc_sampling = self.down_pool(x_enc_ori)

            if self.downsampling_method == "conv":
                x_enc_sampling_list.append(
                    # x_enc_sampling.reshape(B, N, T // (self.downsampling_window ** (i + 1))).permute(0, 2, 1)
                    x_enc_sampling.reshape(B, N, -1).permute(0, 2, 1)
                )
            else:
                x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))

            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, :: self.downsampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, :: self.downsampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc
    
    def encode(self, x_enc, x_mark_enc=None):
        
        # adjusting mask to match the input shape
        if x_mark_enc is not None and x_mark_enc.dim() == 2:
            # x_enc: [B, T, N]
            B, T, N = x_enc.size()
            x_mark_enc = x_mark_enc.unsqueeze(-1).repeat(1, 1, N)

        # rest of the method remains unchanged
        # copy paste from pypots up until the encoder part
        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()

                x = self.revin_layers[i](x, x_mark, mode="norm") if self.use_norm else x  # TODO: validate normalizations (everythign related to revits)
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, T, N = x.size()
                x = self.revin_layers[i](x, mode="norm") if self.use_norm else x
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        if self.channel_mixing and self.channel_independence == 1:
            _, T, D = x_list[-1].size()

            coarse_scale_enc_out = x_list[-1].reshape(B, N, T * D)
            coarse_scale_enc_out, _ = self.channel_mixing_attention(
                coarse_scale_enc_out, coarse_scale_enc_out, coarse_scale_enc_out, None
            )
            x_list[-1] = coarse_scale_enc_out.reshape(B * N, T, D) + x_list[-1]

        enc_out_list = []
        if x_mark_enc is not None:
            for x, x_mark in zip(x_list, x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for x in x_list:
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        for i in range(self.n_layers):
            enc_out_list = self.encoder_model[i](enc_out_list)
        
        return enc_out_list

    def forward(self, x_enc, x_mark_enc=None):
        enc_out_list = self.encode(x_enc=x_enc, x_mark_enc=x_mark_enc)
        return enc_out_list