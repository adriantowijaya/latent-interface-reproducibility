from __future__ import annotations

import numpy as np

from .tarela_py2.config import PY2Config
from .tarela_py2.model_tf import sparsemax_tf


def sinusoidal_encoding(tf, length: int = 7, d_model: int = 32):
    pos = np.arange(length)[:, None]
    i = np.arange(d_model)[None, :]
    angles = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
    pe = np.zeros((length, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.constant(pe[None, :, :], dtype=tf.float32)


def build_gru_receiver_substitution(cfg: PY2Config, tf, layers):
    class SparseLatentRegimeEncoder(tf.keras.Model):
        def __init__(self, hidden_dim: int, topic_dim: int):
            super().__init__()
            self.dense1 = layers.Dense(hidden_dim, activation="relu")
            self.dense2 = layers.Dense(topic_dim)

        def call(self, x):
            return sparsemax_tf(self.dense2(self.dense1(x)), tf)

    class TARELAGRUReceiverSubstitution(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.regime_encoder = SparseLatentRegimeEncoder(cfg.encoder_hidden, cfg.topic_dim)
            self.gru = layers.GRU(50, activation="tanh", recurrent_activation="sigmoid", use_bias=True, reset_after=True, dropout=0.0, recurrent_dropout=0.0)
            self.output_layer = layers.Dense(1)
            self.dir_head = layers.Dense(3)
            self.vol_head = layers.Dense(3)
            self.amp_head = layers.Dense(1)

        def call(self, inputs, training=False):
            x_y_revin, x_num, x_doc = inputs
            theta_seq = self.regime_encoder(x_doc)
            x = tf.concat([x_y_revin, x_num, theta_seq], axis=-1)
            h = self.gru(x, training=training)
            theta_last = theta_seq[:, -1, :]
            return self.output_layer(h), theta_seq, self.dir_head(theta_last), self.vol_head(theta_last), self.amp_head(theta_last)

    return TARELAGRUReceiverSubstitution()


def build_transformer_receiver_substitution(cfg: PY2Config, tf, layers):
    class SparseLatentRegimeEncoder(tf.keras.Model):
        def __init__(self, hidden_dim: int, topic_dim: int):
            super().__init__()
            self.dense1 = layers.Dense(hidden_dim, activation="relu")
            self.dense2 = layers.Dense(topic_dim)

        def call(self, x):
            return sparsemax_tf(self.dense2(self.dense1(x)), tf)

    class CompactTransformerReceiverSubstitution(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.regime_encoder = SparseLatentRegimeEncoder(cfg.encoder_hidden, cfg.topic_dim)
            self.input_projection = layers.Dense(32)
            self.pre_attn_norm = layers.LayerNormalization(epsilon=1e-3)
            self.self_attention = layers.MultiHeadAttention(num_heads=4, key_dim=8, dropout=0.0)
            self.pre_ffn_norm = layers.LayerNormalization(epsilon=1e-3)
            self.ffn_1 = layers.Dense(96, activation="gelu")
            self.ffn_2 = layers.Dense(32)
            self.final_norm = layers.LayerNormalization(epsilon=1e-3)
            self.output_layer = layers.Dense(1)
            self.dir_head = layers.Dense(3)
            self.vol_head = layers.Dense(3)
            self.amp_head = layers.Dense(1)
            self.positional_encoding = sinusoidal_encoding(tf, 7, 32)

        def call(self, inputs, training=False):
            x_y_revin, x_num, x_doc = inputs
            theta_seq = self.regime_encoder(x_doc)
            fused = tf.concat([x_y_revin, x_num, theta_seq], axis=-1)
            x = self.input_projection(fused) + self.positional_encoding[:, : tf.shape(fused)[1], :]
            attn_in = self.pre_attn_norm(x)
            x = x + self.self_attention(attn_in, attn_in, use_causal_mask=True, training=training)
            ffn_in = self.pre_ffn_norm(x)
            x = x + self.ffn_2(self.ffn_1(ffn_in))
            h = self.final_norm(x)[:, -1, :]
            theta_last = theta_seq[:, -1, :]
            return self.output_layer(h), theta_seq, self.dir_head(theta_last), self.vol_head(theta_last), self.amp_head(theta_last)

    return CompactTransformerReceiverSubstitution()
