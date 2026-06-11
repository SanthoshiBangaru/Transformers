import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    Dense,
    LayerNormalization,
    MultiHeadAttention
)

# --------------------------
# CONFIG
# --------------------------

VOCAB_SIZE = 1000
SEQUENCE_LENGTH = 20

# --------------------------
# CUSTOM LAYERS
# --------------------------

class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(
        self,
        sequence_length,
        vocab_size,
        embed_dim,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_embedding = Embedding(
            vocab_size,
            embed_dim
        )

        self.position_embedding = Embedding(
            sequence_length,
            embed_dim
        )

    def call(self, inputs):

        length = tf.shape(inputs)[-1]

        positions = tf.range(
            start=0,
            limit=length,
            delta=1
        )

        embedded_tokens = self.token_embedding(inputs)
        embedded_positions = self.position_embedding(positions)

        return embedded_tokens + embedded_positions

    def get_config(self):

        config = super().get_config()

        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        })

        return config

    def call(self, inputs):

        length = tf.shape(inputs)[-1]

        positions = tf.range(
            start=0,
            limit=length,
            delta=1
        )

        embedded_tokens = self.token_embedding(inputs)

        embedded_positions = self.position_embedding(
            positions
        )

        return embedded_tokens + embedded_positions


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(
        self,
        embed_dim,
        dense_dim,
        num_heads,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.dense_proj = tf.keras.Sequential([
            Dense(dense_dim, activation="relu"),
            Dense(embed_dim)
        ])

        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()

    def call(self, inputs):

        attention_output = self.attention(
            inputs,
            inputs
        )

        proj_input = self.layernorm1(
            inputs + attention_output
        )

        proj_output = self.dense_proj(proj_input)

        return self.layernorm2(
            proj_input + proj_output
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads
        })

        return config
    
class TransformerDecoder(tf.keras.layers.Layer):

    def __init__(
        self,
        embed_dim,
        dense_dim,
        num_heads,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.self_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = tf.keras.Sequential([
            Dense(dense_dim, activation="relu"),
            Dense(embed_dim)
        ])

        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()

    def call(self, inputs, encoder_outputs):

        attention_output = self.self_attention(
            query=inputs,
            value=inputs,
            key=inputs,
            use_causal_mask=True
        )

        out1 = self.layernorm1(
            inputs + attention_output
        )

        attention_output2 = self.cross_attention(
            query=out1,
            value=encoder_outputs,
            key=encoder_outputs
        )

        out2 = self.layernorm2(
            out1 + attention_output2
        )

        ffn_output = self.ffn(out2)

        return self.layernorm3(
            out2 + ffn_output
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads
        })

        return config

# --------------------------
# LOAD VOCABS
# --------------------------

with open("source_vocab.pkl", "rb") as f:
    source_vocab = pickle.load(f)

with open("target_vocab.pkl", "rb") as f:
    target_vocab = pickle.load(f)

source_vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    vocabulary=source_vocab
)

target_vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    vocabulary=target_vocab
)

# --------------------------
# LOAD MODEL
# --------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "translator.keras",
        custom_objects={
            "PositionalEmbedding": PositionalEmbedding,
            "TransformerEncoder": TransformerEncoder,
            "TransformerDecoder": TransformerDecoder
        },
        compile=False
    )

    return model

transformer = load_model()

# --------------------------
# LOOKUP
# --------------------------

index_lookup = dict(
    zip(
        range(len(target_vocab)),
        target_vocab
    )
)

# --------------------------
# TRANSLATE
# --------------------------

def translate(sentence):

    encoder_input_test = source_vectorization(
        [sentence.lower()]
    )

    decoded_sentence = "start"

    for _ in range(SEQUENCE_LENGTH - 1):

        tokenized_target = target_vectorization(
            [decoded_sentence]
        )[:, :-1]

        predictions = transformer.predict(
            [encoder_input_test, tokenized_target],
            verbose=0
        )

        current_pos = (
            len(decoded_sentence.split()) - 1
        )

        sampled_token_index = np.argmax(
            predictions[0, current_pos, :]
        )

        sampled_token = index_lookup.get(
            sampled_token_index,
            ""
        )

        decoded_sentence += " " + sampled_token

        if sampled_token == "end":
            break

    decoded_sentence = (
        decoded_sentence
        .replace("start", "")
        .replace("end", "")
        .strip()
    )

    return decoded_sentence

# --------------------------
# UI
# --------------------------

st.set_page_config(
    page_title="English → Telugu Translator",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 English → Telugu Translator")

text = st.text_area(
    "Enter English Text"
)

if st.button("Translate"):

    if text.strip():

        with st.spinner("Translating..."):

            result = translate(text)

        st.success("Translation Complete")

        st.markdown("### Telugu")

        st.write(result)