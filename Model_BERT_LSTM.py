import numpy as np
import tensorflow as tf
from keras.src.optimizers.adam import Adam
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras import Model
from keras import backend as k
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
from Evaluation import evaluation


def masked_multihead_cross_attention(query, key, value, mask, num_heads=8, key_dim=64, name="cross_attention"):
    mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name=name)

    def make_mask(inputs):
        q, m = inputs
        q_len = tf.shape(q)[1]
        m = tf.expand_dims(m, axis=1)  # (B, 1, S)
        return tf.tile(m, [1, q_len, 1])  # (B, Tq, S)

    mask_3d = layers.Lambda(make_mask)([query, mask])

    attn_out = mha(query=query, key=key, value=value, attention_mask=mask_3d)
    out = layers.Add()([query, attn_out])
    out = layers.LayerNormalization()(out)
    return out


def Model_BERT(Data, Target, sol, max_len=128, epochs=1, batch_size=16):
    if np.issubdtype(Data.dtype, np.number):
        Data = [" ".join(map(str, row)) for row in Data]
    else:
        Data = Data.astype(str).flatten().tolist()

    Target = Target.astype(str).flatten().tolist()

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    X_train = tokenizer(Data, truncation=True, padding=True,
                        max_length=max_len, return_tensors="tf")

    classes = np.unique(Target)
    label_to_id = {c: i for i, c in enumerate(classes)}
    y_train = np.array([label_to_id[t] for t in Target])
    num_labels = len(classes)
    bert_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
    outputs = bert_model(input_ids, attention_mask=attention_mask)
    enc_seq = outputs.last_hidden_state  # (B, S, H)
    # Apply self-attention on BERT sequence
    attn_out = masked_multihead_cross_attention(enc_seq, enc_seq, enc_seq, attention_mask,
                                                sol[0], 12)
    cls_output = attn_out[:, 0, :]  # take [CLS]
    out = layers.Dense(num_labels, activation="softmax")(cls_output)

    model = Model(inputs=[input_ids, attention_mask], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit([X_train["input_ids"], X_train["attention_mask"]],
              y_train, validation_split=0.1,
              batch_size=batch_size, epochs=epochs, verbose=1)
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [k.function([inp], [out]) for out in outputs]  # evaluation functions
    layerNo = 4
    layer_out = np.asarray(functors[layerNo]([X_train])).squeeze()
    return layer_out


def Model_BERT_LSTM(Data, Target, epochs=20, sol=None):
    if sol is None:
        sol = [5, 0.01, 5]
    Feature = Model_BERT(Data, Target, sol)  # Use BERT features
    learnperc = round(Feature.shape[0] * 0.75)
    Train_Data = Feature[:learnperc, :]
    Train_Target = Target[:learnperc, :]
    Test_Data = Feature[learnperc:, :]
    Test_Target = Target[learnperc:, :]

    max_a_len = 50
    embedding_dim = 128

    Train_Target = [str(x) for x in Train_Target.reshape(-1)]
    Test_Target = [str(x) for x in Test_Target.reshape(-1)]

    a_tokenizer = Tokenizer(num_words=10000, filters='')
    a_tokenizer.fit_on_texts(["<sos> " + t + " <eos>" for t in Train_Target])

    y_train_in = pad_sequences(a_tokenizer.texts_to_sequences(["<sos> " + t for t in Train_Target]),
                               maxlen=max_a_len, padding='post')
    y_train_out = pad_sequences(a_tokenizer.texts_to_sequences([t + " <eos>" for t in Train_Target]),
                                maxlen=max_a_len, padding='post')
    y_test_in = pad_sequences(a_tokenizer.texts_to_sequences(["<sos> " + t for t in Test_Target]),
                              maxlen=max_a_len, padding='post')
    y_test_out = pad_sequences(a_tokenizer.texts_to_sequences([t + " <eos>" for t in Test_Target]),
                               maxlen=max_a_len, padding='post')

    vocab_size = len(a_tokenizer.word_index) + 1
    # Encoder
    encoder_inputs = layers.Input(shape=(Train_Data.shape[1], Train_Data.shape[2]))
    enc_out, state_h, state_c = layers.LSTM(sol[2], return_sequences=True, return_state=True)(encoder_inputs)
    # Decoder
    decoder_inputs = layers.Input(shape=(max_a_len,))
    dec_emb = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = layers.LSTM(sol[2], return_sequences=True, return_state=True)
    dec_out, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    # Cross-Attention between decoder output and encoder output
    enc_mask = layers.Input(shape=(1,), dtype=tf.int32, name="enc_mask")  # dummy mask
    cross_out = masked_multihead_cross_attention(dec_out, enc_out, enc_out, enc_mask,
                                                 num_heads=8, key_dim=64)
    decoder_dense = layers.Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(cross_out)
    model = models.Model([encoder_inputs, decoder_inputs, enc_mask], decoder_outputs)
    model.compile(optimizer=Adam(learning_rate=sol[1]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit([Train_Data, y_train_in, np.ones((Train_Data.shape[0], 1))],
              np.expand_dims(y_train_out, -1),
              epochs=epochs, batch_size=16, verbose=1)

    pred = model.predict([Test_Data, y_test_in, np.ones((Test_Data.shape[0], 1))])
    pred_ids = np.argmax(pred, axis=-1)

    inv_vocab = {v: s for s, v in a_tokenizer.word_index.items()}
    decoded_sentences = []
    for seq in pred_ids:
        words = []
        for idx in seq:
            if idx == 0:
                continue
            word = inv_vocab.get(idx, "")
            if word == "<eos>":
                break
            if word not in ["<sos>", "<pad>"]:
                words.append(word)
        decoded_sentences.append(" ".join(words))

    Eval = evaluation(pred, y_test_out)
    return Eval, np.asarray(decoded_sentences)

