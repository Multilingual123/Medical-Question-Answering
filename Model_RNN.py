import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

from Evaluation import evaluation


def Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target, epochs=20):
    num_words = 10000
    max_q_len = 50
    max_a_len = 50
    embedding_dim = 128
    units = 128
    Train_Data = [str(x) for x in Train_Data.reshape(-1)]
    Train_Target = [str(x) for x in Train_Target.reshape(-1)]
    Test_Data = [str(x) for x in Test_Data.reshape(-1)]
    Test_Target = [str(x) for x in Test_Target.reshape(-1)]
    q_tokenizer = Tokenizer(num_words=num_words, filters='')
    q_tokenizer.fit_on_texts(Train_Data)
    a_tokenizer = Tokenizer(num_words=num_words, filters='')
    a_tokenizer.fit_on_texts(["<sos> " + t + " <eos>" for t in Train_Target])
    # encode questions
    X_train = pad_sequences(q_tokenizer.texts_to_sequences(Train_Data), maxlen=max_q_len, padding='post')
    X_test = pad_sequences(q_tokenizer.texts_to_sequences(Test_Data), maxlen=max_q_len, padding='post')
    # encode answers with <sos>/<eos>
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
    encoder_inputs = layers.Input(shape=(max_q_len,))
    enc_emb = layers.Embedding(input_dim=len(q_tokenizer.word_index) + 1,
                               output_dim=embedding_dim, mask_zero=True)(encoder_inputs)
    enc_out, state_h, state_c = layers.LSTM(units, return_state=True)(enc_emb)

    # Decoder
    decoder_inputs = layers.Input(shape=(max_a_len,))
    dec_emb = layers.Embedding(input_dim=vocab_size,
                               output_dim=embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = layers.LSTM(units, return_sequences=True, return_state=True)
    dec_out, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    decoder_dense = layers.Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(dec_out)

    model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit([X_train, y_train_in],
              np.expand_dims(y_train_out, -1),
              epochs=epochs, batch_size=16, verbose=1)
    pred = model.predict([X_test, y_test_in])
    pred_ids = np.argmax(pred, axis=-1)
    inv_vocab = {v: k for k, v in a_tokenizer.word_index.items()}
    decoded_sentences = []
    for seq in pred_ids:
        words = []
        for idx in seq:
            if idx == 0:  # padding
                continue
            word = inv_vocab.get(idx, "")
            if word == "<eos>":
                break
            if word not in ["<sos>", "<pad>"]:
                words.append(word)
        decoded_sentences.append(" ".join(words))
    Eval = evaluation(pred, y_test_out)
    return Eval, np.asarray(decoded_sentences)



