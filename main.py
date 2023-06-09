import tensorflow as tf
import os
import time

from my_model import MyModel
from one_step import OneStep

# シェイクスピアのデータセットをダウンロード
path_to_file = tf.keras.utils.get_file(
    'shakespeare.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# print(f'Length of text: {len(text)} characters')
# print(text[:250])

# ファイルのuniqueな文字
vocab = sorted(set(text))
# print(f'{len(vocab)} unique charactors')


# vocabを使用して、StringLookupレイヤー作成
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)

# この表現を反転して、human-readableな文字列を復元する機能も用意
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)


def text_from_ids(ids):
    """文字列に結合し直す

    Args:
        ids (tf.RaggedTensor): 文字IDのテンソル

    Returns:
        tf.Tensor(dtype=string): 文字列のテンソル
    """
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


# 例
example_texts = ['abcdefg', 'xyz']
# textをtokenに分割する
chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
# 各tokenを文字IDに変換する
ids = ids_from_chars(chars)
# print(ids) # => <tf.RaggedTensor [[40, 41, 42, 43, 44, 45, 46], [63, 64, 65]]>
# IDのベクトルから文字を復元する
chars = chars_from_ids(ids)
# print(chars) # => <tf.RaggedTensor [[b'a', b'b', b'c', b'd', b'e', b'f', b'g'], [b'x', b'y', b'z']]>
# print(text_from_ids(ids)) # => tf.Tensor([b'abcdefg' b'xyz'], shape=(2,), dtype=string)
# print(text_from_ids(ids).numpy()) # => [b'abcdefg' b'xyz']


# textをexample sequenceに分割する。各input sequenceはseq_length文字を含む。
# textをseq_length+1の長さのchunkに分割する
# e.g. seq_length=4, chunk="Hello" => input seq.="Hell", target seq.="ello"

# tokenに分割したtextを、ベクトルに変換
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
# テキストベクトルを文字インデックスのstreamに変換する
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
# for ids in ids_dataset.take(10):
#     print(chars_from_ids(ids).numpy().decode('utf-8'))

seq_length = 100
example_per_epoch = len(text)//(seq_length + 1)
sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)
# for seq in sequences.take(5):
#     print(text_from_ids(seq).numpy())


def split_input_target(sequence):
    """シーケンスを受け取り、各タイムステップの入力とラベルを揃える

    Args:
        sequence (tf.RaggedTensor)

    Returns:
        (tf.RaggedTensor, tf.RaggedTensor)
    """
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)


# トレーニングバッチを作成
# データをシャッフルし、バッチにパックする
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))


# モデル構築
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

model = MyModel(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units
)


# # モデルを試す
# for input_example_batch, target_example_batch in dataset.take(1):
#     example_batch_predictions = model(input_example_batch)
#     # 出力の形状を確認
#     print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
# # summary出力
# model.summary()
# # 実際の予測を取得
# sampled_indices = tf.random.categorical(
#     example_batch_predictions[0], num_samples=1)
# sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
# print(sampled_indices)
# print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
# print()
# print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())


# モデルをトレーニングする
# 損失関数
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
# print("Mean loss:", example_batch_mean_loss)
# print(tf.exp(example_batch_mean_loss).numpy())

# オプティマイザと損失関数のアタッチ
model.compile(optimizer='adam', loss=loss)

# トレーニング中におけるチェックポイントの保存
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# トレーニングの実行
EPOCHS = 20
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


# テキスト生成
# シングルステップ予測
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

start = time.time()
status = None
next_char = tf.constant(['ROMEO:'])
result = [next_char]

for n in range(1000):
    next_char, states = one_step_model.generate_one_step(
        next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)
print(f'\nRun time: {end - start}')
