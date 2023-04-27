import tensorflow as tf

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


example_texts = ['abcdefg', 'xyz']

# textをtokenに分割する
chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')

# vocabを使用して、StringLookupレイヤー作成
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
# 各tokenを文字IDに変換する
ids = ids_from_chars(chars)
# print(ids) # => <tf.RaggedTensor [[40, 41, 42, 43, 44, 45, 46], [63, 64, 65]]>

# この表現を反転して、human-readableな文字列を復元する機能も用意
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
# IDのベクトルから文字を復元する
chars = chars_from_ids(ids)
# print(chars) # => <tf.RaggedTensor [[b'a', b'b', b'c', b'd', b'e', b'f', b'g'], [b'x', b'y', b'z']]>

# 文字列に結合し直す
def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# print(text_from_ids(ids)) # => tf.Tensor([b'abcdefg' b'xyz'], shape=(2,), dtype=string)
# print(text_from_ids(ids).numpy()) # => [b'abcdefg' b'xyz']


all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
# for ids in ids_dataset.take(10):
#     print(chars_from_ids(ids).numpy().decode('utf-8'))

seq_length = 100
example_per_epoch = len(text)//(seq_length + 1)

sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

for seq in sequences.take(5):
    print(text_from_ids(seq).numpy())