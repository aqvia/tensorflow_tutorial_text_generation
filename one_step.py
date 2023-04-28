import tensorflow as tf


class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # "[UNK]"が生成されないようにマスクを作成する
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # bad index に -inf を設定する
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # shape を vocabulary に合わせる
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # strings を token IDs に変換する
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # モデルを実行する
        # predicted_logits.shape は [batch, char, next_char_logits]
        predicted_logits, states = self.model(
            inputs=input_ids, states=states, return_state=True)
        # 最後の予測だけ使用する
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # 予測マスクを適用する: "[UNK]"が生成されないように
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # token IDs を文字に変換する
        predicted_chars = self.chars_from_ids(predicted_ids)

        # 文字とモデル状態を返す
        return predicted_chars, states
