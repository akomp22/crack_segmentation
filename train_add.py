import tensorflow as tf



def weighted_cross_entropy(beta):
    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):

        y_pred = convert_to_logits(y_pred)

        pos = tf.reduce_sum(y_true)
        neg = tf.reduce_sum(tf.cast(y_true==0,dtype = tf.float32))
        w = neg/pos
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=w*beta)
        return tf.reduce_mean(loss)

    return loss


def scheduler(epoch):
        return 0.00005 * tf.math.exp(0.002 * (1 - epoch))
