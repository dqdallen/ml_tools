import tensorflow as tf
from tensorflow.estimator import TrainSpec, EvalSpec, Estimator


class DemoEstimator(Estimator):
    def __init__(self, dataset, config, params):
        def model_fn(features, labels, mode, params):
            embbedding_out = ... # create embedding layer
            dense1 = tf.layers.dense(inputs=embedding_out, units=128, activation=tf.nn.relu)
            dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
            dense2 = tf.layers.dense(inputs=dropout1, units=64, activation=tf.nn.relu)
            dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
            logits = tf.layers.dense(inputs=dropout2, units=1)
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = tf.transpose(tf.reshape(logits, [-1]))
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs={'output': tf.estimator.export.PredictOutput(predictions)})
            label_clk = ...
            label_clk_float32 = tf.cast(label_clk, tf.float32)
            loss_ctr = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_clk_float32, logits=logits)
            loss = tf.reduce_mean(loss_ctr)
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])
            tf.summary.scalar('loss', loss)
            auc_ctr = tf.metrics.auc(labels=label_clk, predictions=tf.sigmoid(logits))
            if mode == tf.estimator.ModeKeys.TRAIN:
                metric_ops = {'auc_ctr': auc_ctr[1]}
                for metric_name, metric_value in metric_ops.items():
                    tf.summary.scalar(metric_name, metric_value)
            summary_op = tf.summary.merge_all()
            summary_hook = tf.train.SummarySaverHook(save_steps=100, output_dir=params['summary_dir'], summary_op=summary_op)
            logging_hook = tf.train.LoggingTensorHook({'loss': loss, 'lossL2': lossL2, 'auc_ctr': auc_ctr[1]}, every_n_iter=100)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
                # train_op = optimizer.minimize(loss + params['l2_reg'] * lossL2, global_step=tf.train.get_global_step())
                grads, variables = zip(*optimizer.compute_gradients(loss + params['l2_reg'] * lossL2))
                grads, global_norm = tf.clip_by_global_norm(grads, 5.0)
                train_op = optimizer.apply_gradients(zip(grads, variables), global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[summary_hook, logging_hook])
            tf.summary.scalar('loss', loss)
            summary_op = tf.summary.merge_all()
            summary_hook = tf.train.SummarySaverHook(save_steps=100, output_dir=params['summary_dir'], summary_op=summary_op)
            eval_metric_ops = {'auc_ctr': auc_ctr}
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=[summary_hook])
        super(DemoEstimator, self).__init__(model_fn=model_fn, config=config, params=params)
