#!/usr/bin/python3

from os.path import join;
import tensorflow as tf;
from load_cora import load_cora;
from Model import GCN;

def main():

    # load dataset
    features, labels, adj = load_cora();
    # create graph convolutional network
    gcn = GCN(input_dim = features.shape[-1], hidden_dim = 16, output_dim = labels.shape[-1], adj = adj);
    # train context
    optimizer = tf.keras.optimizers.Adam(1e-2);
    checkpoint = tf.train.Checkpoint(model = gcn, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
    # create log
    log = tf.summary.create_file_writer('checkpoints');
    avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
    while True:
        with tf.GradientTape() as tape:
            outputs = gcn(features);
            loss = 0;
            for variable in gcn.trainable_variables:
                loss += 5e-4 * tf.nn.l2_loss(variable);
            loss += tf.keras.losses.CategoricalCrossentropy(from_logits = True)(labels, outputs);
        avg_loss.update_state(loss);
        # apply gradients
        grads = tape.gradient(loss, gcn.trainable_variables);
        optimizer.apply_gradients(zip(grads, gcn.trainable_variables));
        # write log
        if tf.equal(optimizer.iterations % 10, 0):
            with log.as_default():
                tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
            print('Step #%d Loss: %.6f' % (optimizer.iterations, avg_loss.result()));
            avg_loss.reset_states();
        # save checkpoint
        checkpoint.save(join('checkpoints','ckpt'));
        gcn.save_weights('gcn.h5');

if __name__ == "__main__":

    assert True == tf.executing_eagerly();
    main();
