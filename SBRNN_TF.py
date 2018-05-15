"""========================================================================
This file contains the code for the sliding bidirectional recurrent
neural network detector.

@author: Nariman Farsad
@copyright: Copyright 2018
========================================================================"""

import numpy as np
import tensorflow as tf
import os


class ModelConfig(object):
    """ The model parameters of the SBRNN detector
    """
    def __init__(self,hidden_size=80,
                 numb_layers=3,
                 peephole=True,
                 input_size=14,
                 dropout=0.2,
                 sliding=True,
                 output_size=2,
                 cellType='LSTM',
                 norm_gain=0.1,
                 norm_shift=0.0):
        self.hidden_size = hidden_size
        self.numb_layers = numb_layers
        self.peephole = peephole
        self.dropout = dropout
        self.sliding = sliding
        self.input_size = input_size
        self.output_size = output_size
        self.cell_type = cellType
        self.norm_gain = norm_gain
        self.norm_shift = norm_shift


class BRNN(object):
    """ A Multilayer Bi-directional Recurrent Neural Network
    """
    def __init__(self ,input, input_len, config):
        """
        Args:
            input: input to the BRNN
            input_len: the length of inputs

            config: includes model configuration parameters
        """
        self.input = input
        self.input_len = input_len
        self.config = config

        self.output, self.predictions, self.probs = self.build_enc_network()



    def build_enc_network(self):
        """Builds the BRNN network
        """

        if self.config.cell_type=='GRU':
            fw_cells = [tf.contrib.rnn.GRUCell(num_units=self.config.hidden_size)
                             for _ in range(self.config.numb_layers) ]
            bw_cells = [tf.contrib.rnn.GRUCell(num_units=self.config.hidden_size)
                             for _ in range(self.config.numb_layers)]
        elif self.config.cell_type=='LSTM':
            fw_cells = [tf.contrib.rnn.LSTMCell(num_units=self.config.hidden_size,
                                                use_peepholes=self.config.peephole)
                        for _ in range(self.config.numb_layers)]
            bw_cells = [tf.contrib.rnn.LSTMCell(num_units=self.config.hidden_size,
                                                use_peepholes=self.config.peephole)
                        for _ in range(self.config.numb_layers)]
        elif self.config.cell_type=='LSTM-Norm':
            fw_cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.config.hidden_size,
                                                              norm_gain=self.config.norm_gain,
                                                              norm_shift=self.config.norm_shift)
                        for _ in range(self.config.numb_layers)]
            bw_cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.config.hidden_size,
                                                              norm_gain=self.config.norm_gain,
                                                              norm_shift=self.config.norm_shift)
                        for _ in range(self.config.numb_layers)]
        else:
            fw_cells = [tf.contrib.rnn.BasicRNNCell(num_units=self.config.hidden_size)
                        for _ in range(self.config.numb_layers)]
            bw_cells = [tf.contrib.rnn.BasicRNNCell(num_units=self.config.hidden_size)
                        for _ in range(self.config.numb_layers)]

        (outputs, fw_final_state,
         bw_final_state) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=fw_cells,
                                                                          cells_bw=bw_cells,
                                                                          inputs=self.input,
                                                                          dtype=tf.float32,
                                                                          sequence_length=self.input_len)


        allout = tf.reshape(outputs, [-1, self.config.hidden_size*2])

        softmax_w = tf.get_variable(
            "softmax_w", [self.config.hidden_size*2, self.config.output_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.config.output_size], dtype=tf.float32)

        logits = tf.nn.xw_plus_b(allout, softmax_w, softmax_b)

        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [tf.shape(outputs)[0], -1, self.config.output_size])

        probs = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 2)

        return logits, predictions, probs


class SBRNN_Detector(object):
    """ The sliding bidirectional recurrent neural network detector
    """
    def __init__(self, config):
        self.config = config

        self.input = tf.placeholder(shape=(None, None, config.input_size),
                                    dtype=tf.float32, name='inputs')
        self.input_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='inputs_length')
        self.lr = tf.placeholder(dtype=tf.float32, name='learning_rate')
        self.targets = tf.placeholder(shape=(None, None), dtype=tf.int64, name='targets')



        self.brnn = BRNN(self.input,self.input_len, self.config)

        # ==== define loss and training op and accuracy ====
        self.loss, self.train_op = self.loss_and_train_op(self.brnn.output,self.targets)
        self.accuracy = self.define_accuracy(self.brnn.predictions,self.targets)

        # ==== set up training/updating procedure ====
        # self.saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=0.25)
        self.saver = tf.train.Saver(max_to_keep=10)

        tf.summary.scalar("CrossEntLoss", self.loss)
        self.tb_summary = tf.summary.merge_all()
        self.tb_val_summ = tf.summary.scalar("Validation_Accuracy", self.accuracy)

    def load_trained_model(self, sess, trained_model_path):
        """
        Loads a trained model from what was saved. Insert the trained model path
        """
        trained_model_folder = os.path.split(trained_model_path)[0]
        ckpt = tf.train.get_checkpoint_state(trained_model_folder)
        v2_path = os.path.join(trained_model_folder, os.path.split(ckpt.model_checkpoint_path)[1] + ".index")
        norm_ckpt_path = os.path.join(trained_model_folder, os.path.split(ckpt.model_checkpoint_path)[1])
        if ckpt and (tf.gfile.Exists(norm_ckpt_path) or
                         tf.gfile.Exists(v2_path)):
            print("Reading model parameters from %s" % norm_ckpt_path)
            self.saver.restore(sess, norm_ckpt_path)
        else:
            print('Error reading weights')


    def define_accuracy(self,predictions,targets):
        """
        Accuracy is defined as the ratio of correctly detected symbols
        """
        eq_indicator = tf.cast(tf.equal(predictions, targets), dtype=tf.float32)
        return tf.reduce_mean(eq_indicator)

    def loss_and_train_op (self, logits, targets):
        """
        Loss and optimization algorithm
        """
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.one_hot(targets, depth=self.config.output_size, dtype=tf.float32),
            logits=logits)
        optimizer = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return loss, optimizer

    def train_on_samples(self, sess, nn_input, labels, lr, len_ss):
        """
        This trains the network by creating all sub sequences from the input
        and then using those for training
        Args
            sess: Tensorflow session instance
            nn_input: A batch of input sequences
            labels: A batch target values
            lr: learning rate
            len_ss: the length of the sub sequences (i.e., the maximum block
                    length for SBRNN detector
        Returns
            loss: the loss values for the batch
            accu: the accuracy for the batch

        """
        if nn_input.shape[1]>len_ss:
            inps, _, targs = self.create_sub_seq(nn_input,len_ss,labels)
        else:
            inps = nn_input
            targs = labels

        input_len = np.zeros(inps.shape[0]) + len_ss
        fd = {self.input: inps, self.targets: targs,
              self.input_len: input_len, self.lr: lr}
        _, loss, accu = sess.run([self.train_op, self.loss, self.accuracy], fd)
        return loss, accu


    def create_sub_seq(self, nn_input, len_ss, labels=None):
        """
        This function creates all sub sequences for the batch
        """
        n_seq = nn_input.shape[0]
        len_seq = nn_input.shape[1]
        n_ss = len_seq - len_ss + 1
        new_labels = []
        new_inp = np.zeros((n_ss*n_seq,len_ss,nn_input.shape[2]))
        if labels is not None:
            new_labels = np.zeros((n_ss*n_seq,len_ss))
        k = 0
        for i in range(n_seq):
            for j in range(n_ss):
                new_inp[k, :, :] = nn_input[i, j:j + len_ss, :]
                if labels is not None:
                    new_labels[k, :] = labels[i, j:j + len_ss]
                k += 1

        return new_inp, n_ss, new_labels

    def preds_to_symbs(self, preds, type="mean"):
        """
        During testing use combine all the estimates of the SBRNN and it slides
        using mean or median
        """
        diags = [preds[::-1, :].diagonal(i) for i in range(-preds.shape[0] + 1, preds.shape[1])]
        if type == "mean":
            symbs = [np.argmax(np.mean(x, axis=1)) for _, x in enumerate(diags)]
        else:
            symbs = [np.argmax(np.median(x, axis=1)) for _, x in enumerate(diags)]
        return np.array(symbs)

    def test_on_sample(self, sess, nn_input, label, len_ss, type="mean"):
        """
        Test the trained SBRNN detector on test samples. The input is first broken into
        sub sequences of length "len_ss". The BRNN is used on each sub sequence and the
        results are then combined using mean to median.
        """
        n_seq = label.shape[0]
        if nn_input.shape[1] > len_ss:
            new_input, n_ss, _ = self.create_sub_seq(nn_input, len_ss)
        else:
            new_input = nn_input
            n_ss = 1

        input_lens = np.zeros(new_input.shape[0]) + len_ss
        fd = {self.input: new_input, self.input_len: input_lens}
        probs = sess.run([self.brnn.probs], fd)
        probs = probs[0]

        pred_symb = np.zeros_like(label)
        for i in range(n_seq):
            pred_symb[i] = self.preds_to_symbs(probs[i*n_ss:(i+1)*n_ss],type=type)

        errors = np.sum(pred_symb != label)

        return pred_symb, errors





if __name__=="__main__":
    config = ModelConfig(cellType='LSTM-Norm')

    sbrnn = SBRNN_Detector(config)
