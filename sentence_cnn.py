__author__ = 'dhl'

import theano.tensor as T

from theano_cnn import LeNetConvPoolLayer, HiddenLayer, relu


class SentenceCNN:
    def __init__(self, input_sentences, shared_words, full_sentence_len, word_vec_len, filter_hs, num_feature_maps,
                 batch_size,
                 output_len, rng, conv_non_linear=relu,
                 hidden_W=None, hidden_b=None, conv_Ws=None, conv_bs=None):
        # self.input_x = input_x
        self.input = input_sentences
        self.non_linear = conv_non_linear

        # batch_size = input_sentences.shape[0]
        # full_sentence_len = input_sentences.shape[1]
        # word_vec_len = shared_words.shape[1]

        filter_shapes = []
        pool_sizes = []
        filter_w = word_vec_len
        for filter_h in filter_hs:
            filter_shapes.append((num_feature_maps, 1, filter_h, filter_w))
            pool_sizes.append((full_sentence_len - filter_h + 1, word_vec_len - filter_w + 1))

        layer0_input = shared_words[input_sentences.flatten()].reshape((input_sentences.shape[0], 1,
                                                                        input_sentences.shape[1],
                                                                        shared_words.shape[1]))
        conv_layers = []
        layer1_inputs = []
        for i in xrange(len(filter_hs)):
            filter_shape = filter_shapes[i]
            pool_size = pool_sizes[i]
            conv_W = None
            conv_b = None
            if conv_Ws is not None:
                conv_W = conv_Ws[i]
            if conv_bs is not None:
                conv_b = conv_bs[i]
            conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,
                                            image_shape=(batch_size, 1, full_sentence_len, word_vec_len),
                                            filter_shape=filter_shape, poolsize=pool_size,
                                            non_linear=conv_non_linear.func_name,
                                            W=conv_W, b=conv_b)
            layer1_input = conv_layer.output.flatten(2)
            conv_layers.append(conv_layer)
            layer1_inputs.append(layer1_input)

        layer1_input = T.concatenate(layer1_inputs, 1)
        matcher0 = HiddenLayer(rng, layer1_input, num_feature_maps * len(filter_hs),
                               output_len, relu, W=hidden_W, b=hidden_b)

        self.hiddenW = matcher0.W
        self.hiddenb = matcher0.b
        self.convWs = list()
        self.convbs = list()
        for conv_layer in conv_layers:
            self.convWs.append(conv_layer.W)
            self.convbs.append(conv_layer.b)

        self.output = matcher0.output  # mention contexts
        self.params = matcher0.params
        for conv_layer in conv_layers:
            self.params += conv_layer.params

        # unit_mc = mc / T.sqrt(T.maximum(T.sum(T.sqr(mc), 1), 0.0001)).dimshuffle(0, 'x')
