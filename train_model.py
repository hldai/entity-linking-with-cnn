__author__ = 'dhl'


import theano
import theano.tensor as T
import numpy as np

from theano_cnn import LeNetConvPoolLayer, HiddenLayer, sgd_updates_adadelta

import data_load

def_filter_hs = [2, 3]
sentence_len = 50
sentence_pad_len = def_filter_hs[-1] - 1
training_part_size = 25000


def relu(x):
    return T.maximum(0.0, x)


def to_theano_shared(vals):
    return theano.shared(value=np.asarray(vals,
                                          dtype=theano.config.floatX),
                         borrow=True)


def train_cnn_for_el(train_data_file_name,
                     val_data_file_name,
                     num_val_candidates,
                     test_data_file_name,
                     num_test_candidates,
                     img_h, img_w,
                     all_words,  # first row of all_words should be a non-existing word
                     wid_idx_dict,
                     entity_vecs,
                     gold_as_first_candidate=False,
                     skip_width_loading=40,  # skip width while loading samples
                     n_epochs=25,
                     batch_size=50,
                     filter_hs=def_filter_hs,
                     num_feature_maps=100,
                     conv_non_linear="relu",
                     lr_decay=0.9,
                     sqr_norm_lim=9,
                     hidden_out_len=50,):
    rng = np.random.RandomState(3435)

    x = T.imatrix('x')
    # es = T.imatrix('es')
    # es_test = T.imatrix('es_test')
    entities = T.imatrix('entities')

    print 'making entity_vecs...', len(entity_vecs)
    shared_entity_vecs = theano.shared(value=np.asarray(entity_vecs, dtype=theano.config.floatX),
                                       name='entity_vecs', borrow=True)
    # shared_entity_vecs = theano.shared(value=np.asarray(entity_vecs, dtype=np.float32),
    #                                    name='entity_vecs', borrow=True)
    print 'making shared_words...', len(all_words)
    shared_words = theano.shared(value=np.asarray(all_words, dtype=theano.config.floatX),
                                 name='shared_words', borrow=True)
    print 'done'

    # test_contexts, test_indices = get_data_set_full(test_data_file_name, wid_idx_dict, skip_width_loading)
    # num_test_batches = test_indices.shape[0] / batch_size
    # num_val_contexts, val_contexts, val_indices = get_data_set_full(val_data_file_name,
    #                                                                 wid_idx_dict, skip_width_loading)
    val_contexts, val_indices = data_load.load_samples_full(val_data_file_name, wid_idx_dict, sentence_len,
                                                            sentence_pad_len,
                                                            skip_width=skip_width_loading,
                                                            num_candidates=num_val_candidates)
    num_val_batches = len(val_contexts) / batch_size
    print num_val_batches, 'validation batches'
    print len(val_indices[0]), 'candidates per mention'

    if gold_as_first_candidate:
        gold_labels = theano.shared(value=np.zeros(batch_size,
                                                   dtype='int32'),
                                    borrow=True)
    else:
        gold_labels = theano.shared(value=np.ones(batch_size,
                                                  dtype='int32'),
                                    borrow=True)

    val_contexts = T.cast(to_theano_shared(val_contexts), 'int32')
    val_indices = T.cast(to_theano_shared(val_indices), 'int32')

    filter_shapes = []
    pool_sizes = []
    filter_w = img_w
    for filter_h in filter_hs:
        filter_shapes.append((num_feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h - filter_h + 1, img_w - filter_w + 1))

    layer0_input = shared_words[x.flatten()].reshape((x.shape[0], 1, x.shape[1], shared_words.shape[1]))
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size, 1, img_h, img_w),
                                        filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)

    layer1_input = T.concatenate(layer1_inputs, 1)
    matcher0 = HiddenLayer(rng, layer1_input, num_feature_maps * len(filter_hs),
                           hidden_out_len, relu)
    mc = matcher0.output  # mention contexts

    unit_mc = mc / T.sqrt(T.maximum(T.sum(T.sqr(mc), 1), 0.0001)).dimshuffle(0, 'x')

    batch_entity_vecs = shared_entity_vecs[entities]
    matcher1 = HiddenLayer(rng, batch_entity_vecs, len(entity_vecs[0]), hidden_out_len, relu)
    entity_reps = matcher1.output
    # entity_reps = batch_entity_vecs

    unit_entity_reps = entity_reps / T.sqrt(T.maximum(T.sum(T.sqr(entity_reps), 2), 0.0001)).dimshuffle(0, 1, 'x')

    similarities = (unit_mc.dimshuffle(0, 'x', 1) * unit_entity_reps).sum(axis=2)
    correct_rate = T.mean(T.eq(gold_labels, T.argmax(similarities, axis=1)))

    loss = T.maximum(0, 1 - similarities[:, 0] + similarities[:, 1]).sum()

    # similarities = (mc.dimshuffle(0, 'x', 1) * batch_entity_vecs).sum(axis=2)  # / mc_norm

    params = matcher0.params + matcher1.params
    # params = matcher0.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    grad_updates = sgd_updates_adadelta(params, loss, lr_decay, 1e-6, sqr_norm_lim)

    index = T.lscalar()

    # test_model = theano.function(
    #     [index],
    #     error_rate,
    #     givens={x: test_contexts[index * batch_size: (index + 1) * batch_size],
    #             es: test_indices[index * batch_size: (index + 1) * batch_size]}
    # )

    val_model = theano.function(
        [index],
        correct_rate,
        givens={x: val_contexts[index * batch_size: (index + 1) * batch_size],
                entities: val_indices[index * batch_size: (index + 1) * batch_size]}
    )

    train_contexts = theano.shared(
        value=np.zeros((3, 2)),
        borrow=True)
    int_train_contexts = T.cast(train_contexts, 'int32')
    train_indices = theano.shared(
        value=np.zeros((3, 2)),
        borrow=True)
    int_train_indices = T.cast(train_indices, 'int32')
    train_model = theano.function(
        [index],
        loss,
        updates=grad_updates,
        givens={x: int_train_contexts[index * batch_size: (index + 1) * batch_size],
                entities: int_train_indices[index * batch_size: (index + 1) * batch_size]}
    )

    fdebug = theano.function(
        [index],
        similarities,
        givens={x: int_train_contexts[index * batch_size: (index + 1) * batch_size],
                entities: int_train_indices[index * batch_size: (index + 1) * batch_size]}
    )
    # print fdebug(0)

    val_perfs = [val_model(i) for i in xrange(num_val_batches)]
    print('init val perf %f' % np.mean(val_perfs))

    print 'training ...'
    f_train = open(train_data_file_name, 'rb')
    epoch = 0
    while epoch < n_epochs:
        epoch += 1

        train_part_cnt = 0
        # num_train_contexts, cur_train_contexts, cur_train_indices = get_data_set_part(
        #     f_train, wid_idx_dict, 50000)
        cur_train_contexts, cur_train_indices = data_load.load_training_samples(f_train,
                                                                                training_part_size,
                                                                                wid_idx_dict,
                                                                                sentence_len,
                                                                                sentence_pad_len)
        while not len(cur_train_contexts) == 0:
            train_contexts.set_value(cur_train_contexts, borrow=True)
            train_indices.set_value(cur_train_indices, borrow=True)
            # print fdebug(0)

            train_part_cnt += 1
            num_train_batches = len(cur_train_contexts) / batch_size
            # print 'num_train_batches', num_train_batches
            mean_loss = 0
            for minibatch_index in xrange(num_train_batches):
                cur_loss = train_model(minibatch_index)
                mean_loss += cur_loss
                # if (minibatch_index + 1) % (num_train_batches / 3) == 0:  # show some progress
                #     print minibatch_index, num_train_batches
            print 'loss:', mean_loss / num_train_batches
            # print fdebug(0)

            val_perfs = [val_model(i) for i in xrange(num_val_batches)]
            val_perf = np.mean(val_perfs)
            print('epoch %i, training part %i, val perf %f'
                  % (epoch, train_part_cnt, val_perf))
            cur_train_contexts, cur_train_indices = data_load.load_training_samples(f_train,
                                                                                    training_part_size,
                                                                                    wid_idx_dict,
                                                                                    sentence_len,
                                                                                    sentence_pad_len)
            # num_train_contexts, cur_train_contexts, cur_train_indices = get_data_set_part(
            #     f_train, wid_idx_dict, 50000)

    f_train.close()


def main():
    _, word_vecs = data_load.load_word_vectors('/media/dhl/Data/el/word2vec/wiki_vectors.jbin')
    word_vec_len = len(word_vecs[0])

    wid_idx_dict, entity_vecs = data_load.load_entities(
        '/media/dhl/Data/el/vec_rep/wid_entity_rep_wiki50_cat.bin',
        False)
    # wid_idx_dict, entity_vecs = data_load.load_entities('/media/dhl/Data/el/vec_rep/wid_entity_rep_wiki50.bin',
    #                                                     True)

    # all_word_vecs =
    num_val_candidates = 30
    num_test_candidates = 30
    skipwidth_loading = 0
    img_h = sentence_len + 2 * sentence_pad_len
    train_cnn_for_el('/media/dhl/Data/el/vec_rep/wiki_train_word_vec_indices_wiki50.td',
                     '/media/dhl/Data/el/vec_rep/tac_2014_training.bin',
                     # '/media/dhl/Data/el/vec_rep/wiki_val_word_vec_indices_wiki50.td',
                     num_val_candidates,
                     '/media/dhl/Data/el/vec_rep/wiki_test_word_vec_indices_wiki50.td',
                     num_test_candidates,
                     img_h, word_vec_len,
                     word_vecs,
                     wid_idx_dict,
                     entity_vecs,
                     skip_width_loading=skipwidth_loading,
                     n_epochs=1)


if __name__ == '__main__':
    main()
