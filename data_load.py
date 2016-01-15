#! /usr/bin/python

__author__ = 'dhl'

import numpy
import math
import copy

def load_word_vectors(file_path):
    print 'loading word vectors...'
    f = open(file_path, 'rb')
    num_words = numpy.fromfile(f, '>i4', 1)
    vec_len = numpy.fromfile(f, '>i4', 1)
    print num_words, vec_len
    words = []
    # word_vecs = []
    # word_vecs.append(0, [0. for i in xrange(word_vec_len)])
    word_vecs = numpy.zeros((num_words + 1, vec_len))
    # word_vecs[0][0] = 1
    for i in xrange(num_words):
        length = ord(f.read(1))
        byte_buf = f.read(length)
        if i == 0:
            words.append(byte_buf.decode('utf-8'))
        words.append(byte_buf.decode('utf-8'))
        word_vecs[i + 1] = numpy.fromfile(f, '>f4', vec_len)
        # vec = numpy.fromfile(f, '>f4', vec_len)
        # word_vecs.append(vec)

    f.close()
    print 'done.'

    return words, word_vecs


def load_index_vec_of_entities_fixed_len(file_path):
    print 'loading representations of entities (word indices, fixed len) ...'
    f = open(file_path, 'rb')
    num_entities = numpy.fromfile(f, '>i4', 1)
    print num_entities
    vec_len = numpy.fromfile(f, '>i4', 1)[0]
    print vec_len

    wid_idx_dict = dict()
    wid_idx_dict[0] = 0
    entity_vecs = numpy.zeros((num_entities + 1, vec_len), dtype='int32')
    for i in xrange(num_entities):
        wid = numpy.fromfile(f, '>i4', 1)
        wid_idx_dict[wid[0]] = i + 1

        # print num_indices
        entity_vecs[i + 1] = numpy.fromfile(f, '>i4', vec_len)

        # print i
        if (i + 1) % 1000000 == 0:
            print i + 1

    f.close()

    print 'done.'
    return wid_idx_dict, entity_vecs, vec_len


def load_entities_indices(file_path, max_num_words=50, pad_len=1):
    print 'loading representations of entities (word indices) ...'

    f = open(file_path, 'rb')
    num_entities = numpy.fromfile(f, '>i4', 1)
    print num_entities

    wid_idx_dict = dict()
    wid_idx_dict[0] = 0
    entity_vecs = numpy.zeros((num_entities + 1, max_num_words + 2 * pad_len), dtype='int32')
    for i in xrange(num_entities):
        wid = numpy.fromfile(f, '>i4', 1)
        wid_idx_dict[wid[0]] = i + 1

        num_indices = numpy.fromfile(f, '>i4', 1)
        # print num_indices
        indices = numpy.fromfile(f, '>i4', num_indices)
        for j in xrange(num_indices):
            if j < max_num_words:
                entity_vecs[i + 1][pad_len + j] = indices[j] + 1
            else:
                break

        # print i
        if (i + 1) % 1000000 == 0:
            print i + 1

    f.close()

    print 'done.'
    return wid_idx_dict, entity_vecs


def load_entities(file_path, div_by_len=False, unknown_vec=None):
    print 'loading entity representations ...'
    f = open(file_path, 'rb')

    num_entities = numpy.fromfile(f, '>i4', 1)
    vec_len = numpy.fromfile(f, '>i4', 1)

    print num_entities, vec_len

    wid_idx_dict = dict()
    wid_idx_dict[0] = 0
    entity_vecs = numpy.zeros((num_entities + 1, vec_len))
    if unknown_vec is None:
        # entity_vecs[0] = numpy.random.uniform(low=0, high=1, size=(vec_len,))
        entity_vecs[0][0] = 1
    else:
        entity_vecs[0] = unknown_vec
    cnt = 1
    while True:
        wid = numpy.fromfile(f, '>i4', 1)

        if not wid:
            break

        entity_vecs[cnt] = numpy.fromfile(f, '>f4', vec_len)
        if div_by_len:
            l2_norm = 0
            for i in xrange(vec_len):
                l2_norm += entity_vecs[cnt][i] * entity_vecs[cnt][i]
            l2_norm = math.sqrt(l2_norm)
            for i in xrange(vec_len):
                entity_vecs[cnt][i] /= l2_norm

        wid_idx_dict[wid[0]] = cnt

        # print entity_vecs[cnt]
        cnt += 1
        # if cnt == 10:
        #     break
        if cnt % 1000000 == 0:
            print cnt

    f.close()
    print 'done.'

    return wid_idx_dict, entity_vecs


def skip_next_training_paragraph(f):
    sentence_len = numpy.fromfile(f, '>i4', 1)
    if not sentence_len:
        return False

    word_indices = numpy.fromfile(f, '>i4', sentence_len)

    num_mentions = numpy.fromfile(f, '>i4', 1)
    for i in xrange(num_mentions):
        mention_span = numpy.fromfile(f, '>i4', 2)
        num_mention_candidates = numpy.fromfile(f, '>i4', 1)

        candidates = numpy.fromfile(f, '>i4', num_mention_candidates)


def load_next_training_paragraph(f):
    sentence_len = numpy.fromfile(f, '>i4', 1)
    if not sentence_len:
        return False

    word_indices = numpy.fromfile(f, '>i4', sentence_len)

    num_mentions = numpy.fromfile(f, '>i4', 1)
    mention_spans = []
    candidates_mentions = []
    for i in xrange(num_mentions):
        mention_span = numpy.fromfile(f, '>i4', 2)
        if mention_span is None:
            print i, num_mentions, 'weird'
        # [mention_beg, mention_end] = numpy.fromfile(f, '>i4', 2)
        # mention_spans.append((mention_beg, mention_end))
        if not len(mention_span) == 2:
            print i, num_mentions, 'weird'
            print mention_span
        [mention_beg, mention_end] = mention_span
        mention_spans.append([mention_beg, mention_end])
        num_mention_candidates = numpy.fromfile(f, '>i4', 1)

        if num_mention_candidates > 37:
            print 'num_mention_candidates', num_mention_candidates
            return False
        # else:
        #     print 'num_mention_candidates', num_mention_candidates

        candidates = numpy.fromfile(f, '>i4', num_mention_candidates)
        candidates_mentions.append(candidates)

    return word_indices, mention_spans, candidates_mentions


def get_mention_centered_context(word_indices, mention_span, sentence_len, pad_len):
    result_indices = []
    for i in xrange(pad_len):
        result_indices.append(0)

    len_mention_span = mention_span[1] - mention_span[0] + 1
    len_side = (sentence_len - len_mention_span) / 2
    pos_left = mention_span[0] - len_side
    pos_right = mention_span[1] + len_side

    if pos_left < 0:
        pos_right -= pos_left
        if pos_right >= len(word_indices):
            pos_right = len(word_indices) - 1
        pos_left = 0
    elif pos_right >= len(word_indices):
        pos_left -= pos_right - len(word_indices) + 1
        if pos_left < 0:
            pos_left = 0
        pos_right = len(word_indices) - 1

    for pos in xrange(pos_left, pos_right + 1):
        cur_word_index = word_indices[pos]
        if cur_word_index > -1:
            result_indices.append(cur_word_index + 1)

    # print sentence_len, pad_len

    while len(result_indices) < sentence_len + 2 * pad_len:
        result_indices.append(0)

    return result_indices


def get_samples_in_paragraph(word_indices, mention_spans, candidates_mentions,
                             wid_idx_dict, dst_contexts, dst_entity_idxs,
                             sentence_len, sentence_pad_len, num_candidates=2):
    cnt0 = 0
    cnt1 = 0
    if len(mention_spans) == len(candidates_mentions):
        entity_indices = numpy.zeros(num_candidates, dtype='int32')
        for mention_span, candidate_mentions in zip(mention_spans, candidates_mentions):
            if len(candidate_mentions) == 1:
                cnt0 += 1
                continue

            pos = 0
            last_index = 0
            while pos < len(candidate_mentions) and pos < num_candidates:
                idx = wid_idx_dict.get(candidate_mentions[pos], 0)
                entity_indices[pos] = idx
                last_index = idx
                pos += 1

            while pos < num_candidates:
                entity_indices[pos] = last_index
                pos += 1

            mention_context = get_mention_centered_context(word_indices, mention_span, sentence_len, sentence_pad_len)
            dst_contexts.append(mention_context)
            dst_entity_idxs.append(copy.copy(entity_indices))
    else:
        print 'number of mention spans does not match number of candidates of mentions'

    return cnt0, cnt1


def load_training_samples(f, num_paragraphs, wid_idx_dict, sentence_len, sentence_pad_len):
    print 'loading data', num_paragraphs, 'paragraphs'
    contexts = []
    entity_idxs = []
    for i in xrange(num_paragraphs):
        result_tuple = load_next_training_paragraph(f)
        if result_tuple:
            word_indices, mention_spans, candidates_mentions = result_tuple
            get_samples_in_paragraph(word_indices, mention_spans, candidates_mentions,
                                     wid_idx_dict, contexts, entity_idxs,
                                     sentence_len, sentence_pad_len)
        else:
            return contexts, entity_idxs

    print 'done.'
    return contexts, entity_idxs


def skip_training_sample(f, num_paragraphs):
    for i in xrange(num_paragraphs):
        skip_next_training_paragraph(f)


def load_samples_full(file_name, wid_idx_dict, sentence_len, sentence_pad_len, skip_width=20, num_candidates=2):
    print 'loading', file_name, '...'
    contexts = []
    entity_idxs = []
    f = open(file_name, 'rb')
    cnt = 0
    result_tuple = load_next_training_paragraph(f)
    while result_tuple:
        if skip_width == 0 or cnt % skip_width == 0:
            word_indices, mention_spans, candidates_mentions = result_tuple
            get_samples_in_paragraph(word_indices, mention_spans, candidates_mentions,
                                     wid_idx_dict, contexts, entity_idxs,
                                     sentence_len, sentence_pad_len, num_candidates)
            # print entity_idxs
        cnt += 1

        if cnt % 500000 == 0:
            print cnt
        result_tuple = load_next_training_paragraph(f)

    f.close()
    print 'done.'
    return contexts, entity_idxs


def main():
    print 'data_load'
    # words, word_vecs = load_word_vectors('/media/dhl/Data/el/word2vec/wiki_vectors.jbin')
    # for i in xrange(300):
    #     print words[i]

    # f = open('/media/dhl/Data/el/vec_rep/wiki_training_word_vec_indices.td', 'rb')
    # word_indices, mention_spans, candidates_mentions = load_next_training_paragraph(f)
    # f.close()

    # for i in range(len(word_indices)):
    #     if word_indices[i] > -1:
    #         print i, words[word_indices[i]]

    # print word_indices
    # print mention_spans
    # mention_context = get_mention_centered_context(word_indices, mention_spans[0])
    # print mention_context

    # wid_idx_dict, entity_vecs = load_entities('/media/dhl/Data/el/vec_rep/wid_entity_rep_wiki50_unit_vec.bin', False)

    # wid_idx_dict, entity_vecs = load_entities_indices('/media/dhl/Data/el/vec_rep/wid_entity_rep_wiki50_indices.bin')
    # print wid_idx_dict[12]
    # print entity_vecs[wid_idx_dict[12]]
    # for idx in entity_vecs[wid_idx_dict[12]]:
    #     print words[idx]


if __name__ == '__main__':
    main()
