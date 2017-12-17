import glob
import tensorflow as tf
from etl import make_vocab, batch2nums, max_len

def build_vocab(filenames):
    all_lines = []
    for filename in filenames:
        with open(filename, 'r') as f:
            lines = f.readlines()
            all_lines = all_lines + lines
    word_dict, rev_dict = make_vocab(all_lines, None)
    return word_dict, rev_dict

def main(filenames, vocab):
    batch_size = 32
    for filename in filenames:
        with open(filename, 'r') as f:
            lines = f.readlines()
        longest_line = max_len(lines)
        story_len = get_story_len(lines)
        max_batch = get_max_batch(lines, batch_size, story_len)
        for j in range(max_batch):
            offset = batch_size * j
            # initialize s_h
            for i in range(story_len):
                lines = make_batch(lines, i, offset, batch_size, story_len)
                import pdb
                pdb.set_trace()
                sentences = get_sentences(lines)
                batch_data = batch2nums(sentences, vocab, longest_line)
                if is_ques(lines):
                    q = lstm(batch_data)
                    predict(q, s_h)
                else:
                    s_h = lstm(batch_data, s_h)



def get_sentences(lines):
    sents = []
    for line in lines:
        sent = line.split('\t')[1].strip()
        sents.append(sent)
    return sents


def make_batch(lines, i, offset, batch_size, story_len):
    sents = []
    for k in range(batch_size):
        sents.append(lines[offset + k*story_len + i])
    return sents



def get_max_batch(lines, batch_size, story_len):
    n_lines = len(lines)
    return n_lines//(batch_size*story_len)


def get_story_len(lines):
    story_num = 0
    for line in lines:
        contents = line.split('\t')
        if story_num > int(contents[0]):
            return story_num
        else:
            story_num = int(contents[0])
    return story_num



if __name__=="__main__":
    filenames = glob.glob('./en/*')
    vocab, _ = build_vocab(filenames)
    main(filenames, vocab)
