from nltk.tokenize import word_tokenize
from functools import partial
from sklearn.model_selection import train_test_split
from collections import Counter

def make_vocab(texts, vocab_size, add_end=True):
    words = [word for sentence in texts for word in word_tokenize(sentence)]
    common_words = [item for item, c in Counter(words).most_common(vocab_size)]
    rev_dict = dict(enumerate(common_words))
    if vocab_size:
        rev_dict[len(rev_dict)] = 'UNK'
    if add_end:
        rev_dict[len(rev_dict)] = "<<END>>"
    word_dict = {y:x for x, y in rev_dict.items()}
    return word_dict, rev_dict


def _sent2nums(word_dict, seq_len, sent, add_end=True):
    words = word_tokenize(str(sent))
    nums = list(map(lambda x: word_dict[x]
                    if x in word_dict
                    else word_dict['UNK'],
                    words))
    nums = nums + [word_dict["<<END>>"]]*(seq_len - len(nums))
    return nums[:seq_len]

def batch2nums(sents, word_dict, seq_len):
    sent2nums_partial = partial(_sent2nums, word_dict, seq_len)
    batch_data = list(map(sent2nums_partial, sents))

    return batch_data


def max_len(data):
    return max(map(lambda x: len(word_tokenize(x)), data))


def class_weights(classes):
    counts = classes.value_counts()
    weights = [sum(counts) / count for count in counts]
    return weights


def sort_data(df, by_col):
    fun = lambda x: len(word_tokenize(x))
    df = df.assign(f = df[by_col].map(fun))
    sorted_df = df.sort_values('f')
    df = sorted_df.drop('f', axis=1)
    return df

def char_form(sentences, seq_len, max_chars):
    # @TODO Clear this part of code
    char_form = torch.LongTensor(batch_size, seq_len, max_chars)
    for i, sentence in enumerate(sentences):
        words = word_tokenize(sentence)
        words = (words * seq_len)[:seq_len]
        for j, word in enumerate(words):
            char_form[i][j] = torch.LongTensor(_chars2id(word))

    return Variable(char_form)


def _chars2id(chars):
    chars = list(map(lambda x: re.sub("[^" + string.ascii_lowercase + "]", "", x), chars))
    char_ids = list(map(string.ascii_lowercase.index, chars))
    char_ids = (char_ids + [26]*(10 - len(char_ids)))[:10]
    return char_ids

