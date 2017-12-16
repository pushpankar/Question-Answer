import tensorflow as tf

def main(files):
    for f in files:
        story_len = get_story_len(f)
        max_batch = get_max_batch(f, batch_size, story_len)
        for j in range(max_batch):
            offset = batch_size * j
            # initialize s_h
            for i in range(story_len):
                lines = get_batch(i, offset, batch_size)
                sentences = get_sentences(lines)
                batch_data = batch2nums(sentences)
                if is_ques(lines):
                    q = lstm(batch_data)
                    predict(q, s_h)
                else:
                    s_h = lstm(batch_data, s_h)



def get_max_batch(filename, batch_size, story_len):
    with open(filename, 'r') as f:
        lines = f.readlines()
        n_lines = len(lines)
    return n_lines//(batch_size*story_len)


def get_story_len(filename):
    with open(filename, 'r') as f:
        story_num = 0
        for line in f:
            contents = line.split('\t')
            if story_num > contents[0]:
                return story_num
            else:
                story_num = contents[0]

