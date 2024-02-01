import collections
import numpy as np
import pickle
from tqdm import tqdm
import os


def calculate_docs_bias():
    data_folder = '/home/sajadeb/msmarco'
    wordlist_path = "/home/shiva_soleimany/RL/deep-q-rerank/util/calculate_bias/wordlist_genderspecific.txt"

    docs_bias_save_paths = {'tc': "docs_bias_tc.pkl",
                            'bool': "docs_bias_bool.pkl",
                            'tf': "docs_bias_tf.pkl"}

    female_words = []
    male_words = []

    with open(wordlist_path, 'r') as f:
        for line in f:
            term, gender = line.strip().split(',')
            if gender == 'f':
                female_words.append(term)
            elif gender == 'm':
                male_words.append(term)

    female_words = set(female_words)
    male_words = set(male_words)

    print(f"Number of the Male-related words: {len(male_words)}")
    print(f"Number of the Female-related words: {len(female_words)}")

    def get_tokens(input_text):
        return input_text.lower().split(" ")

    def get_bias(tokens):
        text_cnt = collections.Counter(tokens)

        cnt_female = 0
        cnt_male = 0
        cnt_log_female = 0
        cnt_log_male = 0
        for word in text_cnt:
            if word in female_words:
                cnt_female += text_cnt[word]
                cnt_log_female += np.log(text_cnt[word] + 1)
            elif word in male_words:
                cnt_male += text_cnt[word]
                cnt_log_male += np.log(text_cnt[word] + 1)

        bias_tc = (float(cnt_female - cnt_male), float(cnt_female), float(cnt_male))
        bias_tf = (np.log(cnt_female + 1) - np.log(cnt_male + 1), np.log(cnt_female + 1), np.log(cnt_male + 1))
        bias_bool = (np.sign(cnt_female) - np.sign(cnt_male), np.sign(cnt_female), np.sign(cnt_male))

        return bias_tc, bias_tf, bias_bool

    print('Loading collection...')
    corpus = {}
    corpus_filepath = os.path.join(data_folder, 'collection.tsv')
    with open(corpus_filepath, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            pid, passage = line.strip().split("\t")
            corpus[pid] = passage.strip()

    docs_bias = {'tc': {}, 'tf': {}, 'bool': {}}
    for pid, text in tqdm(corpus.items()):
        biases = get_bias(get_tokens(text))
        docs_bias['tc'][pid] = biases[0]
        docs_bias['tf'][pid] = biases[1]
        docs_bias['bool'][pid] = biases[2]

    with open('documents_bias_tf.pkl', 'wb') as handle:
        pickle.dump(docs_bias['tf'], handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    calculate_docs_bias()
