# import spacy
#
# encoder = spacy.load('en_core_web_md')
# txt = "Dan's parents were overweight. Dan was overweight as well. The BBC told his parents it was unhealthy. His parents understood and decided to make a change. They got themselves and Dan on a diet."
#
# txt = encoder(txt)
# for sent in txt.sents:
#     print(sent)
#
# for word in txt:
#     print(word.string, word.tag_, word.pos_)


import pandas as pd


def read_file(file_path):
    raw_df = pd.read_csv(file_path, sep=',')

    def func(sen_list):
        return ' '.join(sen_list)

    raw_df['storytitle'] = raw_df.apply(lambda row: func([row.sentence1, row.sentence2, row.sentence3, row.sentence4,
                                                          row.sentence5]), axis=1)
    merge = raw_df['storytitle'].values.tolist()

    return merge


data_path = 'dataset/raw/ROC-2016.csv'
data_a = read_file(data_path)
data_path = 'dataset/raw/ROC-2017.csv'
data_b = read_file(data_path)

data_a.extend(data_b)
df = pd.DataFrame({'sentence': data_a})
df.to_csv('dataset/raw/stories.csv', index=False)
