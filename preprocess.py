import pandas as pd
import numpy as np
import pickle as pkl
import xml.etree.cElementTree as et


def read_copa(filepath):
    xml_tree = et.parse(filepath)
    corpus = xml_tree.getroot()
    premises = []
    alts = []
    answers = []
    modes = []
    for item in corpus:
        mode = item.attrib["asks-for"]
        modes.append(mode)
        answer = int(item.attrib["most-plausible-alternative"]) - 1  # answers are 1 and 2, convert to 0 and 1
        if answer == 0:
            answers.extend([1, 0])
        else:
            answers.extend([0, 1])
        premise = item.find("p").text
        premises.extend([premise] * 2)
        alts.extend([item.find("a1").text, item.find("a2").text])
    return premises, alts, answers


def save_csv(premises, alts, answers, file_type):
    df = pd.DataFrame({'premises': premises, 'alts': alts, 'answers': answers})
    df.to_csv(file_type + '.csv', sep='\t', index=False)


if __name__ == '__main__':
    train_path = './dataset/raw/copa-dev.xml'
    tr_premises, tr_alts, tr_answers = read_copa(train_path)
    save_csv(tr_premises, tr_alts, tr_answers, 'copa-train')
    test_path = './dataset/raw/copa-test.xml'
    te_premises, te_alts, te_answers = read_copa(test_path)
    save_csv(te_premises, te_alts, te_answers, 'copa-test')
