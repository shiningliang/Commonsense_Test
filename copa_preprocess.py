import pandas as pd
import numpy as np
import pickle as pkl
import xml.etree.cElementTree as et
from scipy import stats
import matplotlib.pyplot as plt
import spacy

plt.switch_backend('agg')
encoder = spacy.load('en_core_web_md')


def show_len(premises, talts, falts):
    print('Seq len info: premise')
    rp_len = np.asarray(premises[0])
    idx = np.arange(0, len(rp_len), dtype=np.int32)
    print(stats.describe(rp_len))
    ep_len = np.asarray(premises[1])
    print(stats.describe(ep_len))
    plt.figure(figsize=(16, 9))
    paxes = plt.subplot(131)
    paxes.scatter(idx[:], rp_len[:], c='green')
    paxes.scatter(idx[:], ep_len[:], c='red')
    plt.grid(True)
    plt.xlabel('index')
    plt.ylabel('length')
    plt.title('Premise')

    print('Seq len info: talts')
    rt_len = np.asarray(talts[0])
    idx = np.arange(0, len(rt_len), dtype=np.int32)
    print(stats.describe(rt_len))
    et_len = np.asarray(talts[1])
    print(stats.describe(et_len))
    taxes = plt.subplot(132)
    taxes.scatter(idx[:], rt_len[:], c='green')
    taxes.scatter(idx[:], et_len[:], c='red')
    plt.grid(True)
    plt.xlabel('index')
    plt.ylabel('length')
    plt.title('Talt')

    print('Seq len info: falts')
    rf_len = np.asarray(falts[0])
    idx = np.arange(0, len(rf_len), dtype=np.int32)
    print(stats.describe(rf_len))
    ef_len = np.asarray(falts[1])
    print(stats.describe(ef_len))
    faxes = plt.subplot(132)
    faxes.scatter(idx[:], rf_len[:], c='green')
    faxes.scatter(idx[:], ef_len[:], c='red')
    plt.grid(True)
    plt.xlabel('index')
    plt.ylabel('length')
    plt.title('Falt')


    # plt.subplot(122)
    # plt.hist(seq_len, bins=5, label=['seq_len'])
    # plt.grid(True)
    # plt.xlabel(seq_type)
    # plt.ylabel('freq')
    # plt.title('Histogram')
    # plt.show()
    plt.savefig("./length.jpg", format='jpg')


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


def trans_copa(filepath):
    xml_tree = et.parse(filepath)
    corpus = xml_tree.getroot()
    premises = []
    alts1, alts2 = [], []
    answers = []
    modes = []
    for item in corpus:
        mode = item.attrib["asks-for"]
        modes.append(mode)
        answer = int(item.attrib["most-plausible-alternative"]) - 1  # answers are 1 and 2, convert to 0 and 1
        answers.append(answer)
        premise = item.find("p").text
        premises.append(premise)
        alt1 = item.find("a1").text
        alt2 = item.find("a2").text
        alts1.append(alt1)
        alts2.append(alt2)

    return premises, alts1, alts2, answers, modes


def save_csv(premises, modes, alts1, alts2, answers, file_type):
    talt, falt = [], []
    for a1, a2, ans in zip(alts1, alts2, answers):
        if ans == 0:
            talt.append(a1)
            falt.append(a2)
        else:
            talt.append(a2)
            falt.append(a1)

    df = pd.DataFrame({'premises': premises, 'modes': modes, 'talt': talt, 'falt': falt, 'answers': answers})
    df.to_csv(file_type + '.csv', index=False)


def tokenize(seqs, lemmatize=True):
    new_seqs = []
    new_len = []
    for seq in seqs:
        seq = encoder(seq)
        if lemmatize:
            seq = [word.lemma_ for word in seq]
        else:
            seq = [word.string.strip() for word in seq]
        seq = [word for word in seq if word]
        new_seqs.append(seq)
        new_len.append(len(seq))

    return new_seqs, new_len


def pipeline(premises, alts1, alts2, answers):
    talt, falt = [], []
    for a1, a2, ans in zip(alts1, alts2, answers):
        if ans == 0:
            talt.append(a1)
            falt.append(a2)
        else:
            talt.append(a2)
            falt.append(a1)
    premises, premises_len = tokenize(premises)
    talt, talt_len = tokenize(talt)
    falt, falt_len = tokenize(falt)

    return premises, talt, falt, premises_len, talt_len, falt_len


if __name__ == '__main__':
    # train_path = './dataset/raw/copa-dev.xml'
    # tr_premises, tr_alts, tr_answers = read_copa(train_path)
    # save_csv(tr_premises, tr_alts, tr_answers, 'copa-train')
    test_path = './dataset/raw/copa-test.xml'
    # te_premises, te_alts, te_answers = read_copa(test_path)
    te_premises, te_alts1, te_alts2, te_answers, te_modes = trans_copa(test_path)
    te_premises = np.asarray(te_premises)
    te_alts1 = np.asarray(te_alts1)
    te_alts2 = np.asarray(te_alts2)
    te_answers = np.asarray(te_answers)
    te_modes = np.asarray(te_modes)
    # save_csv(te_premises, te_alts1, te_alts2, te_answers, 'copa-test')

    with open('./dataset/raw/scores_1', 'rb') as f:
        logits = pkl.load(f)
    f.close()

    outputs = np.argmax(logits, axis=1)
    labels = np.asarray(te_answers)
    index = np.arange(0, len(labels))
    error_idx = index[outputs != labels]
    right_idx = index[outputs == labels]
    # save_csv(te_premises[error_idx], te_modes[error_idx], te_alts1[error_idx], te_alts2[error_idx], te_answers[error_idx], 'errors')
    # save_csv(te_premises[right_idx], te_modes[right_idx], te_alts1[right_idx], te_alts2[right_idx], te_answers[right_idx], 'rights')

    err_pre, err_a1, err_a2, err_pre_len, err_a1_len, err_a2_len = pipeline(
        te_premises[error_idx],
        te_alts1[error_idx],
        te_alts2[error_idx],
        te_answers[error_idx])
    rig_pre, rig_a1, rig_a2, rig_pre_len, rig_a1_len, rig_a2_len = pipeline(
        te_premises[right_idx],
        te_alts1[right_idx],
        te_alts2[right_idx],
        te_answers[right_idx])

    print('hello world')
