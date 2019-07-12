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
    idx0 = np.arange(0, len(rp_len), dtype=np.int32)
    print(stats.describe(rp_len))
    ep_len = np.asarray(premises[1])
    idx1 = np.arange(0, len(ep_len), dtype=np.int32)
    print(stats.describe(ep_len))
    plt.figure(figsize=(16, 9))
    paxes = plt.subplot(131)
    paxes.scatter(idx0[:], rp_len[:], c='green')
    paxes.scatter(idx1[:], ep_len[:], c='red')
    plt.grid(True)
    plt.xlabel('index')
    plt.ylabel('length')
    plt.title('Premise')

    print('Seq len info: talts')
    rt_len = np.asarray(talts[0])
    idx0 = np.arange(0, len(rt_len), dtype=np.int32)
    print(stats.describe(rt_len))
    et_len = np.asarray(talts[1])
    idx1 = np.arange(0, len(et_len), dtype=np.int32)
    print(stats.describe(et_len))
    taxes = plt.subplot(132)
    taxes.scatter(idx0[:], rt_len[:], c='green')
    taxes.scatter(idx1[:], et_len[:], c='red')
    plt.grid(True)
    plt.xlabel('index')
    plt.ylabel('length')
    plt.title('Talt')

    print('Seq len info: falts')
    rf_len = np.asarray(falts[0])
    idx0 = np.arange(0, len(rf_len), dtype=np.int32)
    print(stats.describe(rf_len))
    ef_len = np.asarray(falts[1])
    idx1 = np.arange(0, len(ef_len), dtype=np.int32)
    print(stats.describe(ef_len))
    faxes = plt.subplot(133)
    faxes.scatter(idx0[:], rf_len[:], c='green')
    faxes.scatter(idx1[:], ef_len[:], c='red')
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


def show_score(err_scores, rig_scores):
    plt.figure(figsize=(32, 9))
    eaxes = plt.subplot(121)
    idx = np.arange(0, err_scores.shape[0], dtype=np.int32)
    eaxes.scatter(idx[:], err_scores[:, 0], c='red')
    eaxes.scatter(idx[:], err_scores[:, 1], c='green')
    plt.grid(True)
    plt.xlabel('index')
    plt.ylabel('score')
    plt.title('Error')

    raxes = plt.subplot(122)
    idx = np.arange(0, rig_scores.shape[0], dtype=np.int32)
    raxes.scatter(idx[:], rig_scores[:, 0], c='green')
    raxes.scatter(idx[:], rig_scores[:, 1], c='red')
    plt.grid(True)
    plt.xlabel('index')
    plt.ylabel('score')
    plt.title('Right')
    plt.savefig("./score.jpg", format='jpg')


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


def save_csv(premises, modes, alts1, alts2, answers, scores, file_type):
    talt, falt = [], []
    tscore, fscore = [], []
    for a1, a2, score, ans in zip(alts1, alts2, scores, answers):
        if ans == 0:
            talt.append(a1)
            falt.append(a2)
            tscore.append(score[0])
            fscore.append(score[1])
        else:
            talt.append(a2)
            falt.append(a1)
            tscore.append(score[1])
            fscore.append(score[0])

    df = pd.DataFrame(
        {'premises': premises, 'modes': modes, 'talt': talt, 'falt': falt, 'tscore': tscore, 'fscore': fscore})
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

    with open('./dataset/raw/scores_3', 'rb') as f:
        logits = pkl.load(f)
    f.close()
    logits = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    outputs = np.argmax(logits, axis=1)
    labels = np.asarray(te_answers)
    index = np.arange(0, len(labels))
    error_idx = index[outputs != labels]
    right_idx = index[outputs == labels]
    error_scores = logits[error_idx]
    # minus_error = abs(error_scores[:, 0] - error_scores[:, 1])
    # print('Error scores:')
    # print(stats.describe(minus_error))
    # new_error = [[score[0], score[1]] if score[0] > score[1] else [score[1], score[0]] for score in error_scores]
    # new_error = np.asarray(new_error)
    right_scores = logits[right_idx]
    # minus_right = abs(right_scores[:, 0] - right_scores[:, 1])
    # print('Right scores:')
    # print(stats.describe(minus_right))
    # new_right = [[score[0], score[1]] if score[0] > score[1] else [score[1], score[0]] for score in right_scores]
    # new_right = np.asarray(new_right)
    # show_score(new_error, new_right)
    save_csv(te_premises[error_idx], te_modes[error_idx], te_alts1[error_idx], te_alts2[error_idx],
             te_answers[error_idx], error_scores, 'errors')
    save_csv(te_premises[right_idx], te_modes[right_idx], te_alts1[right_idx], te_alts2[right_idx],
             te_answers[right_idx], right_scores, 'rights')

    # err_pre, err_a1, err_a2, err_pre_len, err_a1_len, err_a2_len = pipeline(
    #     te_premises[error_idx].tolist(),
    #     te_alts1[error_idx].tolist(),
    #     te_alts2[error_idx].tolist(),
    #     te_answers[error_idx].tolist())
    # rig_pre, rig_a1, rig_a2, rig_pre_len, rig_a1_len, rig_a2_len = pipeline(
    #     te_premises[right_idx].tolist(),
    #     te_alts1[right_idx].tolist(),
    #     te_alts2[right_idx].tolist(),
    #     te_answers[right_idx].tolist())
    #
    # show_len([rig_pre_len, err_pre_len], [rig_a1_len, err_a1_len], [rig_a2_len, err_a2_len])

    print('hello world')
