import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


def mean_reciprocal_rank(y_true, y_pred_scores):
    rr_total = 0.0
    n = len(y_true)

    for true_labels, scores in zip(y_true, y_pred_scores):
        ranked = np.argsort(scores)[::-1]
        # find rank of first correct answer
        rr = 0
        for rank, idx in enumerate(ranked, start=1):
            if idx in true_labels:
                rr = 1.0 / rank
                break
        rr_total += rr

    return rr_total / n


def meteor_scores(preds, targets):
    scores = []
    for pred, ref in zip(preds, targets):
        score = meteor_score([ref.split()], pred.split())
        scores.append(score)
    return sum(scores) / len(scores)


def bleu_score(preds, targets):
    smoothie = SmoothingFunction().method4
    scores = []
    for pred, ref in zip(preds, targets):
        ref_tokens = [ref.split()]
        pred_tokens = pred.split()
        score = sentence_bleu(ref_tokens, pred_tokens,
                              weights=(0.25, 0.25, 0.25, 0.25),  # BLEU-4
                              smoothing_function=smoothie)
        scores.append(score)
    return sum(scores) / len(scores)


def evaluation(sp, act, acxtText=None, predText=None):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(p)):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))
    A = np.squeeze(act)
    B = np.squeeze(sp)

    accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100
    Recall = (tp / (tp + fn)) * 100
    precision = (tp / (tp + fp)) * 100
    F1_score = ((2 * tp) / (2 * tp + fp + fn)) * 100
    matches = np.all(B == A, axis=1)
    EM = np.mean(matches) * 100
    MRR = mean_reciprocal_rank(A, B)
    BLEU = bleu_score(B, A)
    Meteor = meteor_scores(B, A)
    bertScore = np.asarray(precision.mean(), Recall.mean(), F1_score.mean())
    EVAL_ERR = [accuracy, Recall, precision, F1_score, EM, BLEU, Meteor, bertScore, MRR]
    return EVAL_ERR
