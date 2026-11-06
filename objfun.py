import numpy as np
from Evaluation import evaluation
from Global_Vars import Global_Vars
from Model_BERT_LSTM import Model_BERT_LSTM


def objfun(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, pred = Model_BERT_LSTM(data, Tar, sol)
            Eval = evaluation(pred, Tar)
            Fitn[i] = 1 / (Eval[4] + Eval[3])
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, pred = Model_BERT_LSTM(data, Tar, sol)
        Eval = evaluation(pred, Tar)
        Fitn = 1 / (Eval[4] + Eval[3])
        return Fitn
