from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from numpy import matlib
from APO import APO
from BCO import BCO
from BOA import BOA
from FFO import FFO
from Global_Vars import Global_Vars
from Model_BERT_LSTM import Model_BERT_LSTM
from Model_GRU import Model_GRU
from Model_LSTM import Model_LSTM
from Model_RNN import Model_RNN
from Plot_Results import *
from Proposed import Proposed
from objfun import objfun

No_of_Dataset = 2


# Removing Punctuations
def rem_punct(my_str):
    # define punctuation
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # remove punctuation from the string
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char + " "

    # display the unpunctuated string
    return no_punct


# Read Dataset 1
an = 0
if an == 1:
    ds = load_dataset("HiTZ/MedExpQA", "en")
    print(ds)
    train_data = ds['train']
    qn = train_data['full_question']
    ans = train_data['full_answer']
    np.save('Text_1.npy', np.reshape(np.asarray(qn), (-1, 1)))
    np.save('Answer_1.npy', np.reshape(np.asarray(ans), (-1, 1)))

# Read Dataset 2
an = 0
if an == 1:
    path = './archive/MedQA-USMLE/questions/Mainland/4_options/train.jsonl'
    import json

    Data = []
    Answer = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            qn = data['question']
            ans = data['answer']
            Data.append(qn)
            Answer.append(ans)
    np.save('Text_2.npy', np.reshape(np.asarray(Data), (-1, 1)))
    np.save('Answer_2.npy', np.reshape(np.asarray(Answer), (-1, 1)))

# Data Preprocessing
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        data = np.load('Text_' + str(n + 1) + '.npy', allow_pickle=True)
        ps = PorterStemmer()
        Preprocess = []
        for i in range(data.shape[0]):
            Pre = []
            for j in range(data.shape[1]):
                print(i, j)
                stem = []
                text_tokens = word_tokenize(data[i, j])
                for w in text_tokens:  # Stemming with Lower case conversion
                    stem_tokens = ps.stem(w)
                    stem.append(stem_tokens)
                words = [word for word in stem if
                         not word in stopwords.words()]  # tokens without stop words
                prep = rem_punct(words)  # Punctuation Removal
                Pre.append(prep)
            Preprocess.append(Pre)
        np.save('Pre_Data_' + str(n + 1) + '.npy', np.asarray(Preprocess))

# Optimization for Answer Generation
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Data = np.load('Pre_Data_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Answer_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Data = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden neuron count (BERT and LSTM), Learning Rate
        xmin = matlib.repmat([5, 0.01, 5], Npop, 1)
        xmax = matlib.repmat([255, 0.99, 255], Npop, 1)
        fname = objfun
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("FFO...")
        [bestfit1, fitness1, bestsol1, time1] = FFO(initsol, fname, xmin, xmax, Max_iter)

        print("BCO...")
        [bestfit2, fitness2, bestsol2, time2] = BCO(initsol, fname, xmin, xmax, Max_iter)

        print("APO...")
        [bestfit3, fitness3, bestsol3, time3] = APO(initsol, fname, xmin, xmax, Max_iter)

        print("BOA...")
        [bestfit4, fitness4, bestsol4, time4] = BOA(initsol, fname, xmin, xmax, Max_iter)

        print("Proposed...")
        [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_' + str(n + 1) + '.npy', BestSol)

# Response Generation
an = 0
if an == 1:
    Eval_all = []
    for n in range(No_of_Dataset):
        Data = np.load('Pre_Data_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Answer_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('BestSol_' + str(n + 1) + '.npy', allow_pickle=True)
        EVAL = []
        Epochs = [20, 40, 60, 80, 100]
        for act in range(len(Epochs)):
            learnperc = round(Data.shape[0] * 0.75)  # Split Training and Testing Datas
            Train_Data = Data[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Data[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 10))
            for j in range(BestSol.shape[0]):
                print(act, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred = Model_BERT_LSTM(Data, Target, Epochs[act], sol)  # BERT_LSTM  With optimization
            Eval[5, :], pred1 = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target, Epochs[act])
            Eval[6, :], pred2 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, Epochs[act])
            Eval[7, :], pred3 = Model_GRU(Train_Data, Train_Target, Test_Data, Test_Target, Epochs[act])
            Eval[8, :], pred4 = Model_BERT_LSTM(Data, Target, Epochs[act])  # BERT_LSTM  Without optimization
            Eval[8, :] = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
    np.save('Eval_all.npy', Eval_all)

plotConvResults()
Plot_Results()
Plot_Text_Gen_Results()
Table()
Proposed_Plots_Results()
