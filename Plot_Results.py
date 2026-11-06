import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from matplotlib import pylab

No_of_Dataset = 2


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'FFO-MMCA-ADBGL', 'BCO-MMCA-ADBGL', 'APO-MMCA-ADBGL', 'BOA-MMCA-ADBGL', 'SH-BOA-MMCA-ADBGL']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(No_of_Dataset):
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Convergence Curve')
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'EM', 'BLEU', 'METEOR', 'BERT Score', 'Perplexity', 'MRR']
    Algorithm = ['FFO-MMCA-ADBGL', 'BCO-MMCA-ADBGL', 'APO-MMCA-ADBGL', 'BOA-MMCA-ADBGL', 'SH-BOA-MMCA-ADBGL']
    Classification = ['RNN', 'LSTM', 'GRU', 'DistilBERT-LSTM', 'SH-BOA-MMCA-ADBGL']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Step_per_Epochs = [100, 200, 300, 400, 500]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[n, k, l, Graph_Term[j]]

            fig = plt.figure()
            ax = fig.add_axes([0.19, 0.18, 0.7, 0.7])
            X = np.arange(len(Step_per_Epochs))
            ax.barh(X + 0.00, Graph[:, 0], height=0.15, color='darkgreen', label=Algorithm[0])
            ax.barh(X + 0.15, Graph[:, 1], height=0.15, color='darkorange', label=Algorithm[1])
            ax.barh(X + 0.30, Graph[:, 2], height=0.15, color='#A52A2A', label=Algorithm[2])
            ax.barh(X + 0.45, Graph[:, 3], height=0.15, color='#80B918', label=Algorithm[3])
            ax.barh(X + 0.60, Graph[:, 4], height=0.15, color='k', label=Algorithm[4])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            plt.yticks(X + 0.15, ('100', '200', '300', '400', '500'), fontsize=10, fontweight='bold', color='#7209b7')
            plt.xlabel(Terms[Graph_Term[j]], fontsize=12, fontweight='bold', color='k')
            plt.ylabel('Step Per Epochs', fontsize=12, fontweight='bold', color='k')
            plt.xticks(fontsize=10, fontweight='bold', color='#7209b7')
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=12) for color
                           in ['darkgreen', 'darkorange', '#A52A2A', '#80B918', 'k']]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.44, 1.18), fontsize=11,
                       frameon=False, ncol=3)
            path = "./Results/Dataset_%s_%s_Text_Mod_bar.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Dataset - ' + str(n+1) + 'Step Per Epochs  vs ' + Terms[Graph_Term[j]])
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.19, 0.18, 0.7, 0.7])
            X = np.arange(len(Step_per_Epochs))
            ax.barh(X + 0.00, Graph[:, 5], height=0.15, color='#6a994e', label=Classification[0])
            ax.barh(X + 0.15, Graph[:, 6], height=0.15, color='#00a8e8', label=Classification[1])
            ax.barh(X + 0.30, Graph[:, 7], height=0.15, color='violet', label=Classification[2])
            ax.barh(X + 0.45, Graph[:, 8], height=0.15, color='crimson', label=Classification[3])
            ax.barh(X + 0.60, Graph[:, 4], height=0.15, color='k', label=Classification[4])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            plt.yticks(X + 0.15, ('100', '200', '300', '400', '500'), fontsize=10, fontweight='bold', color='#7209b7')
            plt.xlabel(Terms[Graph_Term[j]], fontsize=12, fontweight='bold', color='k')
            plt.ylabel('Step Per Epochs', fontsize=12, fontweight='bold', color='k')
            plt.xticks(fontsize=10, fontweight='bold', color='#7209b7')
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=12) for color
                           in ['#6a994e', '#00a8e8', 'violet', 'crimson', 'k']]
            plt.legend(dot_markers, Classification, loc='upper center', bbox_to_anchor=(0.5, 1.18), fontsize=12,
                       frameon=False, ncol=3)
            path = "./Results/Dataset_%s_%s_Text_Mod_bar.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Dataset - ' + str(n+1) + 'Step Per Epochs  vs ' + Terms[Graph_Term[j]])
            plt.show()


def Table():
    eval = np.load('Evaluate.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'EM', 'BLEU', 'METEOR', 'BERT Score', 'Perplexity', 'MRR']
    Algorithm = ['Activation Function', 'FFO-MMCA-ADBGL', 'BCO-MMCA-ADBGL', 'APO-MMCA-ADBGL', 'BOA-MMCA-ADBGL', 'SH-BOA-MMCA-ADBGL']
    Classifier = ['Activation Function', 'RNN', 'LSTM', 'GRU', 'DistilBERT-LSTM', 'SH-BOA-MMCA-ADBGL']
    Graph_Term = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(int)
    Table_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    table_terms = [Terms[i] for i in Table_Terms]
    Activation_Function = ['Linear', 'ReLU', 'TanH', 'Softmax', 'Sigmoid']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, :]
            Table = PrettyTable()
            Table.add_column(Algorithm[0], Activation_Function)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Term[k]])
            print('------------------------------- Dataset- ', i + 1, '-', table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Activation_Function)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Term[k]])
            print('------------------------------- Dataset- ', i + 1, '-', table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Proposed_Plots_Results():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'EM', 'BLEU', 'METEOR', 'BERT Score', 'Perplexity', 'MRR']
    Algorithm = ['FFO-MMCA-ADBGL', 'BCO-MMCA-ADBGL', 'APO-MMCA-ADBGL', 'BOA-MMCA-ADBGL', 'SH-BOA-MMCA-ADBGL']
    Classfier = ['RNN', 'LSTM', 'GRU', 'DistilBERT-LSTM', 'SH-BOA-MMCA-ADBGL']
    Graph_Term = [0, 2, 3, 5, 8]
    epoch = [20, 40, 60, 80, 100]
    for i in range(No_of_Dataset):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1] - 4, eval.shape[2]))
            for l in range(eval.shape[2]):
                Graph[:, l] = eval[i, 4, l, Graph_Term[j]]

            fig = plt.figure(figsize=(8, 7), facecolor='#dcf8e3')
            ax = fig.add_axes([0.06, 0.12, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + 'Algorithm Comparison of Epochs')
            ax.set_facecolor("#dcf8e3")
            X = np.arange(Graph.shape[0])
            barWidth = 0.05
            coloru = ['#ff006e', '#fca311', '#99582a', '#b56576', 'k']
            for l in range(0, Graph.shape[1] - 5):
                bars = ax.bar(X + (l * barWidth * 2.5), Graph[:, l], color=coloru[l], width=barWidth,
                              edgecolor='#032b43',
                              label=Algorithm[l - 5])
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2, height + (height * 0.02), f"{(np.round(height, 1))}",
                             ha='center', va='bottom', fontsize=10, weight='bold')

            ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)  # X-axis arrow
            ax.plot(-0.09, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)  # Y-axis arrow
            ax.spines['bottom'].set_position('zero')
            ax.spines['left'].set_position(('data', -0.09))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([0, 0.125, 0.25, 0.375, 0.499])
            ax.set_xticklabels(['FFO-MMCA- \n ADBGL', 'BCO-MMCA- \n ADBGL', 'APO-MMCA- \n ADBGL', 'BOA-MMCA- \n ADBGL', 'SH-BOA- \n MMCA-ADBGL'])
            plt.xlabel('Algorithm', fontsize=16, labelpad=10, fontname="Arial", fontweight='bold', color='#5f0f40')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=16, fontweight='bold', color='#5f0f40')
            plt.xticks(fontname="Arial", fontsize=11, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
            path = "./Results/Dataset_%s_Proposed_%s_Alg_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure(figsize=(8, 7), facecolor='#dcf8e3')
            ax = fig.add_axes([0.06, 0.12, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + 'Method Comparison of Epochs')
            ax.set_facecolor("#dcf8e3")
            X = np.arange(Graph.shape[0])
            barWidth = 0.05
            coloru = ['#e6ebe0', '#5ca4a9', '#9bc1bc', '#f4f1bb', '#ed6a5a']
            for l in range(0, Graph.shape[1] - 5):
                val = Graph[:, 5:]
                bars = ax.bar(X + (l * barWidth * 2.5), val[:, l], color=coloru[l], width=barWidth,
                              edgecolor='#032b43',
                              label=Classfier[l - 5])
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2, height + (height * 0.02), f"{(np.round(height, 1))}",
                             ha='center', va='bottom', fontsize=10, weight='bold')

            ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)  # X-axis arrow
            ax.plot(-0.09, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)  # Y-axis arrow
            ax.spines['bottom'].set_position('zero')
            ax.spines['left'].set_position(('data', -0.09))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([0, 0.125, 0.25, 0.375, 0.499])
            ax.set_xticklabels(['RNN', 'LSTM', 'GRU', 'DistilBERT \n -LSTM', 'SH-BOA- \n MMCA-ADBGL'])
            plt.xlabel('Classifiers', fontsize=16, labelpad=10, fontname="Arial", fontweight='bold', color='#5f0f40')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=16, fontweight='bold', color='#5f0f40')
            plt.xticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
            path = "./Results/Dataset_%s_Proposed_%s_Mod_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()


def Plot_Text_Gen_Results():
    Eval = np.load('Eval_Summ_2.npy', allow_pickle=True)
    Rouge = ['Rouge-1', 'Rouge-2', 'Rouge-l']
    Algorithm = ['FFO-MMCA-ADBGL', 'BCO-MMCA-ADBGL', 'APO-MMCA-ADBGL', 'BOA-MMCA-ADBGL', 'SH-BOA-MMCA-ADBGL']
    Methods = ['RNN', 'LSTM', 'GRU', 'DistilBERT-LSTM', 'SH-BOA-MMCA-ADBGL']
    for n in range(Eval.shape[0]):
        for k in range(len(Rouge)):
            fig = plt.figure()
            ax = fig.add_axes([0.19, 0.18, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset- ' + str(n + 1) + ' Algorithm  Comparison of Rouge')
            X = np.arange(len(Rouge))
            ax.barh(X + 0.00, Eval[n, 0, k, :], height=0.15, color='darkgreen', label=Algorithm[0])
            ax.barh(X + 0.15, Eval[n, 1, k, :], height=0.15, color='darkorange', label=Algorithm[1])
            ax.barh(X + 0.30, Eval[n, 2, k, :], height=0.15, color='#A52A2A', label=Algorithm[2])
            ax.barh(X + 0.45, Eval[n, 3, k, :], height=0.15, color='#80B918', label=Algorithm[3])
            ax.barh(X + 0.60, Eval[n, 4, k, :], height=0.15, color='k', label=Algorithm[4])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=12) for color
                           in ['darkgreen', 'darkorange', '#A52A2A', '#80B918', 'k']]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.44, 1.18), fontsize=11,
                       frameon=False, ncol=3)
            plt.yticks(X + 0.30, ('F1 Score', 'Precision', 'Recall'), fontname="Arial", fontsize=12, fontweight='bold',
                       color='#7209b7')
            plt.xlabel(Rouge[k], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xticks(fontsize=10, fontweight='bold', color='#7209b7')
            path1 = "./Results/Dataset_%s_%s_Alg.png" % (str(n + 1), Rouge[k])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.19, 0.18, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset- ' + str(n + 1) + ' Method  Comparison of Rouge')
            X = np.arange(len(Rouge))
            ax.barh(X + 0.00, Eval[n, 5, k, :], height=0.15, color='#6a994e', label=Methods[0])
            ax.barh(X + 0.15, Eval[n, 6, k, :], height=0.15, color='#00a8e8', label=Methods[1])
            ax.barh(X + 0.30, Eval[n, 7, k, :], height=0.15, color='violet', label=Methods[2])
            ax.barh(X + 0.45, Eval[n, 8, k, :], height=0.15, color='crimson', label=Methods[3])
            ax.barh(X + 0.60, Eval[n, 4, k, :], height=0.15, color='k', label=Methods[4])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=12) for color
                          in ['#6a994e', '#00a8e8', 'violet', 'crimson', 'k']]
            plt.legend(dot_markers, Methods, loc='upper center', bbox_to_anchor=(0.5, 1.18), fontsize=12,
                       frameon=False, ncol=3)
            plt.yticks(X + 0.30, ('F1 Score', 'Precision', 'Recall'), fontname="Arial", fontsize=12, fontweight='bold',
                       color='#7209b7')
            plt.xlabel(Rouge[k], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xticks(fontsize=10, fontweight='bold', color='#7209b7')
            path1 = "./Results/Dataset_%s_%s_Mod.png" % (str(n + 1), Rouge[k])
            plt.savefig(path1)
            plt.show()


if __name__ == '__main__':
    plotConvResults()
    Plot_Results()
    Table()
    Proposed_Plots_Results()
    Plot_Text_Gen_Results()
