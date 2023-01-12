import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import clear_output

def plot_history(epochs, acc):
    # print(history.history.keys())
    
    clear_output(wait = True)
    # 精度の履歴をプロット
    plt.plot(epochs, acc)
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim([-0.02,1.02])
    # plt.savefig(DIR_PATH + 'model_accuracy.png')
    # plt.savefig('figure_acc/figure_' + datetime.now().strftime('%Y%m%d') + '.png')
    # plt.show()

epochs = []
acc = []
for _ in range(50):
	epochs.append(_)
	acc.append(_)

plot_history(epochs, acc)

plt.savefig('figure_test.png')
