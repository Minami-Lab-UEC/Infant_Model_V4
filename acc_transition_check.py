import pandas as pd
import matplotlib.pyplot as plt

acc = pd.read_pickle('result_9/acc_transition.pkl')
acc['noun_acc'] = 0
acc['verb_acc'] = 0

count_noun_ans = 0
count_verb_ans = 0
count_noun_correct = 0
count_verb_correct = 0

for index, row in acc.iterrows():
	if row['nounorverb'] == 0: # nounの場合
		count_noun_ans += 1
		if row['ans'] == 1:
			count_noun_correct += 1
			acc.at[index, 'noun_acc'] = count_noun_correct/count_noun_ans
			acc.at[index, 'verb_acc'] = acc.loc[index-1, 'verb_acc']
		else:
			if index > 0: # 初回ループでなければ
				acc.at[index, 'noun_acc'] = acc.loc[index-1, 'noun_acc'] # 前回のaccuracyを入れる
				acc.at[index, 'verb_acc'] = acc.loc[index-1, 'verb_acc'] # 前回のaccuracyを入れる
			else:
				acc.at[index, 'noun_acc'] = 0
				acc.at[index, 'verb_acc'] = 0
	else: # verbの場合
		count_verb_ans += 1
		if row['ans'] == 1:
			count_verb_correct += 1
			acc.at[index, 'verb_acc'] = count_verb_correct/count_verb_ans
			acc.at[index, 'noun_acc'] = acc.loc[index-1, 'noun_acc']
		else:
			if index > 0: # 初回ループでなければ
				acc.at[index, 'noun_acc'] = acc.loc[index-1, 'noun_acc'] # 前回のaccuracyを入れる
				acc.at[index, 'verb_acc'] = acc.loc[index-1, 'verb_acc'] # 前回のaccuracyを入れる
			else:
				acc.at[index, 'noun_acc'] = 0
				acc.at[index, 'verb_acc'] = 0

# ax = acc.plot(y='noun_acc')
# acc.plot(y='verb_acc', ax=ax)
acc[['noun_acc', 'verb_acc']].plot()
# acc['noun_acc'].plot()
plt.savefig('result_9/acc_transition.png')
plt.close('all')
	
