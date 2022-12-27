import pandas as pd

main_df_1 = pd.read_csv('densetrack_create_takeshita/main_new_1219.csv')
main_df_2 = pd.read_csv('densetrack_create_takeshita/main_new.csv')
move_df = pd.read_csv('densetrack_create_takeshita/new_move_k_medoids_100_30_1222.csv')

# 動画特徴量を追加するためデータセットを2倍のサイズにする
# main_df_2 = main_df_1.copy()
main_df = pd.concat([main_df_1, main_df_2])

# # idを削除してindexを振り直す
main_df = main_df.drop('id', axis=1).reset_index(drop=True)

# # row[1]:id
# # row[2:]:move feature value
for row in move_df.itertuples(name=None):
	# -42600からmove feature valueのデータが保存されている
	main_df.iloc[row[1], -42600:] = row[2:]

main_df = main_df.reset_index()
main_df = main_df.rename(columns={'index':'id'})
# 20221217 main_dfの動き特徴量をすべて入れ替える #
# 既存の動き特徴量を削除
# main_df = main_df.drop(columns=main_df.columns[-57600:])

# # move_dfのidを1~14に置き換える
# move_df = move_df.drop('id', axis=1)
# move_df = move_df.reset_index()
# move_df = move_df.rename(columns={'index':'id'})

# # merge, id基準
# main_df = pd.merge(main_df, move_df, on='id')

# # 保存
main_df.to_csv('densetrack_create_takeshita/main_new_1222.csv', index=False)
