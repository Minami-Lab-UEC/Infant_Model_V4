import pandas as pd

main_df_1 = pd.read_csv('../table/main_new.csv')
move_df = pd.read_csv('densetrack_create_takeshita/new_move_k_medoids_100_30.csv')

# 動画特徴量を追加するためデータセットを2倍のサイズにする
main_df_2 = main_df_1.copy()
main_df = pd.concat([main_df_1, main_df_2])

# idを削除してindexを振り直す
main_df.reset_index(drop=True).drop('id', axis=1)