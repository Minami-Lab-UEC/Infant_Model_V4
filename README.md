# Infant_Model_V4
担当者 : 幼児語彙獲得モデル（M2 竹下）
期間 : 2021年4月～2023年3月

## 概要
田口さんの提案した幼児語彙獲得モデルを参考に、品詞情報を順に獲得する幼児の語彙獲得モデルを提案。品詞情報は名詞から動詞の順とした。
実験では、名詞語彙の獲得を行ったモデルパラメータを事前学習したモデルと事前学習なしのモデルで比較を行った。事前学習ありのモデルは正答率が伸び悩んだ結果に対し、事前学習なしのモデルは正答率がほぼ1まで伸びた。この実験により、実世界のように名詞と動詞の語彙の学習を完全に切り離さない学習が重要ではないかという結論に至った。

## github repository
[Infant_Model_V4](https://github.com/Minami-Lab-UEC/Infant_Model_V4)

## 作業サーバ
- IPアドレス : 172.21.64.245
- ユーザ名 : takeshita
- パスワード : takeshita
- 作業ディレクトリ : /home/takeshita/Infant_Model_V4
- 作業環境 : Dockerコンテナ

### 作業環境
- Dockerコンテナ上で作業
- Dockerコンテナの作成手順 \
	1. docker_takeディレクトリに移動 (cd /home/takeshita/Infant_Model_V4/docker_take)
	2. Dockerfileを元にコンテナを作成するためにtakeshita_build.shを実行 (./takeshita_build.sh)
	3. コンテナを起動するためにtakeshita_run.shを実行してbash呼び出し→作業 (./takeshita_run.sh)
- コンテナでモデル学習スクリプトを実行しっぱなしにするにはtakeshita_train.shを実行

### モデル学習
- proposed_onememory_except.pyを実行する
	- モデル学習スクリプト
	- 学習部分を2回ループ
		- 1回目 : 事前学習なしの名詞→動詞語彙の学習
		- 2回目 : 名詞語彙の事前学習ありで名詞→動詞語彙の学習
	- 事前学習パラメータは result_9/check_points/my_checkpoint_only_noun_10000.data-00000-of-00001と.indexを利用(nounonly_pre_main.pyで名詞語彙を学習した結果)
	- スクリプトの最終更新verでは2回ループする学習自体を、さらに10回ループさせている
		- より多くの実験結果を得るため
	- **すみません、論文データは乱数のseed固定していなかったため、同じ結果が得られないです...**

## Moments_in_Time
- Dense Trajectories(動き特徴量の作成スクリプト)に渡す動作動画にはMoments in Time Datasetを使っている(田口さん論文より)
- Moments in Time Datasetはサイズが大きすぎるので、Moments in Time Mini Datasetを使うのがよい
- Moments in Time Miniはデータサーバー(172.21.65.29)の2023/M2/Takeshita/Infant_Model_takeshita/に保存してある



