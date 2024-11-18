# README

### ビルド

```
$ docker build -t pgvector-env .
$ docker login
$ docker tag pgvector-env engtelur4/pgvector-env:latest
$ docker push engtelur4/pgvector-env:latest
```

### イメージ使用方法

```
$ docker pull engtelur4/pgvector-env:latest
$ docker run -it pgvector-env bash
```

### 参考サイト

- [マルチモーダル画像検索アプリを作ってみた！](https://qiita.com/yuji-arakawa/items/70470b348c90adb82b7f "マルチモーダル画像検索アプリを作ってみた！")
- [Japanese Stable CLIP による画像の分類（a.k.a. 画像によるテキストの検索）、そして画像検索に向けて](https://qiita.com/yuji-arakawa/items/042937eaf16fa00cf491 "Japanese Stable CLIP による画像の分類（a.k.a. 画像によるテキストの検索）、そして画像検索に向けて")

### 参考サイト(Python)

- [Python の nonlocal と global](https://qiita.com/domodomodomo/items/6df1419767e8acb99dd7 "Python の nonlocal と global")
- [Pythonの変数スコープの話](https://qiita.com/msssgur/items/12992fc816e6adf32cff "Pythonの変数スコープの話")
- [Python最新バージョン対応！より良い型ヒントの書き方](https://gihyo.jp/article/2022/09/monthly-python-2209 "Python最新バージョン対応！より良い型ヒントの書き方")
- [Python 3.10の新機能(その6） 明示的な型エイリアス](https://www.python.jp/news/wnpython310/typealias.html "Python 3.10の新機能(その6） 明示的な型エイリアス")

### 参考サイト(パフォーマンス)

- [Windows 10の電源オプションに「高パフォーマンス」「究極のパフォーマンス」を追加する](https://atmarkit.itmedia.co.jp/ait/articles/1810/29/news019.html "Windows 10の電源オプションに「高パフォーマンス」「究極のパフォーマンス」を追加する")

### 参考サイト(正規化)

- [正規化（Normalization）／標準化（Standardization）とは？：AI・機械学習の用語辞典 - ＠IT](https://atmarkit.itmedia.co.jp/ait/articles/2110/07/news027.html "正規化（Normalization）／標準化（Standardization）とは？：AI・機械学習の用語辞典 - ＠IT")

### 参考サイト(データベース)

- [カーソル【DB】とは｜「分かりそう」で「分からない」でも「分かった」気になれるIT用語辞典](https://wa3.i-3-i.info/word11582.html "カーソル【DB】とは｜「分かりそう」で「分からない」でも「分かった」気になれるIT用語辞典")

### 参考サイト(PostgreSQL)

- [とほほのPostgreSQL入門](https://www.tohoho-web.com/ex/postgresql.html "とほほのPostgreSQL入門")
  - PostgreSQL自体の使い方
- [dockerでPostgreSQLのコンテナ作成と初期化](https://qiita.com/asylum/items/17e655d8369c19affbc3 "dockerでPostgreSQLのコンテナ作成と初期化")
- [PythonでPostgreSQLとやりとりする](https://zenn.dev/collabostyle/articles/36e822520182d3 "PythonでPostgreSQLとやりとりする")
- [PythonでPostgreSQLに接続して、データを挿入、取得、更新する方法](https://pydocument.hatenablog.com/entry/2023/03/31/000945 "PythonでPostgreSQLに接続して、データを挿入、取得、更新する方法")
- [【Docker】PostgreSQLコンテナに初期データを投入する(docker-entrypoint-initdb.d)](https://atsum.in/linux/docker-postgres-init/ "【Docker】PostgreSQLコンテナに初期データを投入する(docker-entrypoint-initdb.d)")
- [postgreSQLにコマンドラインからSQLファイルを実行](https://qiita.com/Takashi_Nishimura/items/da5551e6a4cb4b64f055 "postgreSQLにコマンドラインからSQLファイルを実行")
- [PostgreSQLにバイナリデータを格納／出力する](https://www.insight-ltd.co.jp/tech_blog/postgresql/806/ "PostgreSQLにバイナリデータを格納／出力する")
- [Psycopg – PostgreSQL database adapter for Python](https://www.psycopg.org/docs/index.html "Psycopg – PostgreSQL database adapter for Python")

### 参考サイト(pgvector)

- [象使いのための pgvector 入門 (1)](https://qiita.com/hmatsu47/items/b393cecef8ed9df57c35 "象使いのための pgvector 入門 (1)")
  - Dockerをつかってpgvectorの導入から、L2距離、内積、コサイン類似度の3つを使った距離計測までやっている

### 参考サイト(PyTorch)

- [pyTorchのTensor型とは](https://qiita.com/mathlive/items/241bfb42d852bb801b96 "pyTorchのTensor型とは")

### 参考サイト(Docker)

- [devcontainerの運用ベストプラクティス #Docker - Qiita](https://qiita.com/1mono2/items/5bbf91f588ab9d5cd444 "devcontainerの運用ベストプラクティス #Docker - Qiita")
- [Docker や VSCode + Remote-Container のパーミッション問題に立ち向かう](https://zenn.dev/forrep/articles/8c0304ad420c8e "Docker や VSCode + Remote-Container のパーミッション問題に立ち向かう")

### 参考サイト(GPU)

- [メモリ、CPU、GPU に対する実行時オプション — Docker-docs-ja 19.03 ドキュメント](https://docs.docker.jp/v19.03/config/container/resource_constraints.html "メモリ、CPU、GPU に対する実行時オプション — Docker-docs-ja 19.03 ドキュメント")
- [Use a GPU  |  TensorFlow Core](https://www.tensorflow.org/guide/gpu "Use a GPU  |  TensorFlow Core")

### 参考サイト(Gradio)

- [gradio 入門 (1) - 事始め｜npaka](https://note.com/npaka/n/nb9d4902f8f4d "gradio 入門 (1) - 事始め｜npaka")
- [イベントハンドラーとイベントリスナーの違い について](https://designare.jp/blog/tokuyasu/%E3%82%A4%E3%83%99%E3%83%B3%E3%83%88%E3%83%8F%E3%83%B3%E3%83%89%E3%83%A9%E3%83%BC%E3%81%A8%E3%82%A4%E3%83%99%E3%83%B3%E3%83%88%E3%83%AA%E3%82%B9%E3%83%8A%E3%83%BC%E3%81%AE%E9%81%95%E3%81%84.html "イベントハンドラーとイベントリスナーの違い について")

### 参考サイト(大規模開発)

- [Pythonで中~大規模開発をする際のススメ5選](https://qiita.com/ForestMountain1234/items/70a499b1c00175b77407 "Pythonで中~大規模開発をする際のススメ5選")
- [【Pythonコーディング規約】PEP 8 vs Google Style](https://qiita.com/hi-asano/items/f43ced224483ea1f62f4 "【Pythonコーディング規約】PEP 8 vs Google Style")

### 参考サイト(共同編集)

- [Visual Studio Codeの拡張機能を使用し、共同編集を行う方法](https://engineer-blog.ajike.co.jp/vscode-liveshare/ "Visual Studio Codeの拡張機能を使用し、共同編集を行う方法")

### 参考サイト(環境変数)

- [【python3】dotenvを使って環境変数を管理する](https://note.com/yucco72/n/nb52bfb6d65bb "【python3】dotenvを使って環境変数を管理する")

### メモ

- pgvectorはPostgreSQLの拡張機能
- pgvectorはPostgreSQLに以下の機能を追加するもの
  - ベクトル(vector)データ型
  - ベクトル関数(距離計算など)・オペレータ
  - 近似最近傍探索用のインデックス
- ベクトル検索は「精密な検索をしようとするとテーブル内の全行に対して距離を計算し比較する必要がある」ので、通常のデータ行以上に検索負荷に配慮する必要がある

