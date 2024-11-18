# PostgreSQLのpgvector拡張を含むイメージを使用
FROM postgres:15

# デフォルトのshellをbashに変更
SHELL ["/bin/bash", "-c"]

# タイムゾーンを設定
# ENV TZ=Asia/Tokyo
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 環境変数を設定
ENV POSTGRES_PASSWORD='y6KDgfg9'
ENV POSTGRES_USER='postgres'
ENV POSTGRES_DB='postgres'

# 必要なパッケージのインストール
RUN apt update -y && apt upgrade -y && \
    apt install -y sudo wget vim git locales build-essential postgresql-server-dev-15 curl libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 procps libgl1-mesa-dev libglib2.0-0 && \
    git clone https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && make install && \
    cd .. && rm -rf pgvector

# ロケールの設定
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen en_US.UTF-8
ENV LANG="en_US.UTF-8"
ENV LANGUAGE="en_US:en"
ENV LC_ALL="en_US.UTF-8"

# グループ & ユーザの作成 & sudo権限の付与
ARG USERNAME=dev-user
ARG GROUPNAME=dev-user
ARG UID=1100
ARG GID=1100
RUN groupadd -g ${GID} ${GROUPNAME} && \
    useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ワークスペースの作成 & 指定 & ユーザ指定
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
USER ${USERNAME}

# 初期化用のSQLファイルをコンテナにコピー
COPY src/ .
RUN sudo chown -R ${USERNAME}:${GROUPNAME} .

# Anacondaのインストール
# https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
RUN bash anaconda.sh -b

# Anacondaのパスを設定
ENV PATH=/home/${USERNAME}/anaconda3/bin:$PATH

# bashでcondaを使用できるようにする
RUN echo "source /home/${USERNAME}/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc

# pgvector-envというpython3.10ベースの仮想環境を構築する
RUN conda create -n pgvector-env python=3.10

# 一度仮想環境をアクティブにし、requirements.txtに記述されているパッケージをすべてインストールする
RUN source activate pgvector-env && pip install -r requirements.txt

# bash起動時に仮想環境に入るようにする
RUN echo "conda activate pgvector-env" >> ~/.bashrc

# マルチモーダル画像検索アプリのサンプルコードをクローン
RUN git clone https://github.com/kutsushitaneko/multimodal-image_search.git

# Jpanese Stable CLIPのサンプルコードをクローン
RUN git clone https://github.com/kutsushitaneko/Japanese_Stable_CLIP.git


# PostgreSQLを初期化して起動、データを挿入する
RUN sudo chown -R dev-user:dev-user /var/lib/postgresql/data && \
    /usr/lib/postgresql/15/bin/initdb -D /var/lib/postgresql/data && \
    /usr/lib/postgresql/15/bin/postgres -D /var/lib/postgresql/data & \
    until pg_isready -h localhost -p 5432; do \
        echo "Waiting for PostgreSQL to start..."; \
        sleep 1; \
    done && \
    /usr/lib/postgresql/15/bin/psql -d postgres -c "CREATE ROLE postgres WITH LOGIN SUPERUSER PASSWORD 'y6KDgfg9';" && \
    psql -U postgres -d postgres -f ./setup.sql && \
    source activate pgvector-env && \
    python register_images.py

RUN echo "pgrep -x postgres > /dev/null || /usr/lib/postgresql/15/bin/postgres -D /var/lib/postgresql/data & sleep 1" >> ~/.bashrc

CMD ["bash", "-l"]

