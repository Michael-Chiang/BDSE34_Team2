from gensim.models import Word2Vec
import pandas as pd
import os

folder = 'stock_data_cleaned'
file_name = 'stock_vector_and_encoding.csv'
vec_model_name = 'stockID2vec.model'
file_path = os.path.join(folder, file_name)
vec_model_path = os.path.join(folder, vec_model_name)

if not os.path.exists(folder):
    os.mkdir(folder)


df = pd.read_csv('./stock_data/stock_all_include_type .csv')
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# 句子 (也可以是一篇文章)
sentences = [sentence.split() for sentence in df['Symbol'].values]

# 設定參數
sg = 0  # sg=1 -> skip-gram, sg=0 -> cbow

# 向前看幾個字或向後看幾個字
window_size = 2  # 越大越好

# 向量維度
vector_size = 5  # 100, 300, 500, ...

# 訓練幾回
epochs = 20

# 最少多少個字才會被使用
min_count = 1

# seed
seed = 42

# 建立 Word2Vec 模型
model = Word2Vec(
    sentences,
    vector_size=vector_size,
    window=window_size,
    sg=sg,
    min_count=1,
    seed=seed,
    epochs=epochs)


# 儲存模型
model.save(vec_model_path)

# 讀取模型
model = Word2Vec.load(vec_model_path)


new_df = pd.DataFrame(df['Symbol'])
results = []
for e in df['Symbol']:
    results.append(model.wv[e])
embedding_df = pd.DataFrame(results, columns=[
                            'Word_Embedding_Dim1', 'Word_Embedding_Dim2', 'Word_Embedding_Dim3', 'Word_Embedding_Dim4', 'Word_Embedding_Dim5'])

# 將embedding_df與原始的df合併
new_df = pd.concat([new_df, embedding_df], axis=1)

new_df = pd.concat([new_df, pd.get_dummies(df['Sector'], dtype=int)], axis=1)


new_df.to_csv(file_path, index=False)
