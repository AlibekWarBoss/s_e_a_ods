# Импорт библиотек
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Multiply, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Permute, Reshape, Activation
from sklearn.model_selection import train_test_split

# Установка ресурсов NLTK
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Функция предобработки текста
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = ' '.join([stemmer.stem(lemmatizer.lemmatize(word)) for word in text.split() if word not in stop_words])
    return text

# Загрузка данных
print("Loading combined data...")
data = pd.read_csv('combined_data.csv')

# Проверка правильности названия колонок
data.columns = ['text', 'label']

# Разделение данных на текст и метки
print("Splitting data...")
X = data['text'].astype(str).apply(preprocess_text)
y = data['label']

# Токенизация и паддинг
print("Tokenizing and padding sequences...")
tokenizer = Tokenizer(num_words=50000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
max_length = 300
X_padded = pad_sequences(X_seq, maxlen=max_length, padding='post')

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Применение SMOTE для устранения дисбаланса классов
print("Balancing data with SMOTE...")
try:
    smote = SMOTE(random_state=42)
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_reshaped, y_train)
    X_train_balanced = X_train_balanced.reshape(X_train_balanced.shape[0], max_length)
except ValueError as e:
    print(f"SMOTE error: {e}. Proceeding without oversampling.")
    X_train_balanced, y_train_balanced = X_train, y_train

# Реализация слоя внимания
def attention_layer(inputs):
    attention_weights = Dense(1, activation='tanh')(inputs)
    attention_weights = Flatten()(attention_weights)
    attention_weights = Dense(inputs.shape[1], activation='softmax')(attention_weights)
    attention_weights = Activation('softmax')(attention_weights)
    attention_weights = Reshape((inputs.shape[1], 1))(attention_weights)
    attention_output = Multiply()([inputs, attention_weights])
    return attention_output

# Построение модели Bi-LSTM с CNN и Attention
print("Building Bi-LSTM model...")
input_layer = Input(shape=(max_length,))
embedding_layer = Embedding(input_dim=50000, output_dim=128, input_length=max_length)(input_layer)
conv_layer = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding_layer)
pool_layer = MaxPooling1D(pool_size=2)(conv_layer)
bi_lstm_layer = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(pool_layer)
attention_output = attention_layer(bi_lstm_layer)
flatten_layer = Flatten()(attention_output)
dropout_layer = Dropout(0.5)(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dropout_layer)

model_bilstm = Model(inputs=input_layer, outputs=output_layer)

model_bilstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Ранняя остановка для предотвращения переобучения
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Обучение модели
print("Training Bi-LSTM model...")
model_bilstm.fit(
    X_train_balanced, y_train_balanced,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

# Оценка модели
print("Evaluating Bi-LSTM model...")
y_pred_bilstm = (model_bilstm.predict(X_test) > 0.5).astype(int)
f1_bilstm = f1_score(y_test, y_pred_bilstm)
print(f"F1-метрика (Bi-LSTM): {f1_bilstm:.4f}")
