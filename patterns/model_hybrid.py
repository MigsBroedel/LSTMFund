import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class PatternLabeler:
    """Sistema de labeling automático para padrões"""
    
    def create_labels(self, df, lookforward=10, threshold=0.03):
        """Cria labels baseado em movimentos futuros de preço"""
        print("🏷️  Criando labels para treinamento...")
        
        labels = []
        future_returns = df['Close'].pct_change(lookforward).shift(-lookforward)
        
        for i in range(len(df) - lookforward):
            ret = future_returns.iloc[i]
            if ret > threshold:
                labels.append(2)  # COMPRA - movimento positivo forte
            elif ret < -threshold:
                labels.append(1)  # VENDA - movimento negativo forte
            else:
                labels.append(0)  # NEUTRO - movimento lateral
                
        # Preencher últimos valores com neutro
        labels.extend([0] * (len(df) - len(labels)))
        
        print(f"📊 Distribuição de labels: Compra={labels.count(2)}, Venda={labels.count(1)}, Neutro={labels.count(0)}")
        return np.array(labels)

class HybridPatternModel:
    """Modelo híbrido CNN-LSTM para reconhecimento de padrões"""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.label_encoder = PatternLabeler()
        
    def prepare_data(self, df):
        """Prepara dados para treinamento"""
        print("🔄 Preparando dados para o modelo...")
        
        # Criar labels
        labels = self.label_encoder.create_labels(df)
        
        # Selecionar apenas dados com labels válidos
        valid_indices = ~pd.isna(labels)
        df_valid = df[valid_indices]
        labels_valid = labels[valid_indices]
        
        # Garantir que temos dados suficientes
        if len(df_valid) < self.sequence_length * 2:
            raise ValueError(f"Dados insuficientes: {len(df_valid)} amostras, necessárias {self.sequence_length * 2}")
        
        # Preparar sequências
        X, y = self._create_sequences(df_valid, labels_valid)
        
        print(f"📦 Dados preparados: {X.shape[0]} sequências, {X.shape[2]} features")
        return X, y
    
    def _create_sequences(self, df, labels):
        """Cria sequências temporais para o modelo"""
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        self.feature_columns = feature_cols
        
        # Normalizar features
        feature_data = df[feature_cols].values
        feature_data = self.scaler.fit_transform(feature_data)
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(feature_data)):
            X.append(feature_data[i-self.sequence_length:i])
            y.append(labels[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, num_features, num_classes=3):
        """Constrói modelo híbrido CNN-LSTM"""
        print("🏗️  Construindo modelo híbrido CNN-LSTM...")
        
        inputs = keras.Input(shape=(self.sequence_length, num_features))
        
        # CNN para features espaciais/padrões locais
        conv1 = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.Dropout(0.3)(conv1)
        
        conv2 = layers.Conv1D(128, 5, activation='relu', padding='same')(conv1)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.Dropout(0.3)(conv2)
        
        # LSTM para dependências temporais
        lstm1 = layers.LSTM(128, return_sequences=True, dropout=0.3)(conv2)
        lstm2 = layers.LSTM(64, dropout=0.3)(lstm1)
        
        # Attention mechanism
        attention = layers.Dense(64, activation='tanh')(lstm2)
        attention = layers.Dense(1, activation='softmax')(attention)
        
        # Multi-task learning
        flattened = layers.Flatten()(attention)
        
        # Camadas densas
        dense1 = layers.Dense(128, activation='relu')(flattened)
        dense1 = layers.Dropout(0.2)(dense1)
        
        dense2 = layers.Dense(64, activation='relu')(dense1)
        dense2 = layers.Dropout(0.2)(dense2)
        
        # Saída
        outputs = layers.Dense(num_classes, activation='softmax', name='pattern')(dense2)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("✅ Modelo construído com sucesso!")
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """Treina o modelo"""
        if self.model is None:
            raise ValueError("Modelo não foi construído. Use build_model() primeiro.")
        
        print("🎯 Iniciando treinamento do modelo...")
        
        # Dividir dados
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False, random_state=42
        )
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        
        # Treinar
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("✅ Treinamento concluído!")
        return history
    
    def predict(self, X):
        """Faz predições"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado.")
        
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def evaluate(self, X, y):
        """Avalia o modelo"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado.")
        
        predictions = self.predict(X)
        
        print("\n📊 Avaliação do Modelo:")
        print("-" * 50)
        print("Relatório de Classificação:")
        print(classification_report(y, predictions, target_names=['NEUTRO', 'VENDA', 'COMPRA']))
        
        print("\nMatriz de Confusão:")
        cm = confusion_matrix(y, predictions)
        print(cm)
        
        return predictions, cm
    
    def save_model(self, filepath):
        """Salva o modelo treinado"""
        if self.model is None:
            raise ValueError("Nenhum modelo para salvar.")
        
        self.model.save(filepath)
        print(f"💾 Modelo salvo em: {filepath}")
    
    def load_model(self, filepath):
        """Carrega um modelo salvo"""
        self.model = keras.models.load_model(filepath)
        print(f"📂 Modelo carregado de: {filepath}")