import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class AdvancedFeatureEngineer:
    """Classe para engenharia de features avançadas"""
    
    def add_advanced_features(self, df):
        """Adiciona features técnicas mais sofisticadas"""
        
        print("🛠️  Calculando features avançadas...")
        
        # Preço e retornos
        df = self._add_price_features(df)
        
        # Volatilidade
        df = self._add_volatility_features(df)
        
        # Momentum
        df = self._add_momentum_features(df)
        
        # Volume
        df = self._add_volume_features(df)
        
        # Tendência
        df = self._add_trend_features(df)
        
        # Features cíclicas
        df = self._add_cyclical_features(df)
        
        # Regime de mercado
        df = self._add_market_regime_features(df)
        
        print(f"✅ Features calculadas: {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])} indicadores")
        return df
    
    def _add_price_features(self, df):
        """Features de preço"""
        df['Price_Change'] = df['Close'].pct_change()
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['High_Low_Spread'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Spread'] = (df['Close'] - df['Open']) / df['Open']
        
        # Retornos em diferentes períodos
        for period in [1, 5, 10, 20]:
            df[f'Return_{period}d'] = df['Close'].pct_change(period)
            
        return df
    
    def _add_volatility_features(self, df):
        """Features de volatilidade"""
        # ATR (Average True Range)
        df['Volatility_ATR'] = self.calculate_atr(df)
        df['Volatility_Rolling'] = df['Close'].rolling(20).std()
        
        # Bandas de volatilidade
        df['Volatility_Upper'] = df['Close'].rolling(20).mean() + 2 * df['Volatility_Rolling']
        df['Volatility_Lower'] = df['Close'].rolling(20).mean() - 2 * df['Volatility_Rolling']
        df['Volatility_Position'] = (df['Close'] - df['Volatility_Lower']) / (df['Volatility_Upper'] - df['Volatility_Lower'])
        
        return df
    
    def _add_momentum_features(self, df):
        """Features de momentum"""
        df['Momentum_ROC'] = self.rate_of_change(df['Close'], 10)
        df['Momentum_Stochastic'] = self.stochastic_oscillator(df)
        
        # RSI em diferentes períodos
        for period in [7, 14, 21]:
            df[f'RSI_{period}'] = self.calculate_rsi(df['Close'], period)
            
        # MACD
        macd, signal, hist = self.calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = hist
        
        return df
    
    def _add_volume_features(self, df):
        """Features de volume"""
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Volume Profile
        df['Volume_Profile'] = self.volume_profile(df)
        df['Volume_RSI'] = self.volume_rsi(df)
        
        # OBV (On Balance Volume)
        df['OBV'] = self.calculate_obv(df)
        
        return df
    
    def _add_trend_features(self, df):
        """Features de tendência"""
        # Médias móveis
        periods = [5, 10, 12, 20, 26, 50, 200]  # Adicionados 12 e 26 para o MACD e MA_Cross
        for period in periods:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            
        # Tendência das médias
        df['Trend_SMA'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
        df['MA_Cross'] = np.where(df['EMA_12'] > df['EMA_26'], 1, -1)
        
        return df
    
    def _add_cyclical_features(self, df):
        """Features cíclicas temporais"""
        if hasattr(df.index, 'dayofweek'):
            df['Day_of_Week'] = df.index.dayofweek
            df['Month_of_Year'] = df.index.month
            df['Week_of_Year'] = df.index.isocalendar().week
            
            # Features senoidais para capturar sazonalidade
            df['Day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['Day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            df['Month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
            
        return df
    
    def _add_market_regime_features(self, df, period=50):
        """Detecta regime de mercado"""
        returns = df['Close'].pct_change()
        volatility = returns.rolling(period).std()
        
        # Regime baseado na volatilidade
        vol_threshold = volatility.median()
        df['Market_Regime'] = np.where(volatility > vol_threshold, 1, 0)  # 1 = alta vol, 0 = baixa vol
        
        # Tendência de mercado
        price_trend = df['Close'].rolling(period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] > 0 if len(x) == period else np.nan
        )
        df['Price_Trend'] = price_trend
        
        return df
    
    # Métodos auxiliares
    def calculate_atr(self, df, period=14):
        """Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def rate_of_change(self, prices, period):
        """Rate of Change"""
        return prices.pct_change(period)
    
    def stochastic_oscillator(self, df, period=14):
        """Stochastic Oscillator %K"""
        low_min = df['Low'].rolling(period).min()
        high_max = df['High'].rolling(period).max()
        return 100 * (df['Close'] - low_min) / (high_max - low_min)
    
    def volume_profile(self, df, period=20):
        """Volume Profile"""
        return df['Volume'] / df['Volume'].rolling(period).mean()
    
    def volume_rsi(self, df, period=14):
        """RSI do Volume"""
        volume_change = df['Volume'].pct_change()
        gain = volume_change.where(volume_change > 0, 0).rolling(period).mean()
        loss = (-volume_change.where(volume_change < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_obv(self, df):
        """On Balance Volume"""
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return obv
    
    def calculate_rsi(self, prices, period=14):
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist