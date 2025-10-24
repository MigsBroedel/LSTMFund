import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
    def __init__(self, alpha_vantage_key=None):
        self.sources = ['yfinance', 'alpha_vantage']
        self.alpha_vantage_key = alpha_vantage_key
        
    def fetch_real_data(self, symbol, period='2y', interval='1d'):
        """Busca dados reais do Yahoo Finance"""
        try:
            print(f"📥 Baixando dados para {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"❌ Nenhum dado encontrado para {symbol}")
                return None
                
            data = self._validate_data(data)
            print(f"✅ Dados baixados: {len(data)} registros de {data.index[0].date()} a {data.index[-1].date()}")
            return data
            
        except Exception as e:
            print(f"❌ Erro ao baixar dados: {e}")
            return None

    def _validate_data(self, data):
        """Valida e limpa os dados"""
        # Remove linhas com NaN
        original_len = len(data)
        data = data.dropna()
        
        # Verifica se há dados suficientes
        if len(data) < 50:
            raise ValueError("Dados insuficientes após limpeza")
            
        print(f"📊 Dados válidos: {len(data)}/{original_len} registros mantidos")
        return data

    def fetch_multiple_symbols(self, symbols, period='1y'):
        """Busca dados para múltiplos símbolos"""
        all_data = {}
        for symbol in symbols:
            data = self.fetch_real_data(symbol, period)
            if data is not None:
                all_data[symbol] = data
        return all_data

    def get_market_indicators(self):
        """Busca indicadores macroeconômicos"""
        try:
            # Índices de referência
            sp500 = yf.Ticker('^GSPC').history(period='1y')
            vix = yf.Ticker('^VIX').history(period='1y')
            dolar = yf.Ticker('BRL=X').history(period='1y')
            
            return {
                'sp500': sp500,
                'vix': vix,
                'dolar': dolar
            }
        except Exception as e:
            print(f"⚠️  Erro ao buscar indicadores macro: {e}")
            return None