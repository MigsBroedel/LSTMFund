"""
INSTALAÇÃO:
pip install yfinance pandas numpy scikit-learn tensorflow scipy

OUTPUTS GERADOS:
- ranking_fundamentalista.csv: Ranking completo com todos os scores
- sinais_ensemble.csv: Sinais formatados para integração
- red_flags_detalhados.csv: Lista completa de alertas por empresa
- metricas_validacao.txt: Relatório de qualidade do especialista

"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from scipy.stats import spearmanr
import pickle
import time
import json

warnings.filterwarnings('ignore')


#  COLETA DE DADOS


class ColetorDados:
    """Coleta dados fundamentalistas via yfinance"""

    def __init__(self):
        self.scaler = RobustScaler()

    def extrair_fundamentos(self, ticker, stock_info, financials, balance_sheet, cashflow, date):
        """Extrai indicadores fundamentalistas completos"""
        try:
            if date not in financials.columns:
                return None

            idx = financials.columns.get_loc(date)

            # VALUATION
            pe_ratio = stock_info.get('trailingPE', np.nan)
            pb_ratio = stock_info.get('priceToBook', np.nan)
            ps_ratio = stock_info.get('priceToSalesTrailing12Months', np.nan)

            # DRE
            total_revenue = financials.iloc[:, idx].get('Total Revenue', np.nan)
            gross_profit = financials.iloc[:, idx].get('Gross Profit', np.nan)
            operating_income = financials.iloc[:, idx].get('Operating Income', np.nan)
            net_income = financials.iloc[:, idx].get('Net Income', np.nan)
            ebitda = financials.iloc[:, idx].get('EBITDA', np.nan)

            # BALANÇO
            total_assets = balance_sheet.iloc[:, idx].get('Total Assets', np.nan)
            total_equity = balance_sheet.iloc[:, idx].get('Stockholders Equity', np.nan)
            current_assets = balance_sheet.iloc[:, idx].get('Current Assets', np.nan)
            current_liabilities = balance_sheet.iloc[:, idx].get('Current Liabilities', np.nan)
            total_debt = balance_sheet.iloc[:, idx].get('Total Debt', np.nan)
            cash = balance_sheet.iloc[:, idx].get('Cash And Cash Equivalents', np.nan)

            # FLUXO DE CAIXA
            operating_cashflow = cashflow.iloc[:, idx].get('Operating Cash Flow', np.nan)
            free_cashflow = cashflow.iloc[:, idx].get('Free Cash Flow', np.nan)

            # CALCULAR INDICADORES
            gross_margin = (gross_profit / total_revenue * 100) if total_revenue else np.nan
            operating_margin = (operating_income / total_revenue * 100) if total_revenue else np.nan
            net_margin = (net_income / total_revenue * 100) if total_revenue else np.nan
            ebitda_margin = (ebitda / total_revenue * 100) if total_revenue else np.nan

            roe = (net_income / total_equity * 100) if total_equity else np.nan
            roa = (net_income / total_assets * 100) if total_assets else np.nan
            roic = (operating_income / (total_equity + total_debt) * 100) if (total_equity and total_debt) else np.nan

            current_ratio = (current_assets / current_liabilities) if current_liabilities else np.nan
            debt_to_equity = (total_debt / total_equity) if total_equity else np.nan
            net_debt = (total_debt - cash) if (total_debt and cash) else np.nan
            net_debt_to_ebitda = (net_debt / ebitda) if ebitda else np.nan
            asset_turnover = (total_revenue / total_assets) if total_assets else np.nan

            return {
                'ticker': ticker, 'date': date, 'pe_ratio': pe_ratio, 'pb_ratio': pb_ratio,
                'ps_ratio': ps_ratio, 'revenue': total_revenue, 'gross_profit': gross_profit,
                'operating_income': operating_income, 'net_income': net_income, 'ebitda': ebitda,
                'total_assets': total_assets, 'total_equity': total_equity, 'total_debt': total_debt,
                'operating_cashflow': operating_cashflow, 'free_cashflow': free_cashflow,
                'gross_margin': gross_margin, 'operating_margin': operating_margin,
                'net_margin': net_margin, 'ebitda_margin': ebitda_margin, 'roe': roe,
                'roa': roa, 'roic': roic, 'current_ratio': current_ratio,
                'debt_to_equity': debt_to_equity, 'net_debt_to_ebitda': net_debt_to_ebitda,
                'asset_turnover': asset_turnover
            }
        except:
            return None

    def coletar_empresa(self, ticker, n_quarters=24):
        try:
            ticker_yf = f"{ticker}.SA"
            stock = yf.Ticker(ticker_yf)

            info = stock.info
            financials_q = stock.quarterly_financials
            balance_q = stock.quarterly_balance_sheet
            cashflow_q = stock.quarterly_cashflow

            if financials_q.empty:
                return None

            dados_trimestres = []
            n_available = min(n_quarters, len(financials_q.columns))

            for i in range(n_available):
                date = financials_q.columns[i]
                trimestre_data = self.extrair_fundamentos(
                    ticker, info, financials_q, balance_q, cashflow_q, date
                )
                if trimestre_data:
                    dados_trimestres.append(trimestre_data)

            if not dados_trimestres:
                return None

            df = pd.DataFrame(dados_trimestres)
            df = df.sort_values('date')
            return df
        except:
            return None

    def coletar_multiplas_empresas(self, tickers, n_quarters=24):
        todos_dados = []
        print("\n" + "="*70)
        print("📥 COLETANDO DADOS FUNDAMENTALISTAS")
        print("="*70)

        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] {ticker}...", end=" ")
            df = self.coletar_empresa(ticker, n_quarters)

            if df is not None and len(df) >= 4:
                todos_dados.append(df)
                print(f"✅ {len(df)} trimestres")
            else:
                print(f"⚠️  Insuficiente")
            time.sleep(0.2)

        if not todos_dados:
            return None

        df_final = pd.concat(todos_dados, ignore_index=True)
        print(f"\n✅ Total: {len(df_final)} registros | {df_final['ticker'].nunique()} empresas")
        return df_final


# PROCESSAMENTO


class ProcessadorDados:

    @staticmethod
    def calcular_crescimento(df):
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        features = ['revenue', 'net_income', 'ebitda', 'operating_cashflow']

        for feature in features:
            df[f'{feature}_growth_qoq'] = df.groupby('ticker')[feature].pct_change() * 100
            df[f'{feature}_growth_yoy'] = df.groupby('ticker')[feature].pct_change(periods=4) * 100
        return df

    @staticmethod
    def adicionar_features_tecnicas(df):
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

        for col in ['roe', 'net_margin', 'current_ratio', 'ebitda_margin']:
            if col in df.columns:
                df[f'{col}_ma4'] = df.groupby('ticker')[col].transform(
                    lambda x: x.rolling(window=4, min_periods=1).mean()
                )

        for col in ['revenue', 'net_income']:
            if col in df.columns:
                df[f'{col}_volatility'] = df.groupby('ticker')[col].transform(
                    lambda x: x.rolling(window=4, min_periods=2).std()
                )
        return df

    @staticmethod
    def tratar_missing_data(df):
        for col in ['pe_ratio', 'pb_ratio', 'ps_ratio']:
            if col in df.columns:
                df[col] = df.groupby('ticker')[col].ffill()

        for col in ['gross_margin', 'operating_margin', 'net_margin', 'roe', 'roa']:
            if col in df.columns:
                df[col] = df.groupby('ticker')[col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )

        growth_cols = [col for col in df.columns if 'growth' in col]
        df[growth_cols] = df[growth_cols].fillna(0)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['ticker', 'date']]

        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        return df

    @staticmethod
    def calcular_retorno_futuro(df, horizonte_meses=3):
        print(f"\n📊 Calculando retornos futuros ({horizonte_meses} meses)...")
        retornos = []

        for idx, row in df.iterrows():
            try:
                ticker_yf = f"{row['ticker']}.SA"
                stock = yf.Ticker(ticker_yf)
                data_atual = pd.to_datetime(row['date'])
                data_futura = data_atual + pd.DateOffset(months=horizonte_meses)

                hist = stock.history(start=data_atual, end=data_atual + pd.DateOffset(days=5))
                hist_futuro = stock.history(start=data_futura, end=data_futura + pd.DateOffset(days=5))

                if not hist.empty and not hist_futuro.empty:
                    preco_atual = hist['Close'].iloc[0]
                    preco_futuro = hist_futuro['Close'].iloc[0]
                    retorno = (preco_futuro - preco_atual) / preco_atual * 100
                    retornos.append(retorno)
                else:
                    retornos.append(np.nan)
            except:
                retornos.append(np.nan)

        df['retorno_futuro'] = retornos
        validos = df['retorno_futuro'].notna().sum()
        print(f"   ✅ {validos}/{len(df)} targets válidos")
        return df


# MODELO LSTM 


class ModeloLSTM:

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.history = None

    def criar_modelo(self, input_shape):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def preparar_dados(self, df, lookback=4):
        meta_cols = ['ticker', 'date', 'retorno_futuro']
        self.feature_cols = [col for col in df.columns if col not in meta_cols]

        self.scaler = RobustScaler()
        df_normalized = df.copy()
        df_normalized[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])

        X, y, tickers_list, dates_list = [], [], [], []

        for ticker, group in df_normalized.groupby('ticker'):
            group = group.sort_values('date').reset_index(drop=True)
            if len(group) < lookback + 1:
                continue

            for i in range(len(group) - lookback):
                sequence = group.iloc[i:i+lookback][self.feature_cols].values
                target = group.iloc[i+lookback]['retorno_futuro']

                if pd.notna(target):
                    X.append(sequence)
                    y.append(target)
                    tickers_list.append(ticker)
                    dates_list.append(group.iloc[i+lookback]['date'])

        return np.array(X), np.array(y), tickers_list, dates_list

    def treinar(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=16):
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        print("\n" + "="*70)
        print("🧠 TREINANDO MODELO LSTM")
        print("="*70)

        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.criar_modelo(input_shape)

        print(f"Treino: {X_train.shape} | Validação: {X_val.shape}")

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        print(f"✅ Treinamento concluído! Loss final: {self.history.history['val_loss'][-1]:.4f}")
        return self.history


# ESPECIALISTA FUNDAMENTALISTA


class EspecialistaFundamentalista:
    """Sistema especialista otimizado para arquitetura multi-modelo"""

    def __init__(self, modelo_lstm=None, scaler=None, feature_cols=None):
        self.modelo = modelo_lstm
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.usa_lstm = (modelo_lstm is not None)

    def normalizar_percentil(self, series, ascending=True):
        if ascending:
            return series.rank(pct=True, method='average') * 100
        else:
            return (1 - series.rank(pct=True, method='average')) * 100

    def calcular_scores_especializados(self, df):
        """Scores especializados para ensemble"""

        # 1. QUALITY SCORE (Rentabilidade sustentável)
        df['quality_score'] = (
            self.normalizar_percentil(df['roe'], ascending=True) * 0.25 +
            self.normalizar_percentil(df['roic'], ascending=True) * 0.30 +
            self.normalizar_percentil(df['net_margin'], ascending=True) * 0.20 +
            self.normalizar_percentil(df['roa'], ascending=True) * 0.15 +
            self.normalizar_percentil(df['ebitda_margin'], ascending=True) * 0.10
        )

        # 2. GROWTH SCORE
        df['growth_score'] = (
            self.normalizar_percentil(df['revenue_growth_yoy'], ascending=True) * 0.35 +
            self.normalizar_percentil(df['net_income_growth_yoy'], ascending=True) * 0.35 +
            self.normalizar_percentil(df['ebitda_growth_yoy'], ascending=True) * 0.30
        )

        # 3. VALUE SCORE
        df['value_score'] = (
            self.normalizar_percentil(df['pe_ratio'], ascending=False) * 0.40 +
            self.normalizar_percentil(df['pb_ratio'], ascending=False) * 0.35 +
            self.normalizar_percentil(df['ps_ratio'], ascending=False) * 0.25
        )

        # 4. SAFETY SCORE
        df['safety_score'] = (
            self.normalizar_percentil(df['current_ratio'], ascending=True) * 0.30 +
            self.normalizar_percentil(df['debt_to_equity'], ascending=False) * 0.30 +
            self.normalizar_percentil(df['net_debt_to_ebitda'], ascending=False) * 0.40
        )

        # 5. MOMENTUM SCORE
        df['momentum_score'] = (
            self.normalizar_percentil(df['roe_ma4'], ascending=True) * 0.35 +
            self.normalizar_percentil(df['revenue_growth_qoq'], ascending=True) * 0.35 +
            self.normalizar_percentil(df['net_income_growth_qoq'], ascending=True) * 0.30
        )

        # 6. EFFICIENCY SCORE
        df['efficiency_score'] = (
            self.normalizar_percentil(df['roic'], ascending=True) * 0.40 +
            self.normalizar_percentil(df['asset_turnover'], ascending=True) * 0.30 +
            self.normalizar_percentil(df['operating_margin'], ascending=True) * 0.30
        )

        return df

    def detectar_red_flags(self, dados):
        """Sistema avançado de detecção de red flags"""
        red_flags = {'criticos': [], 'importantes': [], 'atencao': []}

        # CRÍTICOS
        if dados['roe'] < 0:
            red_flags['criticos'].append('ROE_NEGATIVO')
        if dados['roic'] < 0:
            red_flags['criticos'].append('ROIC_NEGATIVO')
        if dados['net_margin'] < 0:
            red_flags['criticos'].append('MARGEM_NEGATIVA')
        if dados['free_cashflow'] < 0:
            red_flags['criticos'].append('FCF_NEGATIVO')

        # IMPORTANTES
        if dados['roe'] > 20 and dados['roic'] < dados['roe'] / 3:
            red_flags['importantes'].append('ALAVANCAGEM_EXCESSIVA')
        if dados['debt_to_equity'] > 3:
            red_flags['importantes'].append('ENDIVIDAMENTO_CRITICO')
        if dados['net_debt_to_ebitda'] > 6:
            red_flags['importantes'].append('DIVIDA_ALTA')
        if dados['current_ratio'] < 1:
            red_flags['importantes'].append('LIQUIDEZ_CRITICA')

        # ATENÇÃO
        if 0 < dados['roe'] < 8:
            red_flags['atencao'].append('ROE_BAIXO')
        if 0 < dados['roic'] < 5:
            red_flags['atencao'].append('ROIC_BAIXO')
        if dados['pe_ratio'] > 30:
            red_flags['atencao'].append('PE_ELEVADO')

        # Risk Score
        risk_score = 100
        risk_score -= len(red_flags['criticos']) * 30
        risk_score -= len(red_flags['importantes']) * 15
        risk_score -= len(red_flags['atencao']) * 5
        risk_score = max(0, risk_score)

        return {
            'flags': red_flags,
            'risk_score': risk_score,
            'total_flags': sum(len(v) for v in red_flags.values())
        }

    def calcular_trend_signals(self, df_ticker):
        """Detecta tendências de melhora/piora"""
        if len(df_ticker) < 4:
            return {'trend': 'NEUTRO', 'strength': 0, 'roe_trend': 0, 'roic_trend': 0}

        recent = df_ticker.tail(4).sort_values('date')

        roe_trend = recent['roe'].diff().mean()
        roic_trend = recent['roic'].diff().mean()
        margin_trend = recent['net_margin'].diff().mean()
        revenue_trend = recent['revenue'].pct_change().mean() * 100

        trend_score = (
            (roe_trend / 2) * 0.3 +
            (roic_trend / 2) * 0.3 +
            (margin_trend / 2) * 0.2 +
            (revenue_trend / 10) * 0.2
        )

        if trend_score > 2:
            trend = 'MELHORANDO'
        elif trend_score < -2:
            trend = 'PIORANDO'
        else:
            trend = 'ESTAVEL'

        return {
            'trend': trend,
            'strength': abs(trend_score),
            'roe_trend': roe_trend,
            'roic_trend': roic_trend,
            'margin_trend': margin_trend
        }

    def calcular_ranking(self, df_atual, lookback=4):
        """Gera ranking completo com todos os sinais"""

        print("\n" + "="*70)
        print("🎯 CALCULANDO SCORES ESPECIALIZADOS")
        print("="*70)

        df_com_scores = self.calcular_scores_especializados(df_atual.copy())

        resultados = []
        red_flags_lista = []

        for ticker in df_com_scores['ticker'].unique():
            try:
                df_ticker = df_com_scores[df_com_scores['ticker'] == ticker].sort_values('date')
                dados_atuais = df_ticker.iloc[-1]

                # LSTM se disponível
                lstm_score = 0
                if self.usa_lstm and len(df_ticker) >= lookback:
                    try:
                        ultimos_dados = df_ticker.tail(lookback)
                        X = ultimos_dados[self.feature_cols].values
                        X_normalized = self.scaler.transform(X)
                        X_seq = X_normalized.reshape(1, lookback, -1)
                        lstm_score = self.modelo.predict(X_seq, verbose=0)[0][0]
                    except:
                        lstm_score = 0

                # Q-Score final
                if self.usa_lstm:
                    qscore = (
                        dados_atuais['quality_score'] * 0.35 +
                        dados_atuais['growth_score'] * 0.25 +
                        dados_atuais['value_score'] * 0.15 +
                        dados_atuais['safety_score'] * 0.10 +
                        lstm_score * 0.15
                    )
                else:
                    qscore = (
                        dados_atuais['quality_score'] * 0.40 +
                        dados_atuais['growth_score'] * 0.30 +
                        dados_atuais['value_score'] * 0.20 +
                        dados_atuais['safety_score'] * 0.10
                    )

                qscore = np.clip(qscore, 0, 100)

                # Red Flags
                rf = self.detectar_red_flags(dados_atuais)

                # Tendências
                trends = self.calcular_trend_signals(df_ticker)

                # Categorização
                if qscore >= 70 and rf['risk_score'] >= 70:
                    categoria = 'PREMIUM'
                elif qscore >= 50:
                    categoria = 'STANDARD'
                else:
                    categoria = 'SPECULATIVE'

                # Resultado
                resultados.append({
                    'ticker': ticker,
                    'qscore': qscore,
                    'quality_score': dados_atuais['quality_score'],
                    'growth_score': dados_atuais['growth_score'],
                    'value_score': dados_atuais['value_score'],
                    'safety_score': dados_atuais['safety_score'],
                    'momentum_score': dados_atuais['momentum_score'],
                    'efficiency_score': dados_atuais['efficiency_score'],
                    'lstm_score': lstm_score if self.usa_lstm else np.nan,
                    'risk_score': rf['risk_score'],
                    'total_flags': rf['total_flags'],
                    'flags_criticos': len(rf['flags']['criticos']),
                    'flags_importantes': len(rf['flags']['importantes']),
                    'flags_atencao': len(rf['flags']['atencao']),
                    'trend': trends['trend'],
                    'trend_strength': trends['strength'],
                    'roe_trend': trends['roe_trend'],
                    'roic_trend': trends['roic_trend'],
                    'categoria': categoria,
                    'roe': dados_atuais['roe'],
                    'roic': dados_atuais['roic'],
                    'pe_ratio': dados_atuais['pe_ratio'],
                    'pb_ratio': dados_atuais['pb_ratio'],
                    'net_margin': dados_atuais['net_margin'],
                    'revenue_growth_yoy': dados_atuais['revenue_growth_yoy'],
                    'debt_to_equity': dados_atuais['debt_to_equity'],
                    'current_ratio': dados_atuais['current_ratio'],
                    'free_cashflow': dados_atuais['free_cashflow']
                })

                # Red flags detalhados
                for flag in rf['flags']['criticos']:
                    red_flags_lista.append({
                        'ticker': ticker,
                        'tipo': 'CRITICO',
                        'flag': flag,
                        'risk_score': rf['risk_score']
                    })
                for flag in rf['flags']['importantes']:
                    red_flags_lista.append({
                        'ticker': ticker,
                        'tipo': 'IMPORTANTE',
                        'flag': flag,
                        'risk_score': rf['risk_score']
                    })
                for flag in rf['flags']['atencao']:
                    red_flags_lista.append({
                        'ticker': ticker,
                        'tipo': 'ATENCAO',
                        'flag': flag,
                        'risk_score': rf['risk_score']
                    })

            except Exception as e:
                continue

        df_ranking = pd.DataFrame(resultados)
        df_ranking = df_ranking.sort_values('qscore', ascending=False).reset_index(drop=True)
        df_ranking['rank'] = range(1, len(df_ranking) + 1)
        df_ranking['percentile'] = (1 - df_ranking['rank'] / len(df_ranking)) * 100

        df_red_flags = pd.DataFrame(red_flags_lista)

        print(f"✅ {len(df_ranking)} empresas ranqueadas")
        print(f"⚠️  {len(df_red_flags)} red flags detectados")

        return df_ranking, df_red_flags



# VALIDAÇÃO PARA ESPECIALISTA


def validar_especialista(df_ranking, df_completo):
    """Validação focada em qualidade do especialista, não em predição"""

    print("\n" + "="*70)
    print("🔍 VALIDAÇÃO DO ESPECIALISTA FUNDAMENTALISTA")
    print("="*70)

    validacao = {}

    # 1. SEPARAÇÃO: Top 10 vs Bottom 10
    top_10 = df_ranking.head(10)
    bottom_10 = df_ranking.tail(10)

    print(f"\n1️⃣ SEPARAÇÃO FUNDAMENTALISTA")
    print(f"   ROE médio Top 10: {top_10['roe'].mean():.1f}%")
    print(f"   ROE médio Bottom 10: {bottom_10['roe'].mean():.1f}%")
    print(f"   ROIC médio Top 10: {top_10['roic'].mean():.1f}%")
    print(f"   ROIC médio Bottom 10: {bottom_10['roic'].mean():.1f}%")

    separacao_roe = top_10['roe'].mean() > bottom_10['roe'].mean()
    separacao_roic = top_10['roic'].mean() > bottom_10['roic'].mean()

    if separacao_roe and separacao_roic:
        print("   ✅ EXCELENTE - Top 10 claramente superior")
        validacao['separacao'] = "APROVADO"
    else:
        print("   ❌ RUIM - Top 10 não é superior")
        validacao['separacao'] = "REPROVADO"

    # 2. QUALIDADE DO TOP 10
    print(f"\n2️⃣ QUALIDADE DO TOP 10")

    flags_criticos = top_10['flags_criticos'].sum()
    roe_neg = (top_10['roe'] < 0).sum()
    roic_baixo = (top_10['roic'] < 5).sum()

    print(f"   Flags críticos: {flags_criticos}")
    print(f"   ROE negativo: {roe_neg}")
    print(f"   ROIC < 5%: {roic_baixo}")

    if flags_criticos == 0 and roe_neg == 0 and roic_baixo <= 2:
        print("   ✅ EXCELENTE - Top 10 limpo")
        validacao['qualidade'] = "APROVADO"
    else:
        print("   ⚠️  ATENÇÃO - Top 10 tem problemas")
        validacao['qualidade'] = "REPROVADO"

    # 3. CONSISTÊNCIA INTERNA
    print(f"\n3️⃣ CONSISTÊNCIA DOS SCORES")

    corr_quality = df_ranking['quality_score'].corr(df_ranking['qscore'])
    corr_risk = df_ranking['risk_score'].corr(df_ranking['qscore'])

    print(f"   Corr(Quality, Q-Score): {corr_quality:.3f}")
    print(f"   Corr(Risk, Q-Score): {corr_risk:.3f}")

    if corr_quality > 0.7:
        print("   ✅ Scores consistentes")
        validacao['consistencia'] = "APROVADO"
    else:
        print("   ⚠️  Inconsistência detectada")
        validacao['consistencia'] = "REPROVADO"

    # 4. DISTRIBUIÇÃO
    print(f"\n4️⃣ DISTRIBUIÇÃO DAS CATEGORIAS")

    cat_counts = df_ranking['categoria'].value_counts()
    print(f"   PREMIUM: {cat_counts.get('PREMIUM', 0)}")
    print(f"   STANDARD: {cat_counts.get('STANDARD', 0)}")
    print(f"   SPECULATIVE: {cat_counts.get('SPECULATIVE', 0)}")

    premium_pct = cat_counts.get('PREMIUM', 0) / len(df_ranking) * 100
    if 10 <= premium_pct <= 30:
        print(f"   ✅ Distribuição adequada ({premium_pct:.1f}% PREMIUM)")
        validacao['distribuicao'] = "APROVADO"
    else:
        print(f"   ⚠️  Distribuição enviesada ({premium_pct:.1f}% PREMIUM)")
        validacao['distribuicao'] = "REPROVADO"

    # 5. CORRELAÇÃO COM RETORNOS (secundário)
    df_val = df_ranking[['ticker', 'qscore']].merge(
        df_completo.groupby('ticker').agg({'retorno_futuro': 'last'}).reset_index(),
        on='ticker',
        how='left'
    )
    df_val = df_val.dropna(subset=['retorno_futuro'])

    if len(df_val) >= 10:
        ic, pvalue = spearmanr(df_val['qscore'], df_val['retorno_futuro'])
        print(f"\n5️⃣ CORRELAÇÃO COM RETORNOS (SECUNDÁRIO)")
        print(f"   IC: {ic:.4f} | P-value: {pvalue:.4f}")
        validacao['ic'] = ic
        validacao['pvalue'] = pvalue
    else:
        validacao['ic'] = 0
        validacao['pvalue'] = 1

    # PONTUAÇÃO FINAL
    pontos = sum(1 for k in ['separacao', 'qualidade', 'consistencia', 'distribuicao']
                 if validacao.get(k) == "APROVADO")

    print(f"\n{'='*70}")
    print(f"PONTUAÇÃO ESPECIALISTA: {pontos}/4")
    print("="*70)

    if pontos >= 3:
        print("\n✅ ESPECIALISTA APROVADO - Pronto para ensemble")
        validacao['recomendacao'] = "APROVADO"
    else:
        print("\n⚠️  ESPECIALISTA PRECISA AJUSTES")
        validacao['recomendacao'] = "REVISAR"

    return validacao

# GERAÇÃO DE CSVs


def salvar_resultados(df_ranking, df_red_flags, validacao):


    print("\n" + "="*70)
    print("💾 SALVANDO RESULTADOS")
    print("="*70)

    df_ranking.to_csv('ranking_fundamentalista.csv', index=False)
    print("✅ ranking_fundamentalista.csv - Ranking completo com todos os scores")

    # 2. Sinais para ensemble
    sinais_ensemble = df_ranking[[
        'ticker', 'rank', 'percentile', 'qscore',
        'quality_score', 'growth_score', 'value_score', 'safety_score',
        'momentum_score', 'efficiency_score', 'lstm_score',
        'risk_score', 'categoria', 'trend', 'trend_strength',
        'roe', 'roic', 'net_margin', 'pe_ratio', 'debt_to_equity'
    ]].copy()

    sinais_ensemble.to_csv('sinais_ensemble.csv', index=False)
    print("✅ sinais_ensemble.csv - Sinais formatados para integração")


    if len(df_red_flags) > 0:
        df_red_flags.to_csv('red_flags_detalhados.csv', index=False)
        print(f"✅ red_flags_detalhados.csv - {len(df_red_flags)} alertas detectados")


    top_picks = df_ranking.head(20)[[
        'rank', 'ticker', 'qscore', 'categoria', 'trend',
        'roe', 'roic', 'pe_ratio', 'net_margin', 'revenue_growth_yoy',
        'risk_score', 'total_flags'
    ]].copy()

    top_picks.to_csv('top_picks.csv', index=False)
    print("✅ top_picks.csv - Top 20 empresas com detalhes")

    # 5. Relatório de validação
    with open('metricas_validacao.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RELATÓRIO DE VALIDAÇÃO DO ESPECIALISTA FUNDAMENTALISTA\n")
        f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
        f.write("="*70 + "\n\n")

        f.write(f"RESULTADO: {validacao.get('recomendacao', 'N/A')}\n\n")

        f.write("CRITÉRIOS:\n")
        f.write(f"  Separação Top/Bottom: {validacao.get('separacao', 'N/A')}\n")
        f.write(f"  Qualidade Top 10: {validacao.get('qualidade', 'N/A')}\n")
        f.write(f"  Consistência Scores: {validacao.get('consistencia', 'N/A')}\n")
        f.write(f"  Distribuição: {validacao.get('distribuicao', 'N/A')}\n\n")

        f.write("MÉTRICAS SECUNDÁRIAS:\n")
        f.write(f"  IC: {validacao.get('ic', 0):.4f}\n")
        f.write(f"  P-value: {validacao.get('pvalue', 1):.4f}\n")

    print("✅ metricas_validacao.txt - Relatório de qualidade")

    print(f"\n📊 Total de arquivos gerados: 5")


#  PIPELINE PRINCIPAL

def executar_especialista_fundamentalista(tickers, n_quarters=24, lookback=4, tentar_lstm=True):
    """
    Pipeline completo otimizado para ensemble
    """

    print("\n" + "="*80)
    print("🚀 ESPECIALISTA FUNDAMENTALISTA - VERSÃO ENSEMBLE")
    print("="*80)
    print(f"Empresas: {len(tickers)} | Trimestres: {n_quarters} | LSTM: {'Sim' if tentar_lstm else 'Não'}")

    # 1. COLETA
    coletor = ColetorDados()
    df_raw = coletor.coletar_multiplas_empresas(tickers, n_quarters)
    if df_raw is None:
        print("❌ Falha na coleta")
        return None

    # 2. PROCESSAMENTO
    print("\n🔧 Processando dados...")
    proc = ProcessadorDados()
    df = proc.calcular_crescimento(df_raw)
    df = proc.adicionar_features_tecnicas(df)
    df = proc.tratar_missing_data(df)
    df = proc.calcular_retorno_futuro(df, horizonte_meses=3)

    df_treino = df[df['retorno_futuro'].notna()].copy()
    print(f"✅ {len(df_treino)} registros processados")

    # 3. LSTM (OPCIONAL)
    usar_lstm = False
    modelo_lstm = None

    if tentar_lstm:
        modelo_lstm = ModeloLSTM()
        X, y, tickers_list, dates_list = modelo_lstm.preparar_dados(df_treino, lookback)

        print(f"\n📦 Sequências LSTM: {len(X)}")

        if len(X) >= 100:
            print("✅ Treinando LSTM...")

            indices = np.argsort(dates_list)
            X, y = X[indices], y[indices]

            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            modelo_lstm.treinar(X_train, y_train, X_val, y_val)
            usar_lstm = True
        else:
            print(f"⚠️  Insuficiente ({len(X)} < 100). Usando apenas fundamentos.")

    # 4. RANKING
    if usar_lstm:
        especialista = EspecialistaFundamentalista(
            modelo_lstm=modelo_lstm.model,
            scaler=modelo_lstm.scaler,
            feature_cols=modelo_lstm.feature_cols
        )
    else:
        especialista = EspecialistaFundamentalista()

    df_ranking, df_red_flags = especialista.calcular_ranking(df, lookback)

    # 5. VALIDAÇÃO
    validacao = validar_especialista(df_ranking, df)

    # 6. SALVAR CSVs
    salvar_resultados(df_ranking, df_red_flags, validacao)

    # 7. RESUMO
    print("\n" + "="*80)
    print("📊 RESUMO EXECUTIVO")
    print("="*80)

    print(f"\n🏆 TOP 5 EMPRESAS:")
    print("-" * 80)
    top5 = df_ranking[['rank', 'ticker', 'qscore', 'categoria', 'trend', 'roe', 'roic']].head(5)
    print(top5.to_string(index=False))

    print(f"\n⚠️  RED FLAGS:")
    print(f"   Total de alertas: {len(df_red_flags)}")
    if len(df_red_flags) > 0:
        flags_por_tipo = df_red_flags['tipo'].value_counts()
        print(f"   Críticos: {flags_por_tipo.get('CRITICO', 0)}")
        print(f"   Importantes: {flags_por_tipo.get('IMPORTANTE', 0)}")
        print(f"   Atenção: {flags_por_tipo.get('ATENCAO', 0)}")

    print(f"\n✅ Validação: {validacao.get('recomendacao', 'N/A')}")

    print("\n" + "="*80)
    print("🎯 PRÓXIMOS PASSOS")
    print("="*80)
    print("1. Integre 'sinais_ensemble.csv' no seu sistema multi-modelo")
    print("2. Use 'quality_score', 'growth_score', 'value_score' como features")
    print("3. Considere 'risk_score' e 'red_flags' nas decisões finais")
    print("4. 'trend' indica momento de entrada (MELHORANDO > ESTAVEL > PIORANDO)")

    return {
        'ranking': df_ranking,
        'red_flags': df_red_flags,
        'validacao': validacao,
        'modelo': modelo_lstm if usar_lstm else None,
        'usa_lstm': usar_lstm
    }


# EXECUÇÃO PRINCIPAL


if __name__ == "__main__":

    # Lista expandida de ações para maior confiabilidade
    TICKERS = [
        # Bancos (10)
        'ITUB4', 'BBDC4', 'BBAS3', 'SANB11', 'BPAC11', 'BIDI11', 'BRBI11', 'MODL11', 'BPAN4', 'BBSE3',

        # Energia/Petróleo (12)
        'PETR4', 'PETR3', 'VALE3', 'PRIO3', 'RRRP3', 'ENBR3', 'ENEV3', 'TAEE11', 'CMIG4', 'CPLE6', 'EQTL3', 'NEOE3',

        # Varejo (10)
        'MGLU3', 'LREN3', 'AMER3', 'PCAR3', 'RADL3', 'VIVA3', 'BHIA3', 'SOMA3', 'CRFB3', 'PETZ3',

        # Alimentos (6)
        'ABEV3', 'JBSS3', 'BEEF3', 'BRFS3', 'SMTO3', 'MRFG3',

        # Construção (8)
        'CYRE3', 'TEND3', 'CURY3', 'MRVE3', 'EZTC3', 'JHSF3', 'DIRR3', 'HBOR3',

        # Industrial (8)
        'WEGE3', 'RAIL3', 'EMBR3', 'LEVE3', 'TUPY3', 'RAPT4', 'TGMA3', 'LOGN3',

        # Materiais (8)
        'SUZB3', 'KLBN11', 'CSNA3', 'GGBR4', 'GOAU4', 'USIM5', 'RANI3', 'POMO4',

        # Tech/Telecom (7)
        'VIVT3', 'TIMS3', 'TOTVS3', 'LWSA3', 'POSI3', 'INTB3', 'NGRD3',
    ]

    # EXECUTAR
    print("\n⏱️  Tempo estimado: 10-15 minutos...")

    resultados = executar_especialista_fundamentalista(
        tickers=TICKERS,
        n_quarters=24,      # 6 anos de dados
        lookback=4,         # 1 ano de contexto
        tentar_lstm=True    # Tenta usar LSTM
    )

    if resultados:
        print("\n✅ SISTEMA EXECUTADO COM SUCESSO!")
        print("\n📂 Arquivos gerados:")
        print("   - ranking_fundamentalista.csv")
        print("   - sinais_ensemble.csv")
        print("   - red_flags_detalhados.csv")
        print("   - top_picks.csv")
        print("   - metricas_validacao.txt")
