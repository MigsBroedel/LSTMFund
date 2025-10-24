import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Importações dos módulos
from data_collector import DataCollector
from feature_engineer import AdvancedFeatureEngineer
from pattern_detector import AdvancedPatternDetector
from model_hybrid import HybridPatternModel, PatternLabeler
from risk_manager import RiskManager, PortfolioManager
from backtester import RobustBacktester
from visualizer import AdvancedVisualizer

class AdvancedStockAnalyzer:
    """Sistema avançado de análise técnica com IA"""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.pattern_detector = AdvancedPatternDetector()
        self.risk_manager = RiskManager()
        self.portfolio_manager = PortfolioManager()
        self.backtester = RobustBacktester()
        self.visualizer = AdvancedVisualizer(style='dark')
        
    def analyze_symbol(self, symbol, period='1y', train_model=False):
        """Analisa um símbolo específico"""
        print(f"🎯 Iniciando análise para {symbol}")
        print("=" * 60)
        
        # 1. Coletar dados
        df = self.data_collector.fetch_real_data(symbol, period=period)
        if df is None:
            print(f"❌ Falha ao baixar dados para {symbol}")
            return None
        
        # 2. Engenharia de features
        df = self.feature_engineer.add_advanced_features(df)
        
        # 3. Detecção de padrões clássicos
        patterns = self.pattern_detector.detect_all_patterns(df)
        
        # 4. Suportes e resistências
        pivots = self.pattern_detector.find_swing_pivots(df)
        support_levels = self.pattern_detector.cluster_support_resistance(pivots['lows'])
        resistance_levels = self.pattern_detector.cluster_support_resistance(pivots['highs'])
        
        # 5. Preparar resultados
        results = {
            'symbol': symbol,
            'data': df,
            'patterns': patterns,
            'support_levels': [s['price'] for s in support_levels][:5],
            'resistance_levels': [r['price'] for r in resistance_levels][:5],
            'indicators': {
                'RSI': df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else 0,
                'MACD': df['MACD'].iloc[-1] if 'MACD' in df.columns else 0,
                'Volume': df['Volume'].iloc[-1],
                'Volatility': df['Volatility_ATR'].iloc[-1] if 'Volatility_ATR' in df.columns else 0
            }
        }
        
        # 6. Modelo de ML (opcional)
        if train_model and len(df) > 100:
            print("\n🤖 Treinando modelo de machine learning...")
            model = HybridPatternModel()
            try:
                X, y = model.prepare_data(df)
                model.build_model(X.shape[2], num_classes=3)
                history = model.train(X, y, epochs=50)
                
                # Fazer predições
                predictions = model.predict(X)
                results['ml_predictions'] = predictions
                results['ml_model'] = model
                
                print("✅ Modelo de ML treinado com sucesso!")
            except Exception as e:
                print(f"❌ Erro no treinamento do modelo: {e}")
        
        return results
    
    def generate_report(self, results):
        """Gera relatório completo da análise"""
        if not results:
            return "❌ Nenhum resultado para gerar relatório"
        
        report = []
        report.append("=" * 80)
        report.append(f"RELATÓRIO DE ANÁLISE TÉCNICA - {results['symbol']}")
        report.append("=" * 80)
        report.append(f"Período analisado: {len(results['data'])} candles")
        report.append(f"Preço atual: R$ {results['data']['Close'].iloc[-1]:.2f}")
        report.append("")
        
        # Padrões identificados
        report.append("📊 PADRÕES IDENTIFICADOS:")
        report.append("-" * 40)
        if results['patterns']:
            for i, pattern in enumerate(results['patterns'][:5], 1):
                report.append(f"{i}. {pattern['type']}")
                report.append(f"   Direção: {pattern['direction'].upper()}")
                report.append(f"   Confiança: {pattern['confidence']:.1%}")
                report.append("")
        else:
            report.append("Nenhum padrão clássico identificado")
        report.append("")
        
        # Suportes e resistências
        report.append("📍 NÍVEIS CHAVE:")
        report.append("-" * 40)
        report.append("Suportes:")
        for i, level in enumerate(results['support_levels'][:3], 1):
            distance = ((results['data']['Close'].iloc[-1] - level) / results['data']['Close'].iloc[-1]) * 100
            report.append(f"   S{i}: R$ {level:.2f} ({distance:+.1f}%)")
        
        report.append("\nResistências:")
        for i, level in enumerate(results['resistance_levels'][:3], 1):
            distance = ((level - results['data']['Close'].iloc[-1]) / results['data']['Close'].iloc[-1]) * 100
            report.append(f"   R{i}: R$ {level:.2f} ({distance:+.1f}%)")
        report.append("")
        
        # Indicadores
        report.append("📈 INDICADORES TÉCNICOS:")
        report.append("-" * 40)
        indicators = results['indicators']
        report.append(f"RSI: {indicators['RSI']:.2f}")
        report.append(f"MACD: {indicators['MACD']:.4f}")
        report.append(f"Volatilidade (ATR): {indicators['Volatility']:.4f}")
        report.append(f"Volume: {indicators['Volume']:,.0f}")
        
        # Recomendações
        report.append("\n💡 RECOMENDAÇÕES:")
        report.append("-" * 40)
        current_price = results['data']['Close'].iloc[-1]
        rsi = indicators['RSI']
        
        if rsi > 70:
            report.append("⚠️  RSI em sobrecompra - considere tomar lucros")
        elif rsi < 30:
            report.append("⚠️  RSI em sobrevenda - possível oportunidade de compra")
        else:
            report.append("📊 RSI em zona neutra - aguardar confirmação")
        
        if results['patterns']:
            bullish_patterns = sum(1 for p in results['patterns'] if p['direction'] == 'bullish')
            bearish_patterns = sum(1 for p in results['patterns'] if p['direction'] == 'bearish')
            
            if bullish_patterns > bearish_patterns:
                report.append("📈 Viés positivo - mais padrões de alta identificados")
            elif bearish_patterns > bullish_patterns:
                report.append("📉 Viés negativo - mais padrões de baixa identificados")
        
        report.append("\n" + "=" * 80)
        report.append("AVISO: Esta análise é informativa e não constitui")
        report.append("recomendação de investimento. Consulte um profissional.")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_complete_analysis(self, symbol, period='1y', train_model=False):
        """Executa análise completa"""
        print(f"🚀 INICIANDO ANÁLISE COMPLETA PARA {symbol}")
        print("=" * 60)
        
        # Análise técnica
        results = self.analyze_symbol(symbol, period, train_model)
        
        if results is None:
            return
        
        # Gerar relatório
        report = self.generate_report(results)
        print("\n" + report)
        
        # Visualização
        print("\n📊 Gerando visualizações...")
        self.visualizer.plot_analysis(results['data'], results, 
                                    title=f"Análise Técnica - {symbol}")
        
        # Backtest (se modelo foi treinado)
        if train_model and 'ml_predictions' in results:
            print("\n🔍 Executando backtest...")
            signals = pd.Series(results['ml_predictions'], index=results['data'].index)
            backtest_results = self.backtester.run_backtest(results['data'], signals)
            backtest_report = self.backtester.generate_report(backtest_results)
            print(backtest_report)
        
        return results

# Exemplo de uso
if __name__ == "__main__":
    # Inicializar o sistema
    analyzer = AdvancedStockAnalyzer()
    
    # Analisar um símbolo específico
    symbol = "PETR4.SA"  # Você pode mudar para qualquer símbolo do Yahoo Finance
    
    try:
        results = analyzer.run_complete_analysis(
            symbol=symbol,
            period='1y',
            train_model=True  # Mudar para True se quiser treinar o modelo (demora mais)
        )
        
        if results:
            print(f"\n✅ Análise concluída para {symbol}!")
            print(f"📈 Padrões encontrados: {len(results['patterns'])}")
            print(f"📍 Suportes/Resistências: {len(results['support_levels'])}/{len(results['resistance_levels'])}")
        else:
            print(f"❌ Falha na análise de {symbol}")
            
    except Exception as e:
        print(f"❌ Erro durante a análise: {e}")
        print("💡 Dica: Verifique se o símbolo está correto e se há conexão com a internet")