import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class RobustBacktester:
    """Backtesting robusto com custos de transação e slippage"""
    
    def __init__(self, initial_capital=10000, slippage=0.001, commission=0.0025):
        self.initial_capital = initial_capital
        self.slippage = slippage
        self.commission = commission
        
    def run_backtest(self, df, signals, prices='Close'):
        """Executa backtest completo"""
        print("🔄 Iniciando backtest...")
        
        portfolio = {
            'cash': self.initial_capital, 
            'shares': 0, 
            'total': self.initial_capital,
            'trades': []
        }
        
        for i in range(len(df)):
            if i >= len(signals):
                break
                
            current_price = df[prices].iloc[i]
            signal = signals.iloc[i] if hasattr(signals, 'iloc') else signals[i]
            
            # Aplicar slippage
            buy_price = current_price * (1 + self.slippage)
            sell_price = current_price * (1 - self.slippage)
            
            # Executar sinal de COMPRA
            if signal == 2 and portfolio['cash'] > 0:  # COMPRA
                shares_to_buy = portfolio['cash'] // (buy_price * (1 + self.commission))
                if shares_to_buy > 0:
                    cost = shares_to_buy * buy_price * (1 + self.commission)
                    portfolio['cash'] -= cost
                    portfolio['shares'] += shares_to_buy
                    
                    portfolio['trades'].append({
                        'date': df.index[i],
                        'action': 'BUY',
                        'price': buy_price,
                        'shares': shares_to_buy,
                        'cost': cost,
                        'portfolio_value': portfolio['cash'] + portfolio['shares'] * current_price
                    })
                    
            # Executar sinal de VENDA
            elif signal == 1 and portfolio['shares'] > 0:  # VENDA
                sell_revenue = portfolio['shares'] * sell_price * (1 - self.commission)
                portfolio['cash'] += sell_revenue
                
                portfolio['trades'].append({
                    'date': df.index[i],
                    'action': 'SELL',
                    'price': sell_price,
                    'shares': portfolio['shares'],
                    'revenue': sell_revenue,
                    'portfolio_value': portfolio['cash']
                })
                
                portfolio['shares'] = 0
            
            # Calcular valor total do portfólio
            portfolio['total'] = portfolio['cash'] + portfolio['shares'] * current_price
        
        results = self.calculate_metrics(portfolio, df, prices)
        print("✅ Backtest concluído!")
        return results
    
    def calculate_metrics(self, portfolio, df, prices='Close'):
        """Calcula métricas de performance"""
        initial_value = self.initial_capital
        final_value = portfolio['total']
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Calcular drawdown
        portfolio_values = []
        for i in range(len(df)):
            if i >= len(portfolio['trades']):
                break
            portfolio_values.append(portfolio['trades'][i]['portfolio_value'])
        
        if not portfolio_values:
            portfolio_values = [initial_value] * len(df)
        
        peak = initial_value
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Estatísticas de trades
        trades = portfolio['trades']
        winning_trades = 0
        total_trades = len(trades)
        
        if total_trades > 0:
            for i in range(0, len(trades)-1, 2):
                if i+1 < len(trades):
                    buy_trade = trades[i]
                    sell_trade = trades[i+1]
                    if sell_trade['revenue'] > buy_trade['cost']:
                        winning_trades += 1
            
            win_rate = (winning_trades / (total_trades // 2)) * 100 if total_trades >= 2 else 0
        else:
            win_rate = 0
        
        return {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'trades': trades
        }
    
    def generate_report(self, results):
        """Gera relatório detalhado do backtest"""
        report = []
        report.append("=" * 80)
        report.append("RELATÓRIO DE BACKTESTING")
        report.append("=" * 80)
        report.append(f"Capital Inicial: R$ {results['initial_capital']:,.2f}")
        report.append(f"Capital Final: R$ {results['final_value']:,.2f}")
        report.append(f"Retorno Total: {results['total_return']:.2f}%")
        report.append(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        report.append(f"Total de Trades: {results['total_trades']}")
        report.append(f"Win Rate: {results['win_rate']:.2f}%")
        report.append("=" * 80)
        
        # Detalhes dos trades
        if results['trades']:
            report.append("\nÚLTIMOS 5 TRADES:")
            for trade in results['trades'][-5:]:
                report.append(f"{trade['date'].strftime('%Y-%m-%d')} - {trade['action']} - {trade['shares']} ações a R$ {trade['price']:.2f}")
        
        return "\n".join(report)