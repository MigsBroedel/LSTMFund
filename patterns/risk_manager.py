import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class RiskManager:
    """Sistema avançado de gerenciamento de risco"""
    
    def __init__(self, initial_capital=10000, max_position_size=0.1, max_daily_loss=0.02):
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.portfolio_value = initial_capital
        
    def calculate_position_size(self, current_price, confidence, volatility, portfolio_value=None):
        """Calcula tamanho da posição baseado em risco"""
        if portfolio_value is None:
            portfolio_value = self.portfolio_value
            
        # Tamanho base
        base_size = portfolio_value * self.max_position_size
        
        # Ajuste por confiança
        confidence_adj = base_size * confidence
        
        # Ajuste por volatilidade (reduz posição em alta volatilidade)
        volatility_adj = confidence_adj * (0.02 / max(volatility, 0.02))
        
        # Limites
        max_position = portfolio_value * 0.25  # Máximo 25% do portfólio
        min_position = portfolio_value * 0.01  # Mínimo 1% do portfólio
        
        position_value = max(min(volatility_adj, max_position), min_position)
        shares = int(position_value / current_price)
        
        return max(shares, 1)  # Pelo menos 1 ação
    
    def validate_signal(self, signal, current_volatility, market_regime, confidence):
        """Valida se sinal deve ser executado baseado em condições de mercado"""
        
        # Filtro de volatilidade
        if current_volatility > 0.05:  # Alta volatilidade (>5%)
            print("⚠️  Alta volatilidade - reduzindo exposição")
            return signal * 0.5  # Reduz exposição pela metade
        
        # Filtro de regime de mercado
        if market_regime == 'high_volatility':
            print("⚠️  Regime de alta volatilidade - sinal ignorado")
            return 0  # Não opera em alta volatilidade
        
        # Filtro de confiança
        if confidence < 0.6:
            print("⚠️  Baixa confiança - sinal ignorado")
            return 0
        
        # Filtro de força do sinal
        if abs(signal) < 0.3:
            print("⚠️  Sinal fraco - ignorado")
            return 0
        
        return signal
    
    def calculate_stop_loss(self, entry_price, signal_type, volatility):
        """Calcula níveis de stop-loss dinâmicos"""
        if signal_type == 1:  # COMPRA
            stop_loss = entry_price * (1 - max(volatility * 1.5, 0.02))
            take_profit = entry_price * (1 + max(volatility * 2, 0.04))
        else:  # VENDA
            stop_loss = entry_price * (1 + max(volatility * 1.5, 0.02))
            take_profit = entry_price * (1 - max(volatility * 2, 0.04))
        
        return stop_loss, take_profit
    
    def update_portfolio_value(self, new_value):
        """Atualiza valor do portfólio"""
        self.portfolio_value = new_value
    
    def calculate_var(self, returns, confidence_level=0.95):
        """Calcula Value at Risk (VaR) histórico"""
        if len(returns) < 30:
            return 0.05  # Valor padrão se dados insuficientes
            
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return abs(var)
    
    def check_drawdown(self, current_value, peak_value):
        """Verifica drawdown atual"""
        if peak_value == 0:
            return 0
        drawdown = (peak_value - current_value) / peak_value
        return drawdown

class PortfolioManager:
    """Gerenciador de portfólio"""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
    def execute_trade(self, symbol, action, shares, price, timestamp, confidence):
        """Executa uma trade"""
        cost = shares * price
        commission = cost * 0.0025  # 0.25% de comissão
        total_cost = cost + commission
        
        if action == 'BUY':
            if self.cash >= total_cost:
                self.cash -= total_cost
                if symbol in self.positions:
                    self.positions[symbol]['shares'] += shares
                    self.positions[symbol]['avg_price'] = (
                        (self.positions[symbol]['avg_price'] * self.positions[symbol]['shares'] + cost) / 
                        (self.positions[symbol]['shares'] + shares)
                    )
                else:
                    self.positions[symbol] = {
                        'shares': shares,
                        'avg_price': price,
                        'entry_date': timestamp
                    }
                
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'total_cost': total_cost,
                    'confidence': confidence
                }
                self.trades.append(trade)
                return True
            else:
                print(f"❌ Capital insuficiente para compra: {self.cash:.2f} < {total_cost:.2f}")
                return False
                
        elif action == 'SELL':
            if symbol in self.positions and self.positions[symbol]['shares'] >= shares:
                self.cash += (shares * price) - commission
                self.positions[symbol]['shares'] -= shares
                
                if self.positions[symbol]['shares'] == 0:
                    del self.positions[symbol]
                
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'total_cost': total_cost,
                    'confidence': confidence
                }
                self.trades.append(trade)
                return True
            else:
                print(f"❌ Posição insuficiente para venda")
                return False
    
    def get_portfolio_value(self, current_prices):
        """Calcula valor total do portfólio"""
        stock_value = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                stock_value += position['shares'] * current_prices[symbol]
        
        total_value = self.cash + stock_value
        self.portfolio_history.append(total_value)
        return total_value
    
    def get_portfolio_summary(self, current_prices):
        """Retorna resumo do portfólio"""
        total_value = self.get_portfolio_value(current_prices)
        profit_loss = total_value - self.initial_capital
        profit_loss_pct = (profit_loss / self.initial_capital) * 100
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'stock_value': total_value - self.cash,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'positions_count': len(self.positions),
            'trades_count': len(self.trades)
        }

