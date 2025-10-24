import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec

class AdvancedVisualizer:
    """Visualizador avançado para análise técnica"""
    
    def __init__(self, style='dark'):
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Configura estilo visual"""
        if self.style == 'dark':
            plt.style.use('dark_background')
            self.colors = {
                'bullish': '#00ff00',
                'bearish': '#ff0000',
                'neutral': '#ffff00',
                'support': '#00ffff',
                'resistance': '#ff00ff',
                'pattern': '#ff8c00'
            }
        elif self.style == 'light':
            plt.style.use('seaborn-v0_8-whitegrid')
            self.colors = {
                'bullish': '#2ca02c',
                'bearish': '#d62728',
                'neutral': '#ff7f0e',
                'support': '#1f77b4',
                'resistance': '#e377c2',
                'pattern': '#8c564b'
            }
        else:
            plt.style.use('classic')
            self.colors = {
                'bullish': 'green',
                'bearish': 'red',
                'neutral': 'blue',
                'support': 'cyan',
                'resistance': 'magenta',
                'pattern': 'orange'
            }
    
    def plot_analysis(self, df, results, title="Análise Técnica Completa"):
        """Plota análise técnica completa"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Gráfico principal
        ax_main = fig.add_subplot(gs[0:2, :])
        self._plot_main_chart(ax_main, df, results, title)
        
        # Indicadores
        ax_rsi = fig.add_subplot(gs[2, 0])
        self._plot_rsi(ax_rsi, df)
        
        ax_macd = fig.add_subplot(gs[2, 1])
        self._plot_macd(ax_macd, df)
        
        ax_volume = fig.add_subplot(gs[2, 2])
        self._plot_volume(ax_volume, df)
        
        # Padrões
        ax_patterns = fig.add_subplot(gs[3, 0])
        self._plot_patterns(ax_patterns, results)
        
        # Suportes e resistências
        ax_sr = fig.add_subplot(gs[3, 1])
        self._plot_support_resistance(ax_sr, results)
        
        # Performance
        ax_perf = fig.add_subplot(gs[3, 2])
        self._plot_performance(ax_perf, results)
        
        plt.tight_layout()
        plt.savefig('analise_tecnica.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_main_chart(self, ax, df, results, title):
        """Plota gráfico principal com preços e padrões"""
        # Plot candlestick simplificado
        for i in range(len(df)):
            color = self.colors['bullish'] if df['Close'].iloc[i] >= df['Open'].iloc[i] else self.colors['bearish']
            ax.plot([i, i], [df['Low'].iloc[i], df['High'].iloc[i]], color=color, linewidth=1)
            ax.plot([i, i], [df['Open'].iloc[i], df['Close'].iloc[i]], color=color, linewidth=3)
        
        # Plot médias móveis
        if 'SMA_20' in df.columns:
            ax.plot(df['SMA_20'].values, label='SMA 20', color='cyan', linewidth=1)
        if 'SMA_50' in df.columns:
            ax.plot(df['SMA_50'].values, label='SMA 50', color='magenta', linewidth=1)
        
        # Plot suportes e resistências
        for level in results.get('support_levels', [])[:3]:
            ax.axhline(y=level, color=self.colors['support'], linestyle='--', alpha=0.7)
        
        for level in results.get('resistance_levels', [])[:3]:
            ax.axhline(y=level, color=self.colors['resistance'], linestyle='--', alpha=0.7)
        
        # Marcar padrões
        for pattern in results.get('patterns', [])[:5]:
            self._plot_pattern_markers(ax, pattern, df)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_rsi(self, ax, df):
        """Plota RSI"""
        if 'RSI_14' in df.columns:
            ax.plot(df['RSI_14'].values, color='purple', linewidth=1)
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.7)
            ax.axhline(y=30, color='green', linestyle='--', alpha=0.7)
            ax.set_ylim(0, 100)
            ax.set_title('RSI', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    def _plot_macd(self, ax, df):
        """Plota MACD"""
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            ax.plot(df['MACD'].values, color='blue', linewidth=1, label='MACD')
            ax.plot(df['MACD_Signal'].values, color='red', linewidth=1, label='Signal')
            ax.bar(range(len(df)), df['MACD_Hist'].values, color=np.where(df['MACD_Hist'] > 0, 'green', 'red'), alpha=0.3)
            ax.set_title('MACD', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    def _plot_volume(self, ax, df):
        """Plota volume"""
        colors = [self.colors['bullish'] if df['Close'].iloc[i] >= df['Open'].iloc[i] else self.colors['bearish'] for i in range(len(df))]
        ax.bar(range(len(df)), df['Volume'].values, color=colors, alpha=0.7)
        ax.set_title('Volume', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_patterns(self, ax, results):
        """Plota distribuição de padrões"""
        patterns = results.get('patterns', [])
        if not patterns:
            ax.text(0.5, 0.5, 'Nenhum padrão\nidentificado', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Padrões Identificados', fontsize=10)
            return
        
        pattern_types = {}
        for p in patterns:
            ptype = p['type']
            pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
        
        ax.pie(pattern_types.values(), labels=pattern_types.keys(), autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribuição de Padrões', fontsize=10)
    
    def _plot_support_resistance(self, ax, results):
        """Plota força de suportes e resistências"""
        supports = results.get('support_levels', [])
        resistances = results.get('resistance_levels', [])
        
        levels = []
        strengths = []
        colors = []
        
        for s in supports[:5]:
            levels.append(s)
            strengths.append(1)
            colors.append(self.colors['support'])
        
        for r in resistances[:5]:
            levels.append(r)
            strengths.append(1)
            colors.append(self.colors['resistance'])
        
        if levels:
            ax.barh(range(len(levels)), strengths, color=colors)
            ax.set_yticks(range(len(levels)))
            ax.set_yticklabels([f'{level:.2f}' for level in levels], fontsize=8)
            ax.set_title('Suportes/Resistências', fontsize=10)
    
    def _plot_performance(self, ax, results):
        """Plota performance dos padrões"""
        # Esta função pode ser expandida para mostrar performance histórica dos padrões
        ax.text(0.5, 0.5, 'Análise de Performance\n(Disponível com mais dados)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Performance', fontsize=10)
    
    def _plot_pattern_markers(self, ax, pattern, df):
        """Marca padrões específicos no gráfico"""
        ptype = pattern['type']
        
        if ptype == 'Ombro-Cabeça-Ombro (OCO)':
            left_idx, left_price = pattern['left_shoulder']
            head_idx, head_price = pattern['head']
            right_idx, right_price = pattern['right_shoulder']
            
            ax.plot([left_idx, head_idx, right_idx], [left_price, head_price, right_price], 
                   'o-', color='yellow', markersize=8, linewidth=2, label='OCO')
        
        elif 'Topo Duplo' in ptype:
            idx1, price1 = pattern['first_top']
            idx2, price2 = pattern['second_top']
            ax.plot([idx1, idx2], [price1, price2], 'ro-', markersize=6, linewidth=2, label='Topo Duplo')
        
        elif 'Fundo Duplo' in ptype:
            idx1, price1 = pattern['first_bottom']
            idx2, price2 = pattern['second_bottom']
            ax.plot([idx1, idx2], [price1, price2], 'go-', markersize=6, linewidth=2, label='Fundo Duplo')