import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.signal import argrelextrema

class AdvancedPatternDetector:
    """Detecção avançada de padrões gráficos com validações múltiplas"""
    
    def __init__(self, window=5, sensitivity=0.02):
        self.window = window
        self.sensitivity = sensitivity
        
    def detect_all_patterns(self, df):
        """Detecta todos os padrões gráficos"""
        print("🔍 Detectando padrões gráficos...")
        
        patterns = []
        
        # Encontrar pivots
        pivots = self.find_swing_pivots(df)
        
        # Detectar padrões específicos
        patterns.extend(self.detect_head_shoulders_advanced(df, pivots))
        patterns.extend(self.detect_double_tops_bottoms(df, pivots))
        patterns.extend(self.detect_triangles(df))
        patterns.extend(self.detect_flags_pennants(df))
        patterns.extend(self.detect_wedges(df))
        
        # Filtrar padrões válidos
        valid_patterns = [p for p in patterns if p['confidence'] > 0.6]
        
        print(f"✅ Padrões encontrados: {len(valid_patterns)}")
        return valid_patterns
    
    def find_swing_pivots(self, df, window=5):
        """Encontra pivots de swing (máximos e mínimos locais)"""
        highs = df['High'].values
        lows = df['Low'].values
        
        # Encontrar máximos e mínimos locais
        high_pivots = argrelextrema(highs, np.greater, order=window)[0]
        low_pivots = argrelextrema(lows, np.less, order=window)[0]
        
        resistance_levels = [(i, highs[i]) for i in high_pivots]
        support_levels = [(i, lows[i]) for i in low_pivots]
        
        return {
            'highs': sorted(resistance_levels, key=lambda x: x[0]),
            'lows': sorted(support_levels, key=lambda x: x[0])
        }
    
    def detect_head_shoulders_advanced(self, df, pivots):
        """Detecção melhorada de OCO com validação de volume"""
        patterns = []
        highs = pivots['highs']
        
        for i in range(2, len(highs) - 2):
            left_shoulder = highs[i-2]
            head = highs[i]
            right_shoulder = highs[i+2]
            
            # Validações de preço
            if not (head[1] > left_shoulder[1] * (1 + self.sensitivity) and 
                    head[1] > right_shoulder[1] * (1 + self.sensitivity) and
                    abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < self.sensitivity):
                continue
            
            # Validações temporais
            time_left_to_head = head[0] - left_shoulder[0]
            time_head_to_right = right_shoulder[0] - head[0]
            time_symmetry = abs(time_left_to_head - time_head_to_right) / max(time_left_to_head, time_head_to_right)
            
            if time_symmetry > 0.5:  # Muita assimetria temporal
                continue
            
            # Validação de volume
            volume_confirmation = self.check_head_shoulders_volume(df, left_shoulder[0], head[0], right_shoulder[0])
            
            # Calcular confiança
            confidence = self.calculate_hs_confidence(df, left_shoulder, head, right_shoulder, volume_confirmation)
            
            if confidence > 0.6:
                neckline = self.calculate_neckline(df, left_shoulder, right_shoulder)
                
                patterns.append({
                    'type': 'Ombro-Cabeça-Ombro (OCO)',
                    'direction': 'bearish',
                    'left_shoulder': left_shoulder,
                    'head': head,
                    'right_shoulder': right_shoulder,
                    'neckline': neckline,
                    'confidence': confidence,
                    'volume_confirmation': volume_confirmation,
                    'target_price': neckline - (head[1] - neckline)
                })
        
        return patterns
    
    def detect_double_tops_bottoms(self, df, pivots):
        """Detecta topos e fundos duplos/triplos"""
        patterns = []
        
        # Topos duplos
        highs = pivots['highs']
        for i in range(1, len(highs)):
            first_top = highs[i-1]
            second_top = highs[i]
            
            price_diff = abs(first_top[1] - second_top[1]) / first_top[1]
            time_diff = second_top[0] - first_top[0]
            
            if price_diff < 0.03 and time_diff > 10:
                volume_confirmation = self.check_double_pattern_volume(df, first_top[0], second_top[0])
                confidence = 0.7 if volume_confirmation else 0.5
                
                patterns.append({
                    'type': 'Topo Duplo',
                    'direction': 'bearish',
                    'first_top': first_top,
                    'second_top': second_top,
                    'confidence': confidence,
                    'volume_confirmation': volume_confirmation,
                    'target_price': min(df['Low'].iloc[first_top[0]:second_top[0]]) * 0.98
                })
        
        # Fundos duplos
        lows = pivots['lows']
        for i in range(1, len(lows)):
            first_bottom = lows[i-1]
            second_bottom = lows[i]
            
            price_diff = abs(first_bottom[1] - second_bottom[1]) / first_bottom[1]
            time_diff = second_bottom[0] - first_bottom[0]
            
            if price_diff < 0.03 and time_diff > 10:
                volume_confirmation = self.check_double_pattern_volume(df, first_bottom[0], second_bottom[0])
                confidence = 0.7 if volume_confirmation else 0.5
                
                patterns.append({
                    'type': 'Fundo Duplo',
                    'direction': 'bullish',
                    'first_bottom': first_bottom,
                    'second_bottom': second_bottom,
                    'confidence': confidence,
                    'volume_confirmation': volume_confirmation,
                    'target_price': max(df['High'].iloc[first_bottom[0]:second_bottom[0]]) * 1.02
                })
        
        return patterns
    
    def detect_triangles(self, df, window=20):
        """Detecta triângulos (simétrico, ascendente, descendente)"""
        patterns = []
        
        if len(df) < window * 2:
            return patterns
        
        recent = df.tail(window)
        highs = recent['High'].values
        lows = recent['Low'].values
        
        # Calcular tendências das linhas de suporte e resistência
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        
        threshold = 0.0005
        
        # Triângulo simétrico
        if high_slope < -threshold and low_slope > threshold:
            convergence_rate = abs(high_slope) + abs(low_slope)
            confidence = min(0.8, 0.5 + convergence_rate * 1000)
            
            patterns.append({
                'type': 'Triângulo Simétrico',
                'direction': 'neutral',
                'high_slope': high_slope,
                'low_slope': low_slope,
                'convergence': convergence_rate,
                'confidence': confidence
            })
        
        # Triângulo ascendente
        elif abs(high_slope) < threshold and low_slope > threshold:
            patterns.append({
                'type': 'Triângulo Ascendente',
                'direction': 'bullish',
                'high_slope': high_slope,
                'low_slope': low_slope,
                'confidence': 0.7
            })
        
        # Triângulo descendente
        elif high_slope < -threshold and abs(low_slope) < threshold:
            patterns.append({
                'type': 'Triângulo Descendente',
                'direction': 'bearish',
                'high_slope': high_slope,
                'low_slope': low_slope,
                'confidence': 0.7
            })
        
        return patterns
    
    def detect_flags_pennants(self, df):
        """Detecta padrões de bandeira e flâmula"""
        patterns = []
        # Implementação simplificada
        return patterns
    
    def detect_wedges(self, df):
        """Detecta cunhas (ascendentes e descendentes)"""
        patterns = []
        # Implementação simplificada
        return patterns
    
    # Métodos auxiliares de validação
    def check_head_shoulders_volume(self, df, left_idx, head_idx, right_idx):
        """Verifica padrão de volume típico de OCO"""
        left_volume = df['Volume'].iloc[left_idx]
        head_volume = df['Volume'].iloc[head_idx]
        right_volume = df['Volume'].iloc[right_idx]
        
        # Volume na cabeça deve ser o mais alto
        return head_volume > left_volume and head_volume > right_volume
    
    def check_double_pattern_volume(self, df, first_idx, second_idx):
        """Verifica volume em padrões duplos"""
        first_volume = df['Volume'].iloc[first_idx]
        second_volume = df['Volume'].iloc[second_idx]
        avg_volume = df['Volume'].rolling(20).mean().iloc[second_idx]
        
        # Segundo pico deve ter volume menor
        return second_volume < first_volume and second_volume < avg_volume
    
    def calculate_hs_confidence(self, df, left_shoulder, head, right_shoulder, volume_confirm):
        """Calcula confiança do padrão OCO"""
        confidence = 0.5  # Base
        
        # Fator de volume
        if volume_confirm:
            confidence += 0.2
        
        # Fator de simetria temporal
        time_left_to_head = head[0] - left_shoulder[0]
        time_head_to_right = right_shoulder[0] - head[0]
        time_symmetry = 1 - abs(time_left_to_head - time_head_to_right) / max(time_left_to_head, time_head_to_right)
        confidence += time_symmetry * 0.15
        
        # Fator de magnitude
        price_ratio_left = head[1] / left_shoulder[1]
        price_ratio_right = head[1] / right_shoulder[1]
        if price_ratio_left > 1.02 and price_ratio_right > 1.02:
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    def calculate_neckline(self, df, left_shoulder, right_shoulder):
        """Calcula linha de pescoço para OCO"""
        # Encontra o mínimo entre os ombros
        start_idx = left_shoulder[0]
        end_idx = right_shoulder[0]
        neckline_low = df['Low'].iloc[start_idx:end_idx].min()
        return neckline_low
    
    def cluster_support_resistance(self, levels, threshold=0.02):
        """Agrupa níveis de suporte/resistência próximos"""
        if not levels:
            return []
        
        prices = sorted([price for _, price in levels])
        clusters = []
        current_cluster = [prices[0]]
        
        for price in prices[1:]:
            if abs(price - np.mean(current_cluster)) / np.mean(current_cluster) < threshold:
                current_cluster.append(price)
            else:
                clusters.append({
                    'price': np.mean(current_cluster),
                    'strength': len(current_cluster),
                    'touches': current_cluster
                })
                current_cluster = [price]
        
        if current_cluster:
            clusters.append({
                'price': np.mean(current_cluster),
                'strength': len(current_cluster),
                'touches': current_cluster
            })
        
        return sorted(clusters, key=lambda x: x['strength'], reverse=True)