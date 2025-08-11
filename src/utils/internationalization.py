"""
Internationalization (i18n) support for the liquid neural framework.

Supports multiple languages for error messages, documentation, and user interfaces.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any
import warnings


class I18nManager:
    """Internationalization manager for multi-language support."""
    
    def __init__(self, default_language: str = "en", locale_dir: str = "locales"):
        self.default_language = default_language
        self.current_language = default_language
        self.locale_dir = Path(locale_dir)
        self.locale_dir.mkdir(exist_ok=True)
        
        self.translations = {}
        self.supported_languages = ["en", "es", "fr", "de", "ja", "zh"]
        
        # Initialize default translations
        self._initialize_default_translations()
        self._load_translations()
    
    def _initialize_default_translations(self):
        """Initialize default English translations."""
        default_translations = {
            "en": {
                "errors": {
                    "jax_not_available": "JAX is required but not installed. Install with: pip install jax jaxlib",
                    "invalid_input_size": "Input size must be positive integer",
                    "invalid_hidden_size": "Hidden size must be positive integer", 
                    "invalid_output_size": "Output size must be positive integer",
                    "invalid_tau": "Time constant (tau) must be positive",
                    "dimension_mismatch": "Input dimensions do not match expected size",
                    "batch_size_mismatch": "Batch size mismatch between inputs and hidden state",
                    "negative_dt": "Time step (dt) must be positive",
                    "numerical_instability": "Numerical instability detected",
                    "unknown_solver": "Unknown ODE solver specified",
                    "unknown_activation": "Unknown activation function, defaulting to tanh",
                    "invalid_threshold": "Threshold voltage must be greater than resting potential",
                    "memory_limit_exceeded": "Input exceeds memory safety limits",
                    "security_violation": "Security validation failed",
                    "cache_error": "Cache operation failed",
                    "model_loading_failed": "Failed to load model checkpoint"
                },
                "messages": {
                    "model_initialized": "Model initialized successfully",
                    "training_started": "Training started", 
                    "epoch_completed": "Epoch completed",
                    "validation_completed": "Validation completed",
                    "checkpoint_saved": "Checkpoint saved",
                    "optimization_enabled": "Performance optimizations enabled",
                    "cache_cleared": "Cache cleared successfully",
                    "security_check_passed": "Security validation passed",
                    "benchmark_completed": "Benchmark completed",
                    "experiment_finished": "Experiment finished"
                },
                "model_names": {
                    "liquid_neural_network": "Liquid Neural Network",
                    "continuous_time_rnn": "Continuous-Time RNN",
                    "adaptive_neuron": "Adaptive Neuron",
                    "meta_adaptive_network": "Meta-Adaptive Liquid Network",
                    "multiscale_network": "Multi-Scale Temporal Network",
                    "quantum_inspired_network": "Quantum-Inspired Network"
                },
                "metrics": {
                    "accuracy": "Accuracy",
                    "loss": "Loss", 
                    "mse": "Mean Squared Error",
                    "mae": "Mean Absolute Error",
                    "correlation": "Correlation",
                    "stability": "Stability Measure",
                    "memory_capacity": "Memory Capacity",
                    "processing_time": "Processing Time",
                    "spectral_radius": "Spectral Radius"
                }
            },
            "es": {
                "errors": {
                    "jax_not_available": "JAX es requerido pero no está instalado. Instalar con: pip install jax jaxlib",
                    "invalid_input_size": "El tamaño de entrada debe ser un entero positivo",
                    "invalid_hidden_size": "El tamaño oculto debe ser un entero positivo",
                    "invalid_output_size": "El tamaño de salida debe ser un entero positivo",
                    "invalid_tau": "La constante de tiempo (tau) debe ser positiva",
                    "dimension_mismatch": "Las dimensiones de entrada no coinciden con el tamaño esperado",
                    "batch_size_mismatch": "Discrepancia de tamaño de lote entre entradas y estado oculto",
                    "negative_dt": "El paso de tiempo (dt) debe ser positivo",
                    "numerical_instability": "Inestabilidad numérica detectada",
                    "unknown_solver": "Solucionador de EDO desconocido especificado",
                    "unknown_activation": "Función de activación desconocida, usando tanh por defecto"
                },
                "messages": {
                    "model_initialized": "Modelo inicializado exitosamente",
                    "training_started": "Entrenamiento iniciado",
                    "epoch_completed": "Época completada",
                    "validation_completed": "Validación completada",
                    "checkpoint_saved": "Punto de control guardado"
                },
                "model_names": {
                    "liquid_neural_network": "Red Neuronal Líquida",
                    "continuous_time_rnn": "RNN de Tiempo Continuo",
                    "adaptive_neuron": "Neurona Adaptiva"
                }
            },
            "fr": {
                "errors": {
                    "jax_not_available": "JAX est requis mais n'est pas installé. Installer avec: pip install jax jaxlib",
                    "invalid_input_size": "La taille d'entrée doit être un entier positif",
                    "invalid_hidden_size": "La taille cachée doit être un entier positif",
                    "invalid_output_size": "La taille de sortie doit être un entier positif",
                    "invalid_tau": "La constante de temps (tau) doit être positive",
                    "dimension_mismatch": "Les dimensions d'entrée ne correspondent pas à la taille attendue",
                    "numerical_instability": "Instabilité numérique détectée"
                },
                "messages": {
                    "model_initialized": "Modèle initialisé avec succès",
                    "training_started": "Entraînement commencé",
                    "epoch_completed": "Époque terminée"
                },
                "model_names": {
                    "liquid_neural_network": "Réseau de Neurones Liquides",
                    "continuous_time_rnn": "RNN à Temps Continu",
                    "adaptive_neuron": "Neurone Adaptatif"
                }
            },
            "de": {
                "errors": {
                    "jax_not_available": "JAX ist erforderlich, aber nicht installiert. Installieren mit: pip install jax jaxlib",
                    "invalid_input_size": "Eingabegröße muss eine positive Ganzzahl sein",
                    "invalid_hidden_size": "Versteckte Größe muss eine positive Ganzzahl sein",
                    "numerical_instability": "Numerische Instabilität erkannt"
                },
                "messages": {
                    "model_initialized": "Modell erfolgreich initialisiert",
                    "training_started": "Training gestartet"
                },
                "model_names": {
                    "liquid_neural_network": "Flüssiges Neuronales Netzwerk",
                    "continuous_time_rnn": "Kontinuierliche-Zeit RNN"
                }
            },
            "ja": {
                "errors": {
                    "jax_not_available": "JAXが必要ですがインストールされていません。インストール: pip install jax jaxlib",
                    "invalid_input_size": "入力サイズは正の整数である必要があります",
                    "numerical_instability": "数値不安定性が検出されました"
                },
                "messages": {
                    "model_initialized": "モデルが正常に初期化されました",
                    "training_started": "トレーニングが開始されました"
                },
                "model_names": {
                    "liquid_neural_network": "液体ニューラルネットワーク",
                    "continuous_time_rnn": "連続時間RNN"
                }
            },
            "zh": {
                "errors": {
                    "jax_not_available": "需要JAX但未安装。安装命令: pip install jax jaxlib",
                    "invalid_input_size": "输入大小必须是正整数",
                    "numerical_instability": "检测到数值不稳定性"
                },
                "messages": {
                    "model_initialized": "模型初始化成功",
                    "training_started": "训练开始"
                },
                "model_names": {
                    "liquid_neural_network": "液体神经网络",
                    "continuous_time_rnn": "连续时间RNN"
                }
            }
        }
        
        self.translations = default_translations
        
        # Save default translations to files
        for lang, translations in default_translations.items():
            lang_file = self.locale_dir / f"{lang}.json"
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
    
    def _load_translations(self):
        """Load translations from locale files."""
        for lang in self.supported_languages:
            lang_file = self.locale_dir / f"{lang}.json"
            if lang_file.exists():
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        self.translations[lang] = json.load(f)
                except Exception as e:
                    warnings.warn(f"Failed to load {lang} translations: {e}")
    
    def set_language(self, language: str):
        """Set the current language."""
        if language in self.supported_languages:
            self.current_language = language
        else:
            warnings.warn(f"Unsupported language '{language}', using {self.default_language}")
            self.current_language = self.default_language
    
    def get_text(self, category: str, key: str, **kwargs) -> str:
        """Get translated text for a given category and key."""
        try:
            # Try current language first
            text = self.translations[self.current_language][category][key]
        except KeyError:
            try:
                # Fallback to default language
                text = self.translations[self.default_language][category][key]
            except KeyError:
                # Ultimate fallback
                return f"[{category}.{key}]"
        
        # Format with provided kwargs
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError):
                pass  # Return unformatted text if formatting fails
        
        return text
    
    def get_error(self, key: str, **kwargs) -> str:
        """Get translated error message."""
        return self.get_text("errors", key, **kwargs)
    
    def get_message(self, key: str, **kwargs) -> str:
        """Get translated status message."""
        return self.get_text("messages", key, **kwargs)
    
    def get_model_name(self, key: str) -> str:
        """Get translated model name."""
        return self.get_text("model_names", key)
    
    def get_metric_name(self, key: str) -> str:
        """Get translated metric name."""
        return self.get_text("metrics", key)
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return self.supported_languages.copy()
    
    def add_translation(self, language: str, category: str, key: str, text: str):
        """Add a new translation."""
        if language not in self.translations:
            self.translations[language] = {}
        if category not in self.translations[language]:
            self.translations[language][category] = {}
        
        self.translations[language][category][key] = text
        
        # Save to file
        lang_file = self.locale_dir / f"{language}.json"
        with open(lang_file, 'w', encoding='utf-8') as f:
            json.dump(self.translations[language], f, ensure_ascii=False, indent=2)


# Global i18n manager instance
i18n = I18nManager()


# Convenience functions
def set_language(language: str):
    """Set the framework language."""
    i18n.set_language(language)


def get_error_msg(key: str, **kwargs) -> str:
    """Get translated error message."""
    return i18n.get_error(key, **kwargs)


def get_msg(key: str, **kwargs) -> str:
    """Get translated status message.""" 
    return i18n.get_message(key, **kwargs)


def get_model_name(key: str) -> str:
    """Get translated model name."""
    return i18n.get_model_name(key)


# Auto-detect system language
def detect_system_language() -> str:
    """Detect system language and set it if supported."""
    import locale
    import os
    
    # Try various methods to detect language
    try:
        # Method 1: locale.getdefaultlocale()
        lang_code = locale.getdefaultlocale()[0]
        if lang_code:
            lang = lang_code.split('_')[0]
            if lang in i18n.get_supported_languages():
                return lang
    except:
        pass
    
    # Method 2: Environment variables
    for env_var in ['LANG', 'LC_ALL', 'LC_MESSAGES']:
        if env_var in os.environ:
            lang_code = os.environ[env_var]
            if lang_code:
                lang = lang_code.split('_')[0].split('.')[0]
                if lang in i18n.get_supported_languages():
                    return lang
    
    # Default to English
    return 'en'


# Initialize with system language
system_lang = detect_system_language()
i18n.set_language(system_lang)