"""
settings_service.py - Servicio de Configuracion
===============================================
Maneja la persistencia y administracion de configuraciones
de la aplicacion pupilometer.
"""

import json
import os
from typing import Dict, Any, Optional


class SettingsService:
    """
    Servicio para gestionar configuraciones de la aplicacion.
    
    Maneja la persistencia de configuraciones en un archivo JSON
    y provee acceso centralizado a los valores de configuracion.
    
    Attributes:
        config_path: Ruta al archivo de configuracion
        config: Diccionario con las configuraciones actuales
        
    Example:
        >>> service = SettingsService()
        >>> service.set('flash_detection.auto', True)
        >>> value = service.get('flash_detection.calibration_frames')
    """
    
    DEFAULT_CONFIG = {
        'flash_detection': {
            'auto': False,
            'calibration_frames': 5,
            'change_threshold': 5.0,
            'smoothing_window': 4
        },
        'video_phases': {
            'basal_fps': 2,
            'basal_duration': 3,
            'contraccion_fps': 30,
            'contraccion_duration': 2,
            'relajacion_fps': 10,
            'relajacion_duration': 7
        },
        'protocol': {
            'flash_on_sec': 1.0,
            'flash_off_sec': 1.2
        },
        'calibration': {
            'distancia_regla_mm': 40.0,
            'iris_mm_default': 11.7
        },
        'ui': {
            'canvas_width': 640,
            'canvas_height': 480,
            'target_fps': 30
        },
        'detection': {
            'pupil_algorithm': 'starburst',
            'iris_algorithm': 'gradient',
            'use_ai': False
        }
    }
    
    def __init__(self, config_path: str = 'pupilometer_config.json'):
        """
        Inicializa el servicio de configuracion.
        
        Args:
            config_path: Ruta al archivo de configuracion (default: pupilometer_config.json)
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Carga la configuracion desde el archivo.
        
        Si el archivo no existe o esta corrupto, retorna la configuracion por defecto.
        
        Returns:
            Diccionario con la configuracion
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return self._deep_copy(self.DEFAULT_CONFIG)
    
    def _save_config(self):
        """Guarda la configuracion actual al archivo."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f'Error guardando configuracion: {e}')
    
    def _deep_copy(self, obj: Any) -> Any:
        """
        Hace una copia profunda de un objeto.
        
        Args:
            obj: Objeto a copiar
            
        Returns:
            Copia profunda del objeto
        """
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuracion usando notacion de punto.
        
        Args:
            key: Clave en notacion de punto (ej: 'flash_detection.auto')
            default: Valor por defecto si no existe
            
        Returns:
            Valor de la configuracion o default
            
        Example:
            >>> service.get('flash_detection.auto')
            False
            >>> service.get('nonexistent.key', 'default')
            'default'
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any, save: bool = True):
        """
        Establece un valor de configuracion usando notacion de punto.
        
        Args:
            key: Clave en notacion de punto (ej: 'flash_detection.auto')
            value: Valor a establecer
            save: Si True, guarda inmediatamente al archivo
            
        Example:
            >>> service.set('flash_detection.auto', True)
            >>> service.set('custom.nested.value', 42)
        """
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        
        if save:
            self._save_config()
    
    def get_flash_detection_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuracion de deteccion de flash.
        
        Returns:
            Diccionario con configuracion de flash
        """
        return self.config.get('flash_detection', self.DEFAULT_CONFIG['flash_detection'])
    
    def get_video_phases_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuracion de fases de video.
        
        Returns:
            Diccionario con configuracion de fases
        """
        return self.config.get('video_phases', self.DEFAULT_CONFIG['video_phases'])
    
    def get_protocol_config(self) -> Dict[str, float]:
        """
        Obtiene la configuracion del protocolo de estimulacion.
        
        Returns:
            Diccionario con configuracion del protocolo
        """
        return self.config.get('protocol', self.DEFAULT_CONFIG['protocol'])
    
    def get_calibration_config(self) -> Dict[str, float]:
        """
        Obtiene la configuracion de calibracion.
        
        Returns:
            Diccionario con configuracion de calibracion
        """
        return self.config.get('calibration', self.DEFAULT_CONFIG['calibration'])
    
    def get_detection_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuracion de deteccion.
        
        Returns:
            Diccionario con configuracion de deteccion
        """
        return self.config.get('detection', self.DEFAULT_CONFIG['detection'])
    
    def reset_to_defaults(self):
        """Restablece todas las configuraciones a los valores por defecto."""
        self.config = self._deep_copy(self.DEFAULT_CONFIG)
        self._save_config()
    
    def update_flash_detection_from_temp(self, temp_values: Dict[str, Any]):
        """
        Actualiza la configuracion de flash desde valores de temp.txt.
        
        Args:
            temp_values: Diccionario con valores del archivo temp.txt
        """
        flash_config = self.config.get('flash_detection', {})
        
        if 'CALIBRATION_FRAMES' in temp_values:
            flash_config['calibration_frames'] = temp_values['CALIBRATION_FRAMES']
        if 'CHANGE_THRESHOLD' in temp_values:
            flash_config['change_threshold'] = temp_values['CHANGE_THRESHOLD']
        if 'SMOOTHING_WINDOW' in temp_values:
            flash_config['smoothing_window'] = temp_values['SMOOTHING_WINDOW']
            
        self.config['flash_detection'] = flash_config
        self._save_config()
    
    def load_from_temp_file(self, temp_path: str = 'temp.txt') -> Dict[str, Any]:
        """
        Carga valores desde el archivo temp.txt.
        
        Args:
            temp_path: Ruta al archivo temp.txt
            
        Returns:
            Diccionario con los valores encontrados
        """
        temp_values = {}
        
        if os.path.exists(temp_path):
            try:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for line in content.split('\n'):
                    line = line.strip()
                    if '=' in line and not line.startswith('#') and not line.startswith('"""'):
                        parts = line.split('=')
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value_str = parts[1].strip()
                            
                            try:
                                if '.' in value_str:
                                    value = float(value_str)
                                else:
                                    value = int(value_str)
                                temp_values[key] = value
                            except ValueError:
                                pass
                                
                self.update_flash_detection_from_temp(temp_values)
            except (IOError, ValueError):
                pass
                
        return temp_values
