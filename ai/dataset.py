"""
dataset.py - Dataset Recording Module
=====================================
Grabacion automatica de frames + mascaras para entrenamiento de U-Net.
Incluye soporte para recortes y labels correctos.

IMPORTANTE: Cuando se usa recorte, se guarda el recorte en lugar de la
imagen completa, y los labels se calculan relativos al recorte.
"""

import cv2
import numpy as np
import os
import csv
from datetime import datetime
from typing import Optional, Tuple


class DatasetRecorder:
    """
    Graba pares (imagen, mascara) automaticamente cuando se detectan
    pupila e iris con suficiente confianza.
    
    Maneja correctamente los recortes:
    - Guarda el recorte de la imagen
    - Calcula coordenadas relativas al recorte
    - Genera mascara en las coordenadas correctas
    """

    def __init__(self, base_dir: str = 'dataset'):
        """
        Args:
            base_dir: Directorio base para el dataset
        """
        self.base_dir  = base_dir
        self.img_dir   = os.path.join(base_dir, 'images')
        self.mask_dir  = os.path.join(base_dir, 'masks')
        self.csv_path  = os.path.join(base_dir, 'labels.csv')
        self.recording = False
        self.counter   = 0
        self._ensure_dirs()
        self._load_counter()

    def _load_counter(self):
        """Carga el contador desde el CSV existente."""
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    rows = list(reader)
                    if rows:
                        self.counter = len(rows)
            except Exception:
                pass

    def _ensure_dirs(self):
        """Crea los directorios necesarios."""
        os.makedirs(self.img_dir,  exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(
                    ['filename', 'pupil_cx', 'pupil_cy', 'pupil_r',
                     'iris_cx', 'iris_cy', 'iris_r', 
                     'crop_offset_x', 'crop_offset_y',
                     'image_width', 'image_height',
                     'is_crop', 'timestamp'])

    def start(self):
        """Inicia la grabacion."""
        self.recording = True

    def stop(self):
        """Detiene la grabacion."""
        self.recording = False

    def toggle(self) -> bool:
        """Alterna el estado de grabacion."""
        self.recording = not self.recording
        return self.recording

    @property
    def is_recording(self) -> bool:
        """Devuelve True si esta grabando."""
        return self.recording

    def try_save(self, 
                 frame: np.ndarray,
                 pupil_center: Optional[Tuple[float, float]], 
                 pupil_radius: Optional[float],
                 iris_center: Optional[Tuple[float, float]],  
                 iris_radius: Optional[float],
                 crop_rect: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        Guarda el par (frame/mascara) si la deteccion es valida.
        
        IMPORTANTE: Si crop_rect esta presente, guarda el recorte y
        calcula las coordenadas relativas al recorte.
        
        Args:
            frame: Frame original
            pupil_center: Centro de la pupila (en coordenadas del frame)
            pupil_radius: Radio de la pupila
            iris_center: Centro del iris (en coordenadas del frame)
            iris_radius: Radio del iris
            crop_rect: Tupla (x1, y1, x2, y2) del recorte, o None
            
        Returns:
            True si se guardo exitosamente
        """
        if not self.recording:
            return False
        
        h, w = frame.shape[:2]
        
        if not self._is_valid(frame, pupil_center, pupil_radius):
            return False
        
        if crop_rect is not None:
            x1, y1, x2, y2 = crop_rect
            frame_to_save = frame[y1:y2, x1:x2]
            crop_offset_x, crop_offset_y = x1, y1
            is_crop = True
        else:
            frame_to_save = frame
            crop_offset_x, crop_offset_y = 0, 0
            is_crop = False
        
        pupil_center_crop = (
            (pupil_center[0] - crop_offset_x, pupil_center[1] - crop_offset_y)
            if pupil_center else None
        )
        iris_center_crop = (
            (iris_center[0] - crop_offset_x, iris_center[1] - crop_offset_y)
            if iris_center else None
        )
        
        mask = self._make_mask(
            frame_to_save.shape, 
            pupil_center_crop, 
            pupil_radius,
            iris_center_crop, 
            iris_radius
        )
        
        fname = f'{self.counter:05d}.png'
        cv2.imwrite(os.path.join(self.img_dir,  fname), frame_to_save)
        cv2.imwrite(os.path.join(self.mask_dir, fname), mask)
        
        self._log_csv(
            fname, 
            pupil_center_crop, pupil_radius,
            iris_center_crop, iris_radius,
            crop_offset_x, crop_offset_y,
            frame_to_save.shape[1], frame_to_save.shape[0],
            is_crop
        )
        
        self.counter += 1
        return True

    def save_with_crop_correction(self,
                                   original_frame: np.ndarray,
                                   pupil_center: Optional[Tuple[float, float]], 
                                   pupil_radius: Optional[float],
                                   iris_center: Optional[Tuple[float, float]],  
                                   iris_radius: Optional[float],
                                   crop_rect: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        Alias para try_save que hace enfasis en la correccion de recorte.
        Guarda el recorte (si existe) y las coordenadas relativas al recorte.
        
        Args:
            original_frame: Frame original completo
            pupil_center: Centro de la pupila (coords originales)
            pupil_radius: Radio de la pupila
            iris_center: Centro del iris (coords originales)
            iris_radius: Radio del iris
            crop_rect: Tupla (x1, y1, x2, y2) del recorte
            
        Returns:
            True si se guardo exitosamente
        """
        return self.try_save(
            original_frame,
            pupil_center, pupil_radius,
            iris_center, iris_radius,
            crop_rect
        )

    def count(self) -> int:
        """Devuelve el numero de muestras guardadas."""
        return self.counter

    def _make_mask(self, 
                   shape: Tuple[int, ...],
                   pupil_center: Optional[Tuple[float, float]], 
                   pupil_radius: Optional[float],
                   iris_center: Optional[Tuple[float, float]],  
                   iris_radius: Optional[float]) -> np.ndarray:
        """
        Genera mascara multiclase:
          255 = pupila
          128 = iris (anillo)
            0 = fondo
            
        Args:
            shape: Forma de la mascara (h, w)
            pupil_center: Centro de la pupila en coordenadas relativas
            pupil_radius: Radio de la pupila
            iris_center: Centro del iris en coordenadas relativas
            iris_radius: Radio del iris
        """
        mask = np.zeros(shape[:2], dtype=np.uint8)
        if iris_center and iris_radius:
            cv2.circle(mask, (int(iris_center[0]), int(iris_center[1])), 
                       int(iris_radius), 128, 2)
        if pupil_center and pupil_radius:
            cv2.circle(mask, (int(pupil_center[0]), int(pupil_center[1])), 
                       int(pupil_radius), 255, -1)
        return mask

    @staticmethod
    def _is_valid(frame: np.ndarray,
                  pupil_center: Optional[Tuple[float, float]], 
                  pupil_radius: Optional[float]) -> bool:
        """
        Verifica si la deteccion es valida para guardar.
        
        Args:
            frame: Frame a verificar
            pupil_center: Centro de la pupila
            pupil_radius: Radio de la pupila
            
        Returns:
            True si la deteccion es valida
        """
        if pupil_center is None or pupil_radius is None:
            return False
        h, w = frame.shape[:2]
        cx, cy = pupil_center
        if cx < 10 or cy < 10 or cx > w - 10 or cy > h - 10:
            return False
        if pupil_radius < 5 or pupil_radius > min(h, w) // 3:
            return False
        return True

    def _log_csv(self, 
                 fname: str,
                 pc: Optional[Tuple[float, float]], 
                 pr: Optional[float],
                 ic: Optional[Tuple[float, float]],  
                 ir: Optional[float],
                 crop_x: int,
                 crop_y: int,
                 img_w: int,
                 img_h: int,
                 is_crop: bool):
        """
        Registra los datos en el CSV de labels.
        
        Args:
            fname: Nombre del archivo
            pc: Centro de pupila (relativo al recorte)
            pr: Radio de pupila
            ic: Centro de iris (relativo al recorte)
            ir: Radio de iris
            crop_x: Offset X del recorte
            crop_y: Offset Y del recorte
            img_w: Ancho de la imagen guardada
            img_h: Alto de la imagen guardada
            is_crop: True si es un recorte
        """
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                fname,
                f"{pc[0]:.1f}" if pc else '', 
                f"{pc[1]:.1f}" if pc else '', 
                f"{pr:.1f}" if pr else '',
                f"{ic[0]:.1f}" if ic else '',
                f"{ic[1]:.1f}" if ic else '',
                f"{ir:.1f}" if ir else '',
                crop_x,
                crop_y,
                img_w,
                img_h,
                '1' if is_crop else '0',
                datetime.now().isoformat(timespec='seconds')
            ])

    def get_labels_path(self) -> str:
        """Devuelve la ruta del archivo de labels."""
        return self.csv_path


def review_dataset(base_dir: str = 'dataset'):
    """
    Revision rapida de dataset: muestra cada par imagen/mascara.
    
    Teclas:
      d -> eliminar (malo)
      espacio -> siguiente (bueno)
      q -> salir
      
    Returns:
        Numero de items eliminados
    """
    img_dir  = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'masks')
    files    = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))
    deleted  = 0

    for fname in files:
        img_path  = os.path.join(img_dir,  fname)
        mask_path = os.path.join(mask_dir, fname)

        img  = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        if img is None:
            continue

        overlay = img.copy()
        if mask is not None:
            overlay[mask == 255] = (0,   0, 200)
            overlay[mask == 128] = (200, 0,   0)
        combined = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        cv2.putText(combined, fname,
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.putText(combined, 'd=delete  space=keep  q=quit',
                    (10, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow('Dataset Review', combined)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('d'):
            os.remove(img_path)
            if os.path.exists(mask_path):
                os.remove(mask_path)
            deleted += 1
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    return deleted


def load_dataset_labels(labels_path: str) -> list:
    """
    Carga los labels desde el CSV.
    
    Args:
        labels_path: Ruta al archivo de labels
        
    Returns:
        Lista de diccionarios con los labels
    """
    labels = []
    if not os.path.exists(labels_path):
        return labels
    
    with open(labels_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append({
                'filename': row['filename'],
                'pupil_cx': float(row['pupil_cx']) if row['pupil_cx'] else None,
                'pupil_cy': float(row['pupil_cy']) if row['pupil_cy'] else None,
                'pupil_r': float(row['pupil_r']) if row['pupil_r'] else None,
                'iris_cx': float(row['iris_cx']) if row['iris_cx'] else None,
                'iris_cy': float(row['iris_cy']) if row['iris_cy'] else None,
                'iris_r': float(row['iris_r']) if row['iris_r'] else None,
                'crop_offset_x': int(row['crop_offset_x']),
                'crop_offset_y': int(row['crop_offset_y']),
                'image_width': int(row['image_width']),
                'image_height': int(row['image_height']),
                'is_crop': row['is_crop'] == '1',
            })
    
    return labels
