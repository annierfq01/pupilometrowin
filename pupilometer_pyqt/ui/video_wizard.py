#!/usr/bin/env python3
"""
Video Import Wizard - 5 Step Configuration
Step 1: Illumination timing selection
Step 2: Manual frame editor (if manual selected)
Step 3: Frame reduction per phase
Step 4: Video cropping
Step 5: Confirmation
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QRadioButton, QPushButton, QSpinBox, QMessageBox
)
from PyQt6.QtCore import pyqtSignal
from pupilometer_pyqt.core import VideoProcessor


class VideoImportWizard(QDialog):
    """Multi-step wizard for video import"""
    
    def __init__(self, parent, video_path: str):
        super().__init__(parent)
        self.setWindowTitle("Video Import Wizard")
        self.setGeometry(200, 200, 800, 600)
        self.setModal(True)
        
        self.video_path = video_path
        self.processor = VideoProcessor(video_path)
        
        if not self.processor.open():
            QMessageBox.critical(self, "Error", "Cannot open video file")
            self.reject()
            return
        
        self.current_step = 0
        self.illumination_mode = "predeterminado"
        self.flash_start = None
        self.flash_end = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize wizard UI"""
        self.main_layout = QVBoxLayout(self)
        
        self.title = QLabel("<b>Step 1/5: Illumination Timing</b>")
        self.main_layout.addWidget(self.title)
        
        # Content area
        self.content_layout = QVBoxLayout()
        self.show_step_1()
        self.main_layout.addLayout(self.content_layout, 1)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.previous_step)
        self.prev_btn.setEnabled(False)
        button_layout.addWidget(self.prev_btn)
        
        button_layout.addStretch()
        
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_step)
        button_layout.addWidget(self.next_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        self.main_layout.addLayout(button_layout)
    
    def clear_content(self):
        """Clear content layout"""
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def show_step_1(self):
        """Step 1: Select illumination mode"""
        self.clear_content()
        
        self.content_layout.addWidget(QLabel(\n            \"<b>Select Illumination Timing Mode</b>\\n\\n\"\n            \"Choose how to determine flash timing:\"\n        ))\n        \n        # Predeterminado\n        radio1 = QRadioButton(\"Predeterminado (Use default timing values)\")\n        radio1.setChecked(True)\n        radio1.clicked.connect(lambda: self.set_mode(\"predeterminado\"))\n        self.content_layout.addWidget(radio1)\n        \n        # Automatico\n        radio2 = QRadioButton(\"Automático (Auto-detect flash from brightness)\")\n        radio2.clicked.connect(lambda: self.set_mode(\"automatico\"))\n        self.content_layout.addWidget(radio2)\n        \n        # Manual\n        radio3 = QRadioButton(\"Manual (Select frames manually)\")\n        radio3.clicked.connect(lambda: self.set_mode(\"manual\"))\n        self.content_layout.addWidget(radio3)\n        \n        self.content_layout.addStretch()\n        \n        self.title.setText(\"<b>Step 1/5: Illumination Timing</b>\")\n        self.prev_btn.setEnabled(False)\n        self.next_btn.setText(\"Next\")\n    \n    def set_mode(self, mode: str):\n        self.illumination_mode = mode\n    \n    def show_step_2(self):\n        \"\"\"Step 2: Manual editor (if selected)\"\"\"\n        if self.illumination_mode != \"manual\":\n            self.show_step_3()\n            return\n        \n        self.clear_content()\n        self.content_layout.addWidget(QLabel(\n            \"<b>Manual Flash Detection</b>\\n\\n\"\n            \"Use frame navigation to find flash boundaries:\"\n        ))\n        \n        self.content_layout.addWidget(QLabel(f\n            \"Total Frames: {self.processor.info.frame_count}\\n\"\n            \"FPS: {self.processor.info.fps:.1f}\"\n        ))\n        \n        # Frame controls\n        controls = QHBoxLayout()\n        \n        btn_start = QPushButton(\"Mark Flash Start\")\n        btn_start.clicked.connect(self.mark_start)\n        controls.addWidget(btn_start)\n        \n        btn_end = QPushButton(\"Mark Flash End\")\n        btn_end.clicked.connect(self.mark_end)\n        controls.addWidget(btn_end)\n        \n        self.flash_label = QLabel(\"Flash: [not marked]\")\n        controls.addWidget(self.flash_label)\n        \n        self.content_layout.addLayout(controls)\n        \n        # Frame range input\n        range_layout = QHBoxLayout()\n        range_layout.addWidget(QLabel(\"Go to frame:\"))\n        spin = QSpinBox()\n        spin.setMaximum(self.processor.info.frame_count - 1)\n        spin.valueChanged.connect(self.jump_to_frame)\n        range_layout.addWidget(spin)\n        self.content_layout.addLayout(range_layout)\n        \n        self.content_layout.addStretch()\n        \n        self.title.setText(\"<b>Step 2/5: Manual Flash Detection</b>\")\n        self.prev_btn.setEnabled(True)\n        self.next_btn.setText(\"Next\")\n    \n    def mark_start(self):\n        \"\"\"Mark flash start\"\"\"\n        self.flash_start = self.processor.current_frame_idx\n        self.update_flash_label()\n    \n    def mark_end(self):\n        \"\"\"Mark flash end\"\"\"\n        self.flash_end = self.processor.current_frame_idx\n        self.update_flash_label()\n    \n    def update_flash_label(self):\n        \"\"\"Update flash marker display\"\"\"\n        text = \"Flash: \"\n        if self.flash_start is not None:\n            text += f\"[{self.flash_start}\"\n            if self.flash_end is not None:\n                text += f\" - {self.flash_end}]\"\n            else:\n                text += \" - ?]\"\n        else:\n            text += \"[not marked]\"\n        self.flash_label.setText(text)\n    \n    def jump_to_frame(self, frame_idx: int):\n        \"\"\"Jump to frame\"\"\"\n        frame = self.processor.read_frame(frame_idx)\n        # Could display frame here if viewer widget was available\n    \n    def show_step_3(self):\n        \"\"\"Step 3: Frame reduction\"\"\"\n        self.clear_content()\n        \n        self.content_layout.addWidget(QLabel(\n            \"<b>Frame Reduction Configuration</b>\\n\\n\"\n            \"Specify target FPS for each phase. Frames will be reduced accordingly.\\n\"\n            \"Calculations will use only the reduced frame set.\"\n        ))\n        \n        # Basal\n        basal_layout = QHBoxLayout()\n        basal_layout.addWidget(QLabel(\"Basal Phase FPS:\"))\n        basal_spin = QSpinBox()\n        basal_spin.setRange(1, 30)\n        basal_spin.setValue(2)\n        basal_layout.addWidget(basal_spin)\n        self.content_layout.addLayout(basal_layout)\n        \n        # Illumination\n        illum_layout = QHBoxLayout()\n        illum_layout.addWidget(QLabel(\"Illumination Phase FPS:\"))\n        illum_spin = QSpinBox()\n        illum_spin.setRange(1, 60)\n        illum_spin.setValue(30)\n        illum_layout.addWidget(illum_spin)\n        self.content_layout.addLayout(illum_layout)\n        \n        # Relaxation\n        relax_layout = QHBoxLayout()\n        relax_layout.addWidget(QLabel(\"Relaxation Phase FPS:\"))\n        relax_spin = QSpinBox()\n        relax_spin.setRange(1, 30)\n        relax_spin.setValue(10)\n        relax_layout.addWidget(relax_spin)\n        self.content_layout.addLayout(relax_layout)\n        \n        self.frame_reduction = {\n            'basal': basal_spin.value(),\n            'illumination': illum_spin.value(),\n            'relaxation': relax_spin.value()\n        }\n        \n        self.content_layout.addStretch()\n        \n        self.title.setText(\"<b>Step 3/5: Frame Reduction</b>\")\n        self.prev_btn.setEnabled(True)\n        self.next_btn.setText(\"Next\")\n    \n    def show_step_4(self):\n        \"\"\"Step 4: Cropping\"\"\"\n        self.clear_content()\n        \n        self.content_layout.addWidget(QLabel(\n            \"<b>Video Cropping (Optional)</b>\\n\\n\"\n            \"Leave at 0 to disable cropping\"\n        ))\n        \n        # Crop params\n        crop_layout = QHBoxLayout()\n        crop_layout.addWidget(QLabel(\"X:\"))\n        x_spin = QSpinBox()\n        x_spin.setMaximum(self.processor.info.width)\n        crop_layout.addWidget(x_spin)\n        \n        crop_layout.addWidget(QLabel(\"Y:\"))\n        y_spin = QSpinBox()\n        y_spin.setMaximum(self.processor.info.height)\n        crop_layout.addWidget(y_spin)\n        \n        crop_layout.addWidget(QLabel(\"Width:\"))\n        w_spin = QSpinBox()\n        w_spin.setRange(0, self.processor.info.width)\n        w_spin.setValue(self.processor.info.width)\n        crop_layout.addWidget(w_spin)\n        \n        crop_layout.addWidget(QLabel(\"Height:\"))\n        h_spin = QSpinBox()\n        h_spin.setRange(0, self.processor.info.height)\n        h_spin.setValue(self.processor.info.height)\n        crop_layout.addWidget(h_spin)\n        \n        self.content_layout.addLayout(crop_layout)\n        self.content_layout.addStretch()\n        \n        self.title.setText(\"<b>Step 4/5: Video Cropping</b>\")\n        self.prev_btn.setEnabled(True)\n        self.next_btn.setText(\"Next\")\n    \n    def show_step_5(self):\n        \"\"\"Step 5: Confirmation\"\"\"\n        self.clear_content()\n        \n        summary = (\n            f\"<b>Configuration Summary</b>\\n\\n\"\n            f\"Video: {self.processor.info.filename}\\n\"\n            f\"Frames: {self.processor.info.frame_count}\\n\"\n            f\"FPS: {self.processor.info.fps:.2f}\\n\"\n            f\"Duration: {self.processor.info.duration:.2f}s\\n\"\n            f\"Resolution: {self.processor.info.width}x{self.processor.info.height}\\n\\n\"\n            f\"Illumination Mode: {self.illumination_mode}\\n\"\n        )\n        \n        if self.flash_start is not None:\n            summary += f\"Flash Start Frame: {self.flash_start}\\n\"\n        if self.flash_end is not None:\n            summary += f\"Flash End Frame: {self.flash_end}\\n\"\n        \n        summary += (\n            f\"\\nFrame reduction will be applied.\\n\"\n            f\"All calculations will account for reduced frame count.\\n\"\n            f\"Timestamps will remain in seconds from original video.\"\n        )\n        \n        self.content_layout.addWidget(QLabel(summary))\n        self.content_layout.addStretch()\n        \n        self.title.setText(\"<b>Step 5/5: Confirm & Import</b>\")\n        self.prev_btn.setEnabled(True)\n        self.next_btn.setText(\"Finish\")\n    \n    def next_step(self):\n        \"\"\"Go to next step\"\"\"\n        steps = [\n            self.show_step_1,\n            self.show_step_2,\n            self.show_step_3,\n            self.show_step_4,\n            self.show_step_5\n        ]\n        \n        if self.current_step < len(steps) - 1:\n            self.current_step += 1\n            steps[self.current_step]()\n        else:\n            self.accept()\n    \n    def previous_step(self):\n        \"\"\"Go to previous step\"\"\"\n        steps = [\n            self.show_step_1,\n            self.show_step_2,\n            self.show_step_3,\n            self.show_step_4,\n            self.show_step_5\n        ]\n        \n        if self.current_step > 0:\n            self.current_step -= 1\n            steps[self.current_step]()\n    \n    def get_processor(self) -> VideoProcessor:\n        \"\"\"Return configured processor\"\"\"\n        return self.processor
