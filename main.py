import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QComboBox,
                             QLabel, QMessageBox, QSplitter, QFrame)
from PyQt5.QtCore import Qt

DEFAULT_PEAK_WINDOW = 50.0

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
        self.orig_xlim = None
        self.orig_ylim = None
        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('button_press_event', self.on_double_click)

    def on_scroll(self, event):
        if self.orig_xlim is None:
            self.orig_xlim = self.ax.get_xlim()
            self.orig_ylim = self.ax.get_ylim()
        ax = self.ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        scale_factor = 1.1 if event.button == 'up' else 0.9
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return
        new_xlim = [xdata - (xdata - xlim[0]) * scale_factor,
                    xdata + (xlim[1] - xdata) * scale_factor]
        new_ylim = [ydata - (ydata - ylim[0]) * scale_factor,
                    ydata + (ylim[1] - ydata) * scale_factor]
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.draw_idle()

    def on_double_click(self, event):
        if event.dblclick and self.orig_xlim is not None:
            self.ax.set_xlim(self.orig_xlim)
            self.ax.set_ylim(self.orig_ylim)
            self.draw_idle()

    def set_original_limits(self):
        self.orig_xlim = self.ax.get_xlim()
        self.orig_ylim = self.ax.get_ylim()

def insert_nan_at_gaps(x, y, threshold_factor=10):
    if len(x) < 2:
        return x, y
    diffs = np.diff(x)
    median_diff = np.median(diffs)
    if median_diff == 0:
        gap_threshold = 1
    else:
        gap_threshold = median_diff * threshold_factor
    gap_indices = np.where(diffs > gap_threshold)[0] + 1
    x_new = []
    y_new = []
    start = 0
    for idx in gap_indices:
        x_new.extend(x[start:idx])
        y_new.extend(y[start:idx])
        x_new.append(np.nan)
        y_new.append(np.nan)
        start = idx
    x_new.extend(x[start:])
    y_new.extend(y[start:])
    return np.array(x_new), np.array(y_new)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EPSP / PS Viewer")
        self.setGeometry(100, 100, 1200, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        control_layout = QHBoxLayout()
        self.select_btn = QPushButton("Выбрать папку")
        self.select_btn.clicked.connect(self.select_folder)
        control_layout.addWidget(self.select_btn)

        control_layout.addWidget(QLabel("Пара:"))
        self.pair_combo = QComboBox()
        self.pair_combo.setEnabled(False)
        self.pair_combo.setMinimumWidth(400)
        self.pair_combo.currentIndexChanged.connect(self.on_pair_selected)
        control_layout.addWidget(self.pair_combo)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        splitter = QSplitter(Qt.Vertical)

        self.signal_widget = QFrame()
        signal_layout = QVBoxLayout(self.signal_widget)
        self.canvas_signal = MplCanvas(self, width=9, height=4)
        self.toolbar_signal = NavigationToolbar(self.canvas_signal, self)
        signal_layout.addWidget(self.toolbar_signal)
        signal_layout.addWidget(self.canvas_signal)
        splitter.addWidget(self.signal_widget)

        self.amplitude_widget = QFrame()
        amplitude_layout = QVBoxLayout(self.amplitude_widget)
        self.canvas_amplitude = MplCanvas(self, width=9, height=4)
        self.toolbar_amplitude = NavigationToolbar(self.canvas_amplitude, self)
        amplitude_layout.addWidget(self.toolbar_amplitude)
        amplitude_layout.addWidget(self.canvas_amplitude)
        splitter.addWidget(self.amplitude_widget)

        main_layout.addWidget(splitter)

        self.canvas_amplitude.mpl_connect('button_press_event', self.on_amplitude_click)

        self.pairs = []
        self.current_signal_x = None
        self.current_signal_y = None
        self.current_amplitude_x = None
        self.current_amplitude_y = None
        self.current_amplitude_y_raw = None
        self.current_sig_type = None

        self.current_vline = None
        self.current_hline = None
        self.current_point = None

    def extract_key_and_type(self, filename):
        basename = os.path.basename(filename)
        for prefix in ['EPSP-', 'PS-']:
            if basename.startswith(prefix):
                sig_type = prefix[:-1]
                rest = basename[len(prefix):]
                if '_' in rest:
                    key = rest.split('_')[0]
                else:
                    key = rest
                return sig_type, key
        return None, None

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с файлами")
        if not folder:
            return
        self.scan_folder(folder)

    def scan_folder(self, folder):
        signal_patterns = [os.path.join(folder, "EPSP-*"), os.path.join(folder, "PS-*")]
        signal_files = []
        for pattern in signal_patterns:
            signal_files.extend(glob.glob(pattern))

        if not signal_files:
            QMessageBox.warning(self, "Предупреждение",
                                "Не найдено файлов, начинающихся с EPSP- или PS-")
            self.pair_combo.clear()
            self.pair_combo.setEnabled(False)
            return

        pairs = []
        for sig_path in signal_files:
            sig_type, key = self.extract_key_and_type(sig_path)
            if not sig_type or not key:
                continue
            amp_prefix = f"Amplitude{sig_type}-{key}"
            amp_pattern = os.path.join(folder, amp_prefix + "*")
            amp_files = glob.glob(amp_pattern)
            if amp_files:
                amp_path = amp_files[0]
                pairs.append((sig_type, key, sig_path, amp_path))

        if not pairs:
            QMessageBox.warning(self, "Предупреждение",
                                "Не найдено парных Amplitude-файлов")
            self.pair_combo.clear()
            self.pair_combo.setEnabled(False)
            return

        pairs.sort(key=lambda x: (x[0], x[1]))
        self.pairs = pairs

        self.pair_combo.clear()
        for sig_type, key, _, _ in pairs:
            self.pair_combo.addItem(f"{sig_type}: {key}")
        self.pair_combo.setEnabled(True)

        if pairs:
            self.plot_pair(0)

    def on_pair_selected(self, index):
        if index >= 0:
            self.plot_pair(index)

    def plot_pair(self, index):
        if index < 0 or index >= len(self.pairs):
            return

        sig_type, key, sig_path, amp_path = self.pairs[index]
        self.current_sig_type = sig_type

        try:
            df_sig = pd.read_csv(sig_path, sep=None, engine='python')
            df_amp = pd.read_csv(amp_path, sep=None, engine='python')

            xcol_sig = df_sig.columns[0]
            ycol_sig = df_sig.columns[1] if len(df_sig.columns) > 1 else xcol_sig
            xcol_amp = df_amp.columns[0]
            ycol_amp = df_amp.columns[1] if len(df_amp.columns) > 1 else xcol_amp

            raw_amp = df_amp[ycol_amp].values
            amp_y_values = raw_amp * -1

            self.current_signal_x = df_sig[xcol_sig].values
            self.current_signal_y = df_sig[ycol_sig].values
            self.current_amplitude_x = df_amp[xcol_amp].values
            self.current_amplitude_y = amp_y_values
            self.current_amplitude_y_raw = raw_amp

            x_sig, y_sig = insert_nan_at_gaps(self.current_signal_x, self.current_signal_y)
            x_amp, y_amp = insert_nan_at_gaps(self.current_amplitude_x, self.current_amplitude_y)

            self.canvas_signal.ax.clear()
            self.canvas_signal.ax.plot(x_sig, y_sig, color='b', linewidth=1.5, label=sig_type)
            self.canvas_signal.ax.set_title(f"{sig_type}: {key}")
            self.canvas_signal.ax.set_xlabel("Время (мс)")
            self.canvas_signal.ax.set_ylabel(f"{sig_type} (мВ)")
            self.canvas_signal.ax.grid(True, linestyle='--', alpha=0.7)
            self.canvas_signal.ax.legend()
            self.canvas_signal.fig.tight_layout()
            self.canvas_signal.draw()
            self.canvas_signal.set_original_limits()

            self.canvas_amplitude.ax.clear()
            self.canvas_amplitude.ax.scatter(x_amp, y_amp, color='r', s=10, alpha=0.8,
                                              label=f"Amplitude {sig_type}")
            self.canvas_amplitude.ax.set_title(f"Amplitude {sig_type}: {key}")
            self.canvas_amplitude.ax.set_xlabel("Время (мс)")
            self.canvas_amplitude.ax.set_ylabel(f"Амплитуда {sig_type} (мВ)")
            self.canvas_amplitude.ax.grid(True, linestyle='--', alpha=0.7)
            self.canvas_amplitude.ax.legend()
            self.canvas_amplitude.fig.tight_layout()
            self.canvas_amplitude.draw()
            self.canvas_amplitude.set_original_limits()

            self.current_vline = None
            self.current_hline = None
            self.current_point = None

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить график:\n{str(e)}")

    def find_continuous_segment(self, t, threshold_factor=10):
        x = self.current_signal_x
        if len(x) < 2:
            return False, 0
        diffs = np.diff(x)
        median_diff = np.median(diffs)
        if median_diff == 0:
            gap_threshold = 1
        else:
            gap_threshold = median_diff * threshold_factor

        idx_closest = np.argmin(np.abs(x - t))
        is_continuous = True
        if idx_closest > 0:
            if abs(x[idx_closest] - x[idx_closest-1]) > gap_threshold:
                is_continuous = False
        if idx_closest < len(x)-1:
            if abs(x[idx_closest+1] - x[idx_closest]) > gap_threshold:
                is_continuous = False
        return is_continuous, idx_closest

    def on_amplitude_click(self, event):
        if event.inaxes != self.canvas_amplitude.ax:
            return
        if event.xdata is None:
            return
        t_click = event.xdata

        idx_amp = np.argmin(np.abs(self.current_amplitude_x - t_click))
        t_amp = self.current_amplitude_x[idx_amp]
        raw_amp = self.current_amplitude_y_raw[idx_amp]

        is_cont, idx_closest = self.find_continuous_segment(t_amp)
        if not is_cont:
            t_amp = self.current_signal_x[idx_closest]
            print(f"Коррекция: время смещено к {t_amp:.2f} мс (разрыв)")

        half_window = 50.0
        left = t_amp - half_window
        right = t_amp + half_window

        signal_times = self.current_signal_x
        mask = (signal_times >= left) & (signal_times <= right)
        if not np.any(mask):
            idx_sig = np.argmin(np.abs(signal_times - t_amp))
            t_sig = signal_times[idx_sig]
            left = t_sig - half_window
            right = t_sig + half_window
            mask = (signal_times >= left) & (signal_times <= right)

        x_win = signal_times[mask]
        y_win = self.current_signal_y[mask]
        if len(y_win) == 0:
            return

        y_min, y_max = y_win.min(), y_win.max()
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1
        y_low = y_min - 0.1 * y_range
        y_high = y_max + 0.1 * y_range

        self.canvas_signal.ax.set_xlim(left, right)
        self.canvas_signal.ax.set_ylim(y_low, y_high)

        for art in [self.current_vline, self.current_hline, self.current_point]:
            if art is not None:
                art.remove()

        self.current_vline = self.canvas_signal.ax.axvline(
            x=t_amp, color='red', linestyle='--', linewidth=1, alpha=0.7,
            label=f'пик при t={t_amp:.2f} мс'
        )
        self.current_hline = self.canvas_signal.ax.axhline(
            y=raw_amp, color='green', linestyle=':', linewidth=1, alpha=0.7,
            label=f'амплитуда = {raw_amp:.2f} мВ'
        )
        y_interp = np.interp(t_amp, self.current_signal_x, self.current_signal_y)
        self.current_point = self.canvas_signal.ax.plot(
            t_amp, y_interp, 'ro', markersize=6, markeredgecolor='darkred',
            label='интерп. сигнал'
        )[0]

        self.canvas_signal.ax.legend()
        self.canvas_signal.draw_idle()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()