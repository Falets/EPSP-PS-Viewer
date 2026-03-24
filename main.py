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
        self.setWindowTitle("EPSP / PS Viewer (paired)")
        self.setGeometry(100, 100, 1200, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        control_layout = QHBoxLayout()
        self.select_btn = QPushButton("Выбрать папку")
        self.select_btn.clicked.connect(self.select_folder)
        control_layout.addWidget(self.select_btn)

        control_layout.addWidget(QLabel("Ключ:"))
        self.key_combo = QComboBox()
        self.key_combo.setEnabled(False)
        self.key_combo.setMinimumWidth(400)
        self.key_combo.currentIndexChanged.connect(self.on_key_selected)
        control_layout.addWidget(self.key_combo)

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

        # Данные для текущего ключа
        self.current_key = None
        self.signal_x_epsp = None
        self.signal_y_epsp = None
        self.signal_x_ps = None
        self.signal_y_ps = None
        self.amp_x_epsp = None
        self.amp_y_epsp = None
        self.amp_x_ps = None
        self.amp_y_ps = None
        self.has_epsp_signal = False
        self.has_ps_signal = False
        self.has_epsp_amp = False
        self.has_ps_amp = False

        # Аннотации
        self.current_vline = None
        self.current_hline = None
        self.current_point = None
        self.last_amp_type = None   # 'EPSP' или 'PS'

    def extract_key_and_type(self, filename):
        """Возвращает (тип, ключ, является_ли_амплитудой)"""
        basename = os.path.basename(filename)
        if basename.startswith('AmplitudeEPSP-'):
            sig_type = 'EPSP'
            key = basename[len('AmplitudeEPSP-'):]
            is_amp = True
        elif basename.startswith('AmplitudePS-'):
            sig_type = 'PS'
            key = basename[len('AmplitudePS-'):]
            is_amp = True
        elif basename.startswith('EPSP-'):
            sig_type = 'EPSP'
            key = basename[len('EPSP-'):]
            is_amp = False
        elif basename.startswith('PS-'):
            sig_type = 'PS'
            key = basename[len('PS-'):]
            is_amp = False
        else:
            return None, None, None
        key = os.path.splitext(key)[0]   # убираем расширение
        return sig_type, key, is_amp

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с файлами")
        if not folder:
            return
        self.scan_folder(folder)

    def scan_folder(self, folder):
        patterns = ['EPSP-*', 'PS-*', 'AmplitudeEPSP-*', 'AmplitudePS-*']
        all_files = []
        for pattern in patterns:
            all_files.extend(glob.glob(os.path.join(folder, pattern)))

        if not all_files:
            QMessageBox.warning(self, "Предупреждение",
                                "Не найдено файлов, начинающихся с EPSP-, PS-, AmplitudeEPSP- или AmplitudePS-")
            self.key_combo.clear()
            self.key_combo.setEnabled(False)
            return

        # Собираем данные по ключам
        keys_data = {}
        for f in all_files:
            sig_type, key, is_amp = self.extract_key_and_type(f)
            if sig_type is None or key is None:
                continue
            if key not in keys_data:
                keys_data[key] = {'epsp_sig': None, 'ps_sig': None,
                                  'epsp_amp': None, 'ps_amp': None}
            if not is_amp:
                if sig_type == 'EPSP':
                    keys_data[key]['epsp_sig'] = f
                else:
                    keys_data[key]['ps_sig'] = f
            else:
                if sig_type == 'EPSP':
                    keys_data[key]['epsp_amp'] = f
                else:
                    keys_data[key]['ps_amp'] = f

        if not keys_data:
            QMessageBox.warning(self, "Предупреждение", "Не удалось извлечь ключи из имён файлов")
            return

        # Формируем список для комбобокса
        keys_list = sorted(keys_data.keys())
        self.keys_data = keys_data
        self.key_combo.clear()
        for key in keys_list:
            data = keys_data[key]
            has_epsp_s = data['epsp_sig'] is not None
            has_ps_s = data['ps_sig'] is not None
            has_epsp_a = data['epsp_amp'] is not None
            has_ps_a = data['ps_amp'] is not None
            info = []
            if has_epsp_s:
                info.append('EPSP')
            if has_ps_s:
                info.append('PS')
            if has_epsp_a:
                info.append('AmpEPSP')
            if has_ps_a:
                info.append('AmpPS')
            label = f"{key} ({'+'.join(info)})"
            self.key_combo.addItem(label, userData=key)
        self.key_combo.setEnabled(True)

        if keys_list:
            self.plot_key(keys_list[0])

    def on_key_selected(self, index):
        if index < 0:
            return
        key = self.key_combo.itemData(index)
        if key:
            self.plot_key(key)

    def plot_key(self, key):
        data = self.keys_data.get(key)
        if not data:
            return

        self.current_key = key

        # Загружаем сигналы
        self.signal_x_epsp = None
        self.signal_y_epsp = None
        self.signal_x_ps = None
        self.signal_y_ps = None
        self.has_epsp_signal = False
        self.has_ps_signal = False

        if data['epsp_sig']:
            try:
                df = pd.read_csv(data['epsp_sig'], sep=None, engine='python')
                xcol = df.columns[0]
                ycol = df.columns[1] if len(df.columns) > 1 else xcol
                self.signal_x_epsp = df[xcol].values
                self.signal_y_epsp = df[ycol].values
                self.has_epsp_signal = True
            except Exception as e:
                print(f"Ошибка загрузки EPSP сигнала: {e}")

        if data['ps_sig']:
            try:
                df = pd.read_csv(data['ps_sig'], sep=None, engine='python')
                xcol = df.columns[0]
                ycol = df.columns[1] if len(df.columns) > 1 else xcol
                self.signal_x_ps = df[xcol].values
                self.signal_y_ps = df[ycol].values
                self.has_ps_signal = True
            except Exception as e:
                print(f"Ошибка загрузки PS сигнала: {e}")

        # Загружаем амплитуды
        self.amp_x_epsp = None
        self.amp_y_epsp = None
        self.amp_x_ps = None
        self.amp_y_ps = None
        self.has_epsp_amp = False
        self.has_ps_amp = False

        if data['epsp_amp']:
            try:
                df = pd.read_csv(data['epsp_amp'], sep=None, engine='python')
                xcol = df.columns[0]
                ycol = df.columns[1] if len(df.columns) > 1 else xcol
                self.amp_x_epsp = df[xcol].values
                self.amp_y_epsp = df[ycol].values
                self.has_epsp_amp = True
            except Exception as e:
                print(f"Ошибка загрузки AmplitudeEPSP: {e}")

        if data['ps_amp']:
            try:
                df = pd.read_csv(data['ps_amp'], sep=None, engine='python')
                xcol = df.columns[0]
                ycol = df.columns[1] if len(df.columns) > 1 else xcol
                self.amp_x_ps = df[xcol].values
                self.amp_y_ps = df[ycol].values
                self.has_ps_amp = True
            except Exception as e:
                print(f"Ошибка загрузки AmplitudePS: {e}")

        # Отрисовка верхнего графика (сигналы)
        self.canvas_signal.ax.clear()
        if self.has_epsp_signal:
            x, y = insert_nan_at_gaps(self.signal_x_epsp, self.signal_y_epsp)
            self.canvas_signal.ax.plot(x, y, color='blue', linewidth=1.5, label='EPSP')
        if self.has_ps_signal:
            x, y = insert_nan_at_gaps(self.signal_x_ps, self.signal_y_ps)
            self.canvas_signal.ax.plot(x, y, color='red', linewidth=1.5, label='PS')
        if not self.has_epsp_signal and not self.has_ps_signal:
            self.canvas_signal.ax.text(0.5, 0.5, "Нет сигналов",
                                       transform=self.canvas_signal.ax.transAxes,
                                       ha='center', va='center', fontsize=12)
        self.canvas_signal.ax.set_title(f"Сигналы: {key}")
        self.canvas_signal.ax.set_xlabel("Время (мс)")
        self.canvas_signal.ax.set_ylabel("мВ")
        self.canvas_signal.ax.grid(True, linestyle='--', alpha=0.7)
        self.canvas_signal.ax.legend()
        self.canvas_signal.fig.tight_layout()
        self.canvas_signal.draw()
        self.canvas_signal.set_original_limits()

        # Отрисовка нижнего графика (амплитуды)
        self.canvas_amplitude.ax.clear()
        plotted = False
        if self.has_epsp_amp:
            x, y = insert_nan_at_gaps(self.amp_x_epsp, self.amp_y_epsp)
            self.canvas_amplitude.ax.scatter(x, y, color='blue', s=10, alpha=0.8,
                                              label='Amplitude EPSP')
            plotted = True
        if self.has_ps_amp:
            x, y = insert_nan_at_gaps(self.amp_x_ps, self.amp_y_ps)
            self.canvas_amplitude.ax.scatter(x, y, color='red', s=10, alpha=0.8,
                                              label='Amplitude PS')
            plotted = True
        if not plotted:
            self.canvas_amplitude.ax.text(0.5, 0.5, "Нет амплитуд",
                                          transform=self.canvas_amplitude.ax.transAxes,
                                          ha='center', va='center', fontsize=12)
        self.canvas_amplitude.ax.set_title(f"Амплитуды: {key}")
        self.canvas_amplitude.ax.set_xlabel("Время (мс)")
        self.canvas_amplitude.ax.set_ylabel("Амплитуда (мВ)")
        self.canvas_amplitude.ax.grid(True, linestyle='--', alpha=0.7)
        self.canvas_amplitude.ax.legend()
        self.canvas_amplitude.fig.tight_layout()
        self.canvas_amplitude.draw()
        self.canvas_amplitude.set_original_limits()

        # Сбрасываем аннотации
        self.current_vline = None
        self.current_hline = None
        self.current_point = None
        self.last_amp_type = None

    def find_continuous_segment(self, x, t, threshold_factor=10):
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

        # Определяем, к какой амплитуде ближе клик
        best_dist = float('inf')
        best_type = None
        best_idx = None
        best_t = None
        best_amp = None

        if self.has_epsp_amp:
            idx = np.argmin(np.abs(self.amp_x_epsp - t_click))
            dist = abs(self.amp_x_epsp[idx] - t_click)
            if dist < best_dist:
                best_dist = dist
                best_type = 'EPSP'
                best_idx = idx
                best_t = self.amp_x_epsp[idx]
                best_amp = self.amp_y_epsp[idx]

        if self.has_ps_amp:
            idx = np.argmin(np.abs(self.amp_x_ps - t_click))
            dist = abs(self.amp_x_ps[idx] - t_click)
            if dist < best_dist:
                best_dist = dist
                best_type = 'PS'
                best_idx = idx
                best_t = self.amp_x_ps[idx]
                best_amp = self.amp_y_ps[idx]

        if best_type is None:
            return

        t_amp = best_t
        amp_value = best_amp

        # Выбираем соответствующий сигнал
        if best_type == 'EPSP':
            if not self.has_epsp_signal:
                return
            sig_x = self.signal_x_epsp
            sig_y = self.signal_y_epsp
        else:
            if not self.has_ps_signal:
                return
            sig_x = self.signal_x_ps
            sig_y = self.signal_y_ps

        # Проверка на разрыв
        is_cont, idx_closest = self.find_continuous_segment(sig_x, t_amp)
        if not is_cont:
            t_amp = sig_x[idx_closest]

        half_window = DEFAULT_PEAK_WINDOW
        left = t_amp - half_window
        right = t_amp + half_window

        mask = (sig_x >= left) & (sig_x <= right)
        if not np.any(mask):
            idx_sig = np.argmin(np.abs(sig_x - t_amp))
            t_sig = sig_x[idx_sig]
            left = t_sig - half_window
            right = t_sig + half_window
            mask = (sig_x >= left) & (sig_x <= right)

        x_win = sig_x[mask]
        y_win = sig_y[mask]
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

        # Удаляем старые аннотации
        for art in [self.current_vline, self.current_hline, self.current_point]:
            if art is not None:
                art.remove()

        # Вертикальная линия
        self.current_vline = self.canvas_signal.ax.axvline(
            x=t_amp, color='red', linestyle='--', linewidth=1, alpha=0.7,
            label=f'пик при t={t_amp:.2f} мс ({best_type})'
        )
        # Горизонтальная линия на уровне амплитуды
        self.current_hline = self.canvas_signal.ax.axhline(
            y=amp_value, color='green', linestyle=':', linewidth=1, alpha=0.7,
            label=f'амплитуда = {amp_value:.2f} мВ ({best_type})'
        )
        # Интерполированная точка на сигнале
        y_interp = np.interp(t_amp, sig_x, sig_y)
        self.current_point = self.canvas_signal.ax.plot(
            t_amp, y_interp, 'ro', markersize=6, markeredgecolor='darkred',
            label=f'интерп. сигнал ({best_type})'
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