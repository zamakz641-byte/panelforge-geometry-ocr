from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import traceback

import webview

from engine import process_source

APP_DIR = Path(__file__).resolve().parent
UI_FILE = APP_DIR / 'ui' / 'index.html'


class Api:
    def __init__(self):
        self._window = None
        self.source_path = None
        self.source_type = None
        self.running = False
        self.last_output = None

    def _set_window(self, window):
        self._window = window

    def _emit(self, event: str, data: dict):
        if not self._window:
            return
        payload = json.dumps(data, ensure_ascii=False)
        try:
            self._window.evaluate_js(f"window.PanelForge && window.PanelForge.onPythonEvent({json.dumps(event)}, {payload});")
        except Exception:
            pass

    def _dialog_open_cbz(self):
        # Support both modern and legacy pywebview APIs.
        try:
            dialog_type = getattr(getattr(webview, 'FileDialog', None), 'OPEN', None)
            if dialog_type is not None:
                return self._window.create_file_dialog(dialog_type, allow_multiple=False, file_types=('CBZ (*.cbz)', 'All files (*.*)'))
        except Exception:
            pass
        try:
            return self._window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=('CBZ (*.cbz)', 'All files (*.*)'))
        except Exception:
            return None

    def _dialog_folder(self):
        try:
            dialog_type = getattr(getattr(webview, 'FileDialog', None), 'FOLDER', None)
            if dialog_type is not None:
                return self._window.create_file_dialog(dialog_type)
        except Exception:
            pass
        try:
            return self._window.create_file_dialog(webview.FOLDER_DIALOG)
        except Exception:
            return None

    def choose_cbz(self):
        result = self._dialog_open_cbz()
        if not result:
            return {'ok': False}
        path = result[0] if isinstance(result, (list, tuple)) else result
        if not path:
            return {'ok': False}
        self.source_path = str(Path(path))
        self.source_type = 'cbz'
        self.last_output = None
        return {
            'ok': True,
            'type': 'cbz',
            'path': self.source_path,
            'name': Path(self.source_path).name,
            'output_hint': str(Path(self.source_path).parent / f'{Path(self.source_path).stem} output')
        }

    def choose_folder(self):
        result = self._dialog_folder()
        if not result:
            return {'ok': False}
        path = result[0] if isinstance(result, (list, tuple)) else result
        if not path:
            return {'ok': False}
        self.source_path = str(Path(path))
        self.source_type = 'folder'
        self.last_output = None
        return {
            'ok': True,
            'type': 'folder',
            'path': self.source_path,
            'name': Path(self.source_path).name or self.source_path,
            'output_hint': str(Path(self.source_path) / 'PanelCrops_Geometry')
        }

    def start_processing(self, options=None):
        if self.running:
            return {'ok': False, 'error': 'Un traitement est déjà en cours.'}
        if not self.source_path or not self.source_type:
            return {'ok': False, 'error': 'Choisis d’abord un CBZ ou un dossier.'}
        options = options or {}
        try:
            analysis_width = int(options.get('analysis_width', 192))
        except Exception:
            analysis_width = 192
        save_debug = bool(options.get('save_debug', True))
        use_ocr = bool(options.get('use_ocr', True))
        try:
            ocr_width = int(options.get('ocr_width', 768))
        except Exception:
            ocr_width = 768
        if ocr_width not in (512, 640, 768, 896, 1024, 1152):
            ocr_width = 768

        self.running = True
        thread = threading.Thread(
            target=self._worker,
            args=(analysis_width, save_debug, use_ocr, ocr_width),
            daemon=True,
        )
        thread.start()
        return {'ok': True}

    def _worker(self, analysis_width: int, save_debug: bool, use_ocr: bool, ocr_width: int):
        try:
            self._emit('state', {'running': True})

            def progress(payload):
                self._emit('progress', payload)

            report = process_source(
                self.source_path,
                self.source_type,
                analysis_width=analysis_width,
                save_debug=save_debug,
                use_ocr=use_ocr,
                ocr_width=ocr_width,
                progress=progress,
            )
            self.last_output = report.get('output_dir')
            self._emit('complete', report)
        except Exception as exc:
            self._emit('error', {
                'message': str(exc),
                'details': traceback.format_exc(limit=8),
            })
        finally:
            self.running = False
            self._emit('state', {'running': False})

    def open_output(self):
        path = self.last_output
        if not path or not Path(path).exists():
            return {'ok': False, 'error': 'Aucun dossier de sortie disponible.'}
        try:
            if os.name == 'nt':
                os.startfile(path)  # type: ignore[attr-defined]
            elif os.uname().sysname == 'Darwin':
                os.system(f'open {json.dumps(path)}')
            else:
                os.system(f'xdg-open {json.dumps(path)} >/dev/null 2>&1 &')
            return {'ok': True}
        except Exception as exc:
            return {'ok': False, 'error': str(exc)}

    def get_state(self):
        return {
            'source_path': self.source_path,
            'source_type': self.source_type,
            'running': self.running,
            'last_output': self.last_output,
        }


def main():
    api = Api()
    window = webview.create_window(
        'PanelForge Geometry + OCR',
        url=UI_FILE.as_uri(),
        js_api=api,
        width=1080,
        height=760,
        min_size=(900, 650),
        background_color='#0b0d12',
    )
    api._set_window(window)
    webview.start(debug=False)


if __name__ == '__main__':
    main()
