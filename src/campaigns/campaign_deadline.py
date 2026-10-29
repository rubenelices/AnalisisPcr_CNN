"""Guardia de plazo absoluto: reconsulta el reloj tras cada espera corta."""
import math
import threading

class WallClockDeadline:
    def __init__(self, remaining_seconds, callback, interval=1.0):
        if not callable(remaining_seconds) or not callable(callback):
            raise ValueError('Reloj y callback deben ser funciones')
        if type(interval) not in (int,float) or not math.isfinite(interval) or not 0<interval<=1:
            raise ValueError('Intervalo finito entre0y1segundo')
        self.remaining_seconds=remaining_seconds
        self.callback=callback
        self.interval=float(interval)
        self.stop=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True)

    def _run(self):
        while not self.stop.is_set():
            seconds=self.remaining_seconds()
            if seconds<=0:
                if not self.stop.is_set():self.callback()
                return
            if self.stop.wait(min(self.interval,seconds)):return

    def start(self):
        self.thread.start()

    def cancel(self):
        self.stop.set()
