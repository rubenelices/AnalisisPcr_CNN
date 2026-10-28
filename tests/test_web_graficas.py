import numpy as np

from src import web_graficas as W


def test_reducir_conserva_extremos_y_limita_puntos():
    x = np.linspace(0, 1, 1000)
    puntos = W.reducir(x, x ** 2, n=50)
    assert len(puntos) <= 51
    assert puntos[0] == [0.0, 0.0] and puntos[-1] == [1.0, 1.0]


def test_curvas_y_calibracion_con_datos_sinteticos():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.3 + 0.2 * y + rng.normal(0, 0.1, 200), 0, 1)
    c = W.curvas(y, p)
    assert c['n'] == 200 and c['pcr'] == int(y.sum())
    assert 0.5 < c['auc'] <= 1 and c['roc'][0] == [0.0, 0.0]
    cal = W.calibracion(y, p)
    assert sum(n for _, _, n in cal) == 200
    h = W.histograma(y, p, 0.325)
    assert sum(h['sin_pcr']) + sum(h['pcr']) == 200
