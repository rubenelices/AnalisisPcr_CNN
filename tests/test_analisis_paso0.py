import numpy as np

from src import analisis_paso0 as A


def test_realce_centrado_queda_dentro_y_periferico_fuera():
    r = np.zeros((256, 256)); r[120:136, 120:136] = 1
    assert A.fracciones_dentro(r)[128] == 1.0
    r = np.zeros((256, 256)); r[0:10, 0:10] = 1
    assert A.fracciones_dentro(r)[192] == 0.0
    cy, cx = A.centro_de_masas(np.ones((256, 256)))
    assert abs(cy - 127.5) < 1e-6 and abs(cx - 127.5) < 1e-6


def test_razones_de_oclusion():
    pre = np.ones((256, 256), dtype=np.float32)
    uniforme = np.ones((16, 16))
    r_perif, r_borde, _ = A.razones(uniforme, pre, 128)
    assert abs(r_perif - 1.0) < 1e-6 and np.isnan(r_borde)          # sin aire no hay borde
    solo_fuera = np.ones((16, 16)); solo_fuera[4:12, 4:12] = 0
    r_perif, _, _ = A.razones(solo_fuera, pre, 128)
    assert abs(r_perif - 1 / 0.75) < 1e-6                              # todo el efecto en el 75 % exterior
    pre[:, :120] = 0                                                   # mitad izquierda es aire
    borde = np.zeros((16, 16)); borde[:, 7] = 1                        # columna de cuadrados mezclados
    _, r_borde, _ = A.razones(borde, pre, 128)
    assert r_borde > 1.5


def test_variables_de_realce_relativo_y_absoluto():
    from src.analisis_paso0b import variables_corte
    pre = np.full((64, 64), 0.2, dtype=np.float32)
    early = pre.copy(); late = pre.copy()
    early[:16] = 0.4      # PE = 1,0 en una cuarta parte del tejido
    late[:16] = 0.3       # SER = 0,2 / 0,1 = 2 > 1,1: lavado
    v = variables_corte(pre, early, late)
    assert abs(v['frac_pe70'] - 0.25) < 1e-6 and v['frac_lavado'] == 1.0
    assert abs(v['frac_abs10'] - 0.25) < 1e-6 and v['frac_abs_lavado'] == 1.0
    # Doblar la ganancia del escáner no cambia las relativas y sí las absolutas.
    w = variables_corte(pre * 2, early * 2, late * 2)
    assert abs(w['frac_pe70'] - v['frac_pe70']) < 1e-6 and abs(w['pe_top5'] - v['pe_top5']) < 1e-6
    assert w['abs_top5'] > v['abs_top5']


def test_region_candidata_encuentra_el_foco_de_realce():
    from src.analisis_paso0c import region_candidata
    pre = np.full((256, 256), 0.2, dtype=np.float32)
    early = pre.copy(); late = pre.copy()
    early[150:170, 90:110] = 0.5      # foco con PE = 1,5
    late[150:170, 90:110] = 0.4       # lavado: SER = 0,3 / 0,2 = 1,5
    (y, x), v = region_candidata(pre, early, late)
    assert 150 <= y < 170 and 90 <= x < 110
    assert v['frac_pe70'] > 0 and v['frac_lavado'] == 1.0 and v['tamano'] == 400
