import torch

from src.web.web_aumento import a_realce, variantes


def _imagen():
    torch.manual_seed(0)
    pre = torch.rand(256, 256) * 0.4
    return torch.stack([pre, pre + torch.rand(256, 256) * 0.3, pre + torch.rand(256, 256) * 0.4]).float()


def test_los_aumentos_geometricos_mueven_las_tres_fases_a_la_vez():
    img = _imagen()
    v = variantes(img)
    realce = lambda x: x[1] - x[0]
    # Al ser una operación común, el realce del aumentado es el realce aumentado (espejo: exacto).
    assert torch.equal(realce(v['espejo_horizontal']), torch.flip(realce(img), dims=(-1,)))
    assert torch.equal(realce(v['espejo_vertical']), torch.flip(realce(img), dims=(-2,)))
    assert all(x.shape == (3, 256, 256) for x in v.values())


def test_ganancia_compartida_conserva_el_realce_escalado_y_la_de_por_fase_lo_falsea():
    img = _imagen()
    v = variantes(img)
    assert torch.allclose(v['ganancia'][1] - v['ganancia'][0], 1.1 * (img[1] - img[0]), atol=1e-6)
    assert not torch.allclose(v['ganancia_por_fase'][1] - v['ganancia_por_fase'][0], img[1] - img[0], atol=1e-3)


def test_realce_es_una_imagen_rgb_de_256():
    im = a_realce(_imagen())
    assert im.size == (256, 256) and im.mode == 'RGB'
