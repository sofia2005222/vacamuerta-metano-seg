import torch

from vmseg.perdidas import PERDIDAS, PerdidaSegmentacion
from vmseg.recorte import SIN_ETIQUETA


def _mascara():
    target = torch.zeros(2, 32, 32, dtype=torch.long)
    target[:, 10:15, 10:15] = 1             # una locación
    target[0, :2] = SIN_ETIQUETA            # una franja sin etiquetar
    return target


def test_todas_las_perdidas_dan_gradientes():
    torch.manual_seed(0)
    logits = torch.randn(2, 2, 32, 32, requires_grad=True)
    for nombre in PERDIDAS:
        perdida = PerdidaSegmentacion(nombre, pesos_clases=[1.0, 10.0])(logits, _mascara())
        perdida.backward()
        assert torch.isfinite(perdida) and torch.isfinite(logits.grad).all()
        logits.grad = None


def test_prediccion_perfecta_da_perdida_casi_cero():
    target = _mascara()
    # logits muy seguros de la clase correcta
    logits = torch.stack([(target != 1).float(), (target == 1).float()], dim=1) * 20
    for nombre in PERDIDAS:
        assert PerdidaSegmentacion(nombre)(logits, target) < 0.01