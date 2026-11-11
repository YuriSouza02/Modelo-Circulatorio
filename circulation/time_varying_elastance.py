import math


def blanco_ventricle(
    EA: float = 1.0,
    EB: float = 0.1,
    tC: float = 0.0,
    TC: float = 0.4,
    TR: float = 0.2,
    RR: float = 1.0,
):
    # Cria e retorna uma função que calcula a elastância do ventrículo variando no tempo.
    time_R = tC + TC
    pi_TC = math.pi / TC
    pi_TR = math.pi / TR

    def elastance(t: float) -> float:
        # Calcula a elastância em um dado instante de tempo considerando as fases do ciclo cardíaco.
        t_mod = t % RR

        if t_mod >= tC and t_mod < time_R:
            e = 0.5 * (1.0 - math.cos(pi_TC * (t_mod - tC)))
        elif t_mod >= time_R and t_mod < (time_R + TR):
            e = 0.5 * (1.0 + math.cos(pi_TR * (t_mod - time_R)))
        else:
            e = 0.0

        if e > 1.0:
            e = 1.0
        elif e < 0.0:
            e = 0.0

        return EA * e + EB

    return elastance