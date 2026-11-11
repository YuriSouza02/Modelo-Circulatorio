from __future__ import annotations
from typing import Callable, Any
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import time
import logging

from . import time_varying_elastance

logger = logging.getLogger(__name__)


def smooth_heavyside(x: float) -> float:
    # Calcula uma aproximação suave da função degrau de Heaviside.
    return np.arctan(np.pi / 2 * x * 200) * 1 / np.pi + 0.5


def deep_update(d: dict, u: dict) -> dict:
    # Atualiza recursivamente um dicionário com os valores de outro dicionário.
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class CirculationModel(ABC):
    def __init__(
        self,
        parameters: dict[str, Any] | None = None,
        outdir: Path = Path("results"),
        verbose: bool = False,
        initial_state: dict[str, float] | None = None,
        **kwargs,
    ):
        # Inicializa o modelo configurando parâmetros, estados iniciais e arrays para integração.
        self.parameters = type(self).default_parameters()
        if parameters is not None:
            self.parameters = deep_update(self.parameters, parameters)

        self._initial_state = type(self).default_initial_conditions()
        if initial_state is not None:
            self._initial_state.update(initial_state)

        self.state = np.array(list(self._initial_state.values()), dtype=np.float64)
        self.state_old = np.copy(self.state)
        self.dy = np.zeros_like(self.state)
        self.outdir = outdir

    @property
    def states_names(self) -> list[str]:
        # Retorna a lista com os nomes das variáveis de estado do modelo.
        return list(self._initial_state.keys())

    @property
    def num_states(self) -> int:
        # Retorna a quantidade total de variáveis de estado do modelo.
        return len(self.state)

    @staticmethod
    @abstractmethod
    def var_names() -> list[str]:
        # Retorna a lista de nomes das variáveis auxiliares calculadas durante a simulação.
        return []

    @abstractmethod
    def rhs(self, t: float, y: npt.NDArray) -> npt.NDArray:
        # Calcula as derivadas do sistema (lado direito da EDO) para o passo de tempo atual.
        ...

    @property
    def num_vars(self) -> int:
        # Retorna o número total de variáveis auxiliares do modelo.
        return len(type(self).var_names())

    def _initialize(self):
        # Aloca os arrays na memória para armazenar as variáveis auxiliares atuais e anteriores.
        self.var = np.zeros(self.num_vars, dtype=np.float64)
        self.var_old = np.copy(self.var)

    def initialize_results(self):
        # Prepara as matrizes de histórico para salvar os resultados dos estados e variáveis ao longo do tempo.
        N = len(self.times_eval)
        self.results_state = np.zeros((self.num_states, N))
        self.results_state[:, 0] = self.state

        self.rhs(0.0, self.state)

        self.results_var = np.zeros((self.num_vars, N))
        self.results_var[:, 0] = self.var
        self._index = 1

    @property
    @abstractmethod
    def HR(self) -> float:
        # Retorna a frequência cardíaca (Heart Rate) do modelo.
        ...

    @staticmethod
    @abstractmethod
    def default_parameters() -> dict[str, Any]:
        # Retorna o dicionário de parâmetros estruturais padrão do sistema cardiovascular.
        ...

    @staticmethod
    @abstractmethod
    def default_initial_conditions() -> dict[str, float]:
        # Retorna o dicionário com os valores iniciais padrão para os estados do sistema.
        ...

    def time_varying_elastance(self, EA, EB, tC, TC, TR, **kwargs):
        # Instancia a função de elastância variante no tempo baseada no modelo de ventrículo de Blanco.
        return time_varying_elastance.blanco_ventricle(
            EA=EA, EB=EB, tC=tC, TC=TC, TR=TR, RR=1 / self.HR
        )

    def flux_through_valve(
        self, p1: float, p2: float, R: Callable[[float, float], float]
    ) -> float:
        # Calcula o fluxo de sangue através de uma válvula usando a diferença de pressão e sua resistência.
        return (p1 - p2) / R(p1, p2)

    def _R(self, Rmin: float, Rmax: float) -> Callable[[float, float], float]:
        # Cria a função que calcula dinamicamente a resistência variável da válvula com base na pressão.
        log_Rmin = np.log10(Rmin)
        delta_log_R = np.log10(Rmax) - log_Rmin
        return lambda w, v: 10.0 ** (log_Rmin + delta_log_R * smooth_heavyside(v - w))

    def times_n_beats(self, dt: float, n: int = 1) -> np.ndarray:
        # Gera o array de passos de tempo iterativos para simular um número específico de batimentos.
        return np.arange(0, n / self.HR, dt)

    def solve(
        self,
        num_beats: int | None = None,
        dt: float = 1e-4,
        dt_eval: float | None = None,
        **kwargs,
    ):
        # Executa a integração numérica iterativa do modelo no tempo para o número de batimentos especificado.
        if dt_eval is None:
            dt_eval = dt
        output_every_n_steps = int(np.round(dt_eval / dt))

        if num_beats is None:
            raise ValueError("num_beats precisa ser especificado.")

        times_one_beat = self.times_n_beats(dt, n=1)
        T = (times_one_beat[-1] + dt) * num_beats

        N = (
            sum(
                1
                for _ in range(num_beats)
                for i in range(len(times_one_beat))
                if i % output_every_n_steps == 0
            )
            + 1
        )

        self.times_eval = np.linspace(0, T, N)
        self.initialize_results()

        time_start = time.time()

        for beat in range(num_beats):
            print(f"Processando batimento {beat + 1} de {num_beats}...")
            for i, t in enumerate(times_one_beat):
                ti = t
                self.state_old[:] = self.state.copy()

                dy = self.rhs(ti, self.state)
                self.state += dt * dy

                if i % output_every_n_steps == 0:
                    self.store()

        duration = time.time() - time_start
        logger.info(f"Modelo resolvido em {duration:.2f} s")

        return self.history

    @property
    def history(self) -> dict[str, np.ndarray]:
        # Consolida e retorna o histórico mapeado de tempo, estados e variáveis auxiliares registrados na simulação.
        history = {}
        for i, name in enumerate(self.states_names):
            history[name] = self.results_state[i, :]
        for i, name in enumerate(type(self).var_names()):
            history[name] = self.results_var[i, :]
        history["time"] = self.times_eval
        return history

    def store(self):
        # Salva o estado atual e as variáveis auxiliares da iteração corrente nas matrizes de resultado.
        try:
            self.results_state[:, self._index] = self.state[:]
            self.results_var[:, self._index] = self.var[:]
        except IndexError:
            pass
        finally:
            self._index += 1
