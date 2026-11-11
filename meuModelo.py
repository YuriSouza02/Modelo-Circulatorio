import numpy as np
from circulation.base import CirculationModel


class MeuModeloCardiovascular(CirculationModel):
    def __init__(self, *args, **kwargs):
        # Inicializa o modelo cardiovascular configurando resistências das válvulas, elastâncias das câmaras e parâmetros da circulação.
        super().__init__(*args, **kwargs)

        p_valves = self.parameters["valves"]
        self.R_MV = self._R(p_valves["MV"]["Rmin"], p_valves["MV"]["Rmax"])
        self.R_AV = self._R(p_valves["AV"]["Rmin"], p_valves["AV"]["Rmax"])
        self.R_TV = self._R(p_valves["TV"]["Rmin"], p_valves["TV"]["Rmax"])
        self.R_PV = self._R(p_valves["PV"]["Rmin"], p_valves["PV"]["Rmax"])

        ch = self.parameters["chambers"]
        self._E_LA = self.time_varying_elastance(**ch["LA"])
        self._E_LV = self.time_varying_elastance(**ch["LV"])
        self._E_RA = self.time_varying_elastance(**ch["RA"])
        self._E_RV = self.time_varying_elastance(**ch["RV"])

        self.V0_LA = ch["LA"]["V0"]
        self.V0_LV = ch["LV"]["V0"]
        self.V0_RA = ch["RA"]["V0"]
        self.V0_RV = ch["RV"]["V0"]

        circ = self.parameters["circulation"]
        self.C_AR_SYS = circ["SYS"]["C_AR"]
        self.C_VEN_SYS = circ["SYS"]["C_VEN"]
        self.C_AR_PUL = circ["PUL"]["C_AR"]
        self.C_VEN_PUL = circ["PUL"]["C_VEN"]

        self.R_AR_SYS = circ["SYS"]["R_AR"]
        self.R_VEN_SYS = circ["SYS"]["R_VEN"]
        self.R_AR_PUL = circ["PUL"]["R_AR"]
        self.R_VEN_PUL = circ["PUL"]["R_VEN"]

        self.L_AR_SYS = circ["SYS"]["L_AR"]
        self.L_VEN_SYS = circ["SYS"]["L_VEN"]
        self.L_AR_PUL = circ["PUL"]["L_AR"]
        self.L_VEN_PUL = circ["PUL"]["L_VEN"]

        self._initialize()

    @property
    def HR(self):
        # Retorna a frequência cardíaca (Heart Rate) configurada nos parâmetros do modelo.
        return self.parameters["HR"]

    @staticmethod
    def default_parameters():
        # Fornece os parâmetros fisiológicos e estruturais padrão para a simulação cardiovascular.
        return {
            "HR": 1.25,
            "chambers": {
                "LA": {
                    "EA": 0.07,
                    "EB": 0.18,
                    "TC": 0.17,
                    "TR": 0.17,
                    "tC": 0.9,
                    "V0": 4.0,
                },
                "LV": {
                    "EA": 4.482,
                    "EB": 0.170,
                    "TC": 0.25,
                    "TR": 0.40,
                    "tC": 0.10,
                    "V0": 42.0,
                },
                "RA": {
                    "EA": 0.06,
                    "EB": 0.07,
                    "TC": 0.17,
                    "TR": 0.17,
                    "tC": 0.9,
                    "V0": 4.0,
                },
                "RV": {
                    "EA": 0.200,
                    "EB": 0.029,
                    "TC": 0.25,
                    "TR": 0.40,
                    "tC": 0.10,
                    "V0": 16.0,
                },
            },
            "valves": {
                "MV": {"Rmin": 0.0075, "Rmax": 75006.2},
                "AV": {"Rmin": 0.0075, "Rmax": 75006.2},
                "TV": {"Rmin": 0.0075, "Rmax": 75006.2},
                "PV": {"Rmin": 0.0075, "Rmax": 75006.2},
            },
            "circulation": {
                "SYS": {
                    "R_AR": 0.733,
                    "C_AR": 1.372,
                    "R_VEN": 0.32,
                    "C_VEN": 11.363,
                    "L_AR": 0.005,
                    "L_VEN": 0.0005,
                },
                "PUL": {
                    "R_AR": 0.046,
                    "C_AR": 20.0,
                    "R_VEN": 0.0015,
                    "C_VEN": 16.0,
                    "L_AR": 0.0005,
                    "L_VEN": 0.0005,
                },
            },
        }

    @staticmethod
    def default_initial_conditions():
        # Fornece os valores iniciais padrão para as variáveis de estado, como volumes, pressões e fluxos.
        return {
            "V_LA": 87.183,
            "V_LV": 118.520,
            "V_RA": 86.833,
            "V_RV": 166.177,
            "p_AR_SYS": 87.675,
            "p_VEN_SYS": 35.898,
            "p_AR_PUL": 19.545,
            "p_VEN_PUL": 15.004,
            "Q_AR_SYS": 71.104,
            "Q_VEN_SYS": 94.039,
            "Q_AR_PUL": 94.084,
            "Q_VEN_PUL": 473.279,
        }

    @staticmethod
    def var_names():
        # Retorna a lista de nomes das variáveis auxiliares registradas dinamicamente durante a integração.
        return ["p_LA", "p_LV", "p_RA", "p_RV", "Q_MV", "Q_AV", "Q_TV", "Q_PV", "I_ext"]

    def rhs(self, t, y):
        # Calcula as derivadas temporais do sistema de EDOs e atualiza as variáveis auxiliares para o passo atual.
        V_LA, V_LV, V_RA, V_RV = y[0], y[1], y[2], y[3]
        p_AR_SYS, p_VEN_SYS, p_AR_PUL, p_VEN_PUL = y[4], y[5], y[6], y[7]
        Q_AR_SYS, Q_VEN_SYS, Q_AR_PUL, Q_VEN_PUL = y[8], y[9], y[10], y[11]

        p_LA = self._E_LA(t) * (V_LA - self.V0_LA)
        p_LV = self._E_LV(t) * (V_LV - self.V0_LV)
        p_RA = self._E_RA(t) * (V_RA - self.V0_RA)
        p_RV = self._E_RV(t) * (V_RV - self.V0_RV)

        Q_MV = self.flux_through_valve(p_LA, p_LV, self.R_MV)
        Q_AV = self.flux_through_valve(p_LV, p_AR_SYS, self.R_AV)
        Q_TV = self.flux_through_valve(p_RA, p_RV, self.R_TV)
        Q_PV = self.flux_through_valve(p_RV, p_AR_PUL, self.R_PV)

        I_ext = 0.0

        self.var[0] = p_LA
        self.var[1] = p_LV
        self.var[2] = p_RA
        self.var[3] = p_RV
        self.var[4] = Q_MV
        self.var[5] = Q_AV
        self.var[6] = Q_TV
        self.var[7] = Q_PV
        self.var[8] = I_ext

        self.dy[0] = Q_VEN_PUL - Q_MV
        self.dy[1] = Q_MV - Q_AV
        self.dy[2] = Q_VEN_SYS - Q_TV
        self.dy[3] = Q_TV - Q_PV
        self.dy[4] = (Q_AV - Q_AR_SYS) / self.C_AR_SYS
        self.dy[5] = (Q_AR_SYS - Q_VEN_SYS + I_ext) / self.C_VEN_SYS
        self.dy[6] = (Q_PV - Q_AR_PUL) / self.C_AR_PUL
        self.dy[7] = (Q_AR_PUL - Q_VEN_PUL) / self.C_VEN_PUL
        self.dy[8] = -(
            (self.R_AR_SYS * Q_AR_SYS + p_VEN_SYS - p_AR_SYS) / self.L_AR_SYS
        )
        self.dy[9] = -((self.R_VEN_SYS * Q_VEN_SYS + p_RA - p_VEN_SYS) / self.L_VEN_SYS)
        self.dy[10] = -(
            (self.R_AR_PUL * Q_AR_PUL + p_VEN_PUL - p_AR_PUL) / self.L_AR_PUL
        )
        self.dy[11] = -(
            (self.R_VEN_PUL * Q_VEN_PUL + p_LA - p_VEN_PUL) / self.L_VEN_PUL
        )

        return self.dy
