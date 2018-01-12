import math
import numpy as np


class MCTS:
    """
    Класс для метода поиска Монте - Крало в дереве.

    Принцип работы:
        На всякой итерации алгоритма, чтобы оценить позицию, запускаются случайные симуляции,
        начинающиеся с этой позиции, смотрится, в каких ветвях первый игрок или второй игрок
        выиграли больше случайных партий, а затем рекурсивно повторяется этот поиск в самых
        перспективных узлах получающегося дерева.
    """
    def __init__(self, game, nnet, args):
        self.game = game    # Игра
        self.nnet = nnet    # Сеть
        self.args = args    # Словарь, состоящий из числа симуляций для Монте - Карло и значения констаты c
        self.Qsa = {}       # Ожидаемая награда за применения действия а из состояния доски s
        self.Nsa = {}       # Количество посещений ребра (s, a)
        self.Ns = {}        # Количество посещений состояния доски s
        self.Ps = {}        # Априорная вероятность применение действия a из состояния доски s
        self.Es = {}        # Конечные состояния игры для состояния доски s (game.getGameEnded())
        self.Vs = {}        # Допустимые ходы для состояния доски s (game.getValidMoves())

    def getActionProb(self, canonicalBoard, temp=1):
        """
        Выполняет симуляции метода Монте - Карло, количество которых равно значению numMCTSSims из словаря self.args,
        начиная из состояния canonicalBoard.

        Возвращает:
            probs: вектор вероятностей (политик),
            в которых вероятность i-ого действия пропорциональна величине Nsa([s, a])^(1./temp)

        Замечание:
            temp - гиперпараметр, контролирующий степень исследования при обучении.
            Если параметр temp = 0, то мы получаем наилучшее действие а. Иначе - равномерное распределение.
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1./temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        Метод выполняет одну итерацию метода Монте - Карло. Он рекурсивно вызывается до тех пор,
        пока не будет найдена всячая вершина (лист). Действие, выбираемое в каждой вершине, имеет
        максимальное значение Upper Confidence Bound (UCB).

        Upper Confidence Bound подсчитывается по следующей формуле:

            U(s, a) = Q(s, a) + cpuct * Ps(s, a) * (sqrt(sum_b(N(s, b))) / (1 + N(s, a))

        где sum_b(N(s, b)) - это сумма значений N(s, b) по всем вершинам b.

        Когда будет найден лист, вызывается нейронная сеть, возвращающая вектор политик (вероятностей) P и
        величину v, которая является оценкой вероятности победы текущего игрока из заданного состояния доски.
        Эта величина распространяется обратно вверх по пути поиска. Если лист - это терминальное состояние,
        то весь исход распространяется обратно вверх по пути поиска. Значения Ns, Nsa, Qas обновляются.

        Замечание:
            Возвращаемые значения являются отрицательными значениями текущего состояния. Это так,
            потому что величина v принадлежит отрезку [-1, 1], и если v - это величина для состояния текущего игрока,
            то величиной для состояния другого игрока будет -v, поскольку игра антогонистическая.

        Возвращает:
            v: отрицательная оценка для текущей canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]      # терминальная вершина

        if s not in self.Ps:
            # лист
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids    # маскирование недопустимых ходов
            self.Ps[s] /= np.sum(self.Ps[s])    # ренормализация
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # Выбираем действие с максимальным значением UCB
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] \
                        * math.sqrt(self.Ns[s]) / (1+self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s])
                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)]+1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1

        return -v
