class NeuralNet:
    """
    Класс, описывающий базовый класс для нейронной сети. Для описания своей собственной
    нейронной сети, унаследуйте этот класс и опишите функции, данные ниже. Нейронная сеть
    не рассматривает текущего игрока. Вместо этого она имеет дело только с канонической
    формой доски.

    Примеры:
      othello/tensorflow/NNet.py
      othello/tensorflow_new_model/NNet.py
      tic_tac_toe/tensorflow/NNet.py
    """
    def __init__(self, game):
        pass

    def train(self, examples):
        """
        Данный метод тренирует нейронную сеть на данных, полученных из игр с самой собой.

        Input:
            examples:
                Список тренировочных примеров, где каждый пример имеет форму
                (board, pi, v). pi - это вектор вероятностей (политик) для
                метода Монте - Карло для данной доски (board), а v - оценка
                вероятности победы на данной доске. Параметр board в примерах
                является доской в канонической форме.
        """
        pass

    def predict(self, board):
        """
        Input:
            board: Текущая доска в своей канонической форме.

        Returns:
            pi:
                Вектор вероятностей (политик) для текущий доски - массив numpy
                длины game.getActionSize().

            v:
                Вещественное число из [-1,1], дающее оценку вероятности победы для
                текущей доски.
        """
        pass

    def save_checkpoint(self, folder, filename):
        """
        Метод сохраняет текущую нейронную сеть вместе с ее параметрами в
        folder/filename
        """
        pass

    def load_checkpoint(self, folder, filename):
        """
        Загружает нейронную сеть из folder/filename
        """
        pass
