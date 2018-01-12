class Board:
    def __init__(self, n):
        # Размер игры . . .
        self.n = n
        # Создаем списковое представление пустой доски
        self.pieces = [None] * self.n
        for i in range(self.n):
            self.pieces[i] = [0] * self.n

    # Добавляем возможность использования оператора взятия индекса ([] и [][])
    def __getitem__(self, index):
        return self.pieces[index]

    def get_count(self, color):
        """
        :param color: <Int>
                      "цвет" игроков: 1 для крестиков, -1 для ноликов, 0 для незанятых клеток.

        :return: count: <Int>
                        количество фигур типа color.

        Подсчитывает количество фигур типа color на доске.
        """
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    count += 1
        return count

    def count_diff(self, color):
        """
        :param color: <Int>
                      "цвет" игроков: 1 для крестиков, -1 для ноликов, 0 для незанятых клеток.

        :return: count: <Int>
                        Разница между количеством фигур типа color и -color на доске.

        """
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    count += 1
                if self[x][y] == -color:
                    count -= 1
        return count

    def get_moves_for_square(self):
        """
        :param square: tuple(<Int>, <Int>)
                       координаты клетки, для которой рассчитываются возможные ходы

        :return: moves: <List of tuples(<Int>, <Int>)>
                        список возможных ходов для клетки square

        Ищем возможные ходы для квадрата square.
        Возможным ходом является любая незанятая клетка.
        Такие и ищем . . .
        """
        moves = []
        for i in range(self.n):
            for j in range(self.n):
                if self[j][i] == 0:
                    move = (j, i)
                    moves.append(move)
        return moves

    def execute_move(self, move, color):
        """
        :param move: tuple(<Int>, <Int>), кортеж, представляющий текущий ход
        :param color: <Int>, 1 для крестиков, -1 для ноликов
        Ставит в клетку с координатами move фигуру color.
        """
        (x, y) = move
        self[x][y] = color

    def has_legal_moves(self):
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    new_moves = self.get_moves_for_square()
                    if len(new_moves) > 0:
                        return True
        return False

    def get_legal_moves(self):
        moves = set()
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    new_moves = self.get_moves_for_square()
                    if new_moves:
                        moves.update(new_moves)
        return list(moves)

    def print_board(self):
        for piece in self.pieces:
            print(piece)
