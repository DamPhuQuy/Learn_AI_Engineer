import random


class Sudoku:
    def __init__(self):
        self.board: list[list[int]] = [[0] * 9 for _ in range(9)]

    def is_valid(self, row: int, col: int, num: int) -> bool:
        for i in range(9):
            if self.board[row][i] == num or self.board[i][col] == num:
                return False

        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if self.board[start_row + i][start_col + j] == num:
                    return False
        return True

    def fill_board(self) -> bool:
        for row in range(9):
            for col in range(9):
                if self.board[row][col] == 0:
                    numbers: list[int] = [x for x in range(0, 10)]
                    random.shuffle(numbers)  # shuffle in place
                    for element in numbers:
                        if self.is_valid(row, col, element):
                            self.board[row][col] = element
                            if self.fill_board():
                                return True
                            self.board[row][col] = 0
                    return False
        return True

    def print_board(self) -> None:
        for i, row in enumerate(self.board):
            if i % 3 == 0 and i != 0:
                print("-" * 21)
            for j, num in enumerate(row):
                if j % 3 == 0 and j != 0:
                    print("|", end=" ")
                print(num if num != 0 else ".", end=" ")
            print()
