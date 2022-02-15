import pygame
import sys, os
import numpy as np
import random

#INITIALIZE
class CreateConfig:
    def __init__(self, rows, columns, inarow) -> None:
        self.rows, self.columns, self.inarow = rows, columns, inarow

class CreateObs:
    def __init__(self, board, mark) -> None:
        self.board, self.mark, self.backup = board, mark, board

    def change_turn(self) -> None:
        self.mark = self.mark%2 + 1
    
    def update_board(self, grid) -> None:
        self.backup = self.board
        self.board = list(grid.reshape(-1))

#METHODS
def get_grid(obs, config):
    board = np.array(obs.board)
    return board.reshape(config.rows, config.columns)

def drop_piece(grid, col, piece, config):
    next_grid = grid.copy()
    for row in range(config.rows-1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = piece
    return next_grid

def print_board(obs, config, screen):
    grid = get_grid(obs, config)
    for row in range(config.rows):
        for piece in range(config.columns):
            if grid[row][piece] == 1:
                width_center  = (2*piece + 1)/2 * WIDTH_UNIT
                height_center = (2*row + 1)/2   * (HEIGHT_UNIT) +4
                pygame.draw.circle(screen, 'red', (width_center, height_center), RADIUS)
            
            if grid[row][piece] == 2:
                width_center  = (2*piece + 1)/2 * WIDTH_UNIT
                height_center = (2*row + 1)/2   * (HEIGHT_UNIT) +4
                pygame.draw.circle(screen, 'yellow', (width_center, height_center), RADIUS)


#AGENTS
def check_window(window, num_discs, piece, config):
    return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)

def count_windows(grid, config):
    num_windows = {'2_1': 0, '3_1': 0, '4_1': 0, '2_2': 0, '3_2': 0, '4_2': 0}
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            for num_discs in range(2,5):
                for piece in [1, 2]:
                    if check_window(window, num_discs, piece, config):
                        num_windows[f'{num_discs}_{piece}'] += 1
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            for num_discs in range(2,5):
                for piece in [1, 2]:
                    if check_window(window, num_discs, piece, config):
                        num_windows[f'{num_discs}_{piece}'] += 1
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            for num_discs in range(2,5):
                for piece in [1, 2]:
                    if check_window(window, num_discs, piece, config):
                        num_windows[f'{num_discs}_{piece}'] += 1
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            for num_discs in range(2,5):
                for piece in [1, 2]:
                    if check_window(window, num_discs, piece, config):
                        num_windows[f'{num_discs}_{piece}'] += 1
    return num_windows

def is_terminal_window(window, config):
    return window.count(1) == config.inarow or window.count(2) == config.inarow

def is_terminal_node(grid, config):
    # Check for draw 
    if list(grid[0, :]).count(0) == 0:
        return True
    # Check for win: horizontal, vertical, or diagonal
    # horizontal 
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            if is_terminal_window(window, config):
                return True
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            if is_terminal_window(window, config):
                return True
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return True
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return True
    return False

def get_heuristic(grid, mark, config):
    windows = count_windows(grid, config)
    
    num_twos = windows[f'2_{mark}']
    num_threes = windows[f'3_{mark}']
    num_fours = windows[f'4_{mark}']

    num_twos_opp = windows[f'2_{mark%2+1}']
    num_threes_opp = windows[f'3_{mark%2+1}']
    num_fours_opp = windows[f'4_{mark%2+1}']

    A, B, C, D, E, F = 1e6, 2, 1, -1, -100, -1e5
    score = A*num_fours + B*num_threes + C*num_twos + D*num_twos_opp + E*num_threes_opp + F*num_fours_opp
    return score

def minimax_a_b(node, depth, alpha, beta, maximizingPlayer, mark, config):
    is_terminal = is_terminal_node(node, config)
    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
    if depth == 0 or is_terminal:
        return get_heuristic(node, mark, config)
    if maximizingPlayer:
        value = -np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark, config)
            value = max(value, minimax_a_b(child, depth-1, alpha, beta, False, mark, config))
            alpha = max(alpha, value)
            if value >= beta: break
            
        return value
    else:
        value = np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark%2+1, config)
            value = min(value, minimax_a_b(child, depth-1, alpha, beta, True, mark, config))
            beta = min(beta, value)
            if value <= alpha: break
            
        return value

def score_move_a_b(grid, col, mark, config, nsteps):
    next_grid = drop_piece(grid, col, mark, config)
    score = minimax_a_b(next_grid, nsteps-1, -np.Inf, np.Inf, False, mark, config)
    return score

def agent4(obs, config):
    N_STEPS = 3
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    grid_ = get_grid(obs, config)
    scores = dict(zip(valid_moves, [score_move_a_b(grid_, col, obs.mark, config, N_STEPS) for col in valid_moves]))
    print(scores)
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    return random.choice(max_cols)


#BACKEND

ROWS = 6
COLUMNS = 7
INAROW = 4
config = CreateConfig(ROWS, COLUMNS, INAROW)

SIZE = config.rows * config.columns
ZEROS = [0 for z in range(SIZE)]
obs = CreateObs(ZEROS, 1)
obs.change_turn()

grid = get_grid(obs, config)

oponent = agent4

#FRONTEND

pygame.init()

SIZE = WIDTH, HEIGHT = 700, 600
RADIUS = 42
WIDTH_UNIT = WIDTH/config.columns
HEIGHT_UNIT = (HEIGHT-10)/config.rows

screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption('Connect 4   0-0')

myfont = pygame.font.SysFont('Comic Sans MS', 30)

clock = pygame.time.Clock()

board_surf = pygame.image.load('img/board.png').convert()

game = True
turn = 0
wins, losses = 0, 0

while True:
    col = -1
    
    if game: 
        screen.blit(board_surf, (0,0))
        print_board(obs, config, screen)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if game:
            if event.type == pygame.MOUSEBUTTONDOWN and obs.mark == 1:
                for i in range(config.columns):
                    if i*WIDTH_UNIT <= pygame.mouse.get_pos()[0] < (i+1)*WIDTH_UNIT:
                        col = i
        else:
            text_surf = myfont.render(text, False, 'green')
            screen.blit(text_surf, (WIDTH/2 -100, 100))
            if event.type == pygame.MOUSEBUTTONDOWN:
                obs.update_board(np.array(ZEROS))
                grid = get_grid(obs, config)
                game = True
                turn += 1
                obs.mark = turn%2 + 1
                pygame.display.set_caption(f'Connect 4   {wins}-{losses}')
    
    if obs.mark == 2 and game:
        col = oponent(obs, config)

    if col != -1:
        grid = drop_piece(grid, col, obs.mark, config)
        obs.update_board(grid)

        if is_terminal_node(grid, config):
            print_board(obs, config, screen)
            game = False
            if obs.mark == 1:
                text = 'Has ganado a una bestia'
                wins += 1
            else:
                text = 'Has perdido tolay'
                losses += 1
        else:
            obs.change_turn()
    

    pygame.display.update()
    clock.tick(60)