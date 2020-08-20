import pygame as pg
import os, random, time, pygame_menu
from pygame import gfxdraw

import game_engine




SCREEN_SIZE = (1200, 600)
TILE_SIZE = (60,60) #(width, height)
PIECE_RADIUS = (TILE_SIZE[0] // 2 - 5)
MARGIN_SIZE = 10    #margin between tiles
BASE_WIDTH = 600
BASE_HEIGHT = 60
PERIPHERAL_OFFSET_X = (9 * MARGIN_SIZE + 8 * TILE_SIZE[0]) + (TILE_SIZE[0] * 2 + 3 * MARGIN_SIZE)
PERIPHERAL_OFFSET_Y = TILE_SIZE[1]
MESSAGE_BOX_WIDTH = 300
MESSAGE_BOX_HEIGHT = 50
DICE_BUTTON_WIDTH = 80
DICE_BUTTON_HEIGHT = 40

FONT_SIZE = 36
MESSAGE_FONT_SIZE = 24

WHITE = (250,250,250)
BLACK = (5,5,5)
GREY = (212,210,209)
RED = (250, 5, 5)
ORANGE = (237,178,90)
OFF_WHITE = (240,240,240)
TILE_COLOR = (240,221,147)
ROSETTE_COLOR = (232,240,147)
MARGIN_COLOR = (224,172,98)

BOARD_START_Y = SCREEN_SIZE[1] // 2 - (4 * MARGIN_SIZE + 3 * TILE_SIZE[1]) // 2
BOARD_START_X = 100
BOARD_START_XY = (BOARD_START_X,BOARD_START_Y)  #Top left corner of the board

EMPTY_SQUARES = [(0,4),(0,5),(2,4),(2,5)]
NO_HINT = (-1, -1)
WHITE_BASE_ROW = 0
BLACK_BASE_ROW = 2

W = 'W'
B = 'B'
EMPTY = 'empty'
ROLL_BUTTON_TEXT = 'Roll'
WHITE_ROLLED = "White player rolled"
BLACK_ROLLED = "Black player rolled"

PLAYER_DICT = {1: 'HUMAN_PLAYER', 2: 'EXPECTIMAX_AGENT', 3: 'QLEARNING_AGENT'}


#TODO#########################IMPORTANT_NOTES####################################################

#all the code here assumes that white is the human player

#theres a difference in coordinates between numpy and pygame.screen. screen = (x,y), numpy = (y,x)

# get players at init (human, agent etc...)

#change (some?) player checks (player == W) to (player == human / agent) (Base button class...)

#Board.get_base_pieces() - should return the number if pieces in the players base


#TODO############################################################################################


def is_rosette(row, col):
    """
    checks if given square position is a rosette
    :param row: board row index
    :type row: int
    :param col: board column index
    :type col: int
    :return: True if is a rosette, False o.w
    :rtype: bool
    """
    return (row, col) in [(0, 0), (2, 0), (1, 3), (0, 6), (2, 6)]




def draw_circle(surface, x, y, radius, color):
    gfxdraw.aacircle(surface, x, y, radius, color)
    gfxdraw.filled_circle(surface, x, y, radius, color)

class GUI:



    def __init__(self, Board):      #TODO does not get board if doing menus n shit later
        pg.init()
        pg.font.init()
        self.screen = pg.display.set_mode(SCREEN_SIZE)
        background = pg.Surface(self.screen.get_size())
        background = background.convert()
        background.fill(OFF_WHITE)
        self.background = background
        MENU_SIZE = (SCREEN_SIZE[1] - 50, SCREEN_SIZE[0] - 200)
        theme = pygame_menu.themes.THEME_DARK.copy()
        theme.widget_alignment = pygame_menu.locals.ALIGN_LEFT
        theme.widget_font = pygame_menu.font.FONT_HELVETICA
        theme.widget_font_size = FONT_SIZE
        theme.widget_font_antialias = True

        menu = pygame_menu.Menu(*MENU_SIZE, 'The Royal Game of Ur - AI Project',
                                theme= theme)
        menu.add_selector('White player:',
                          [('Human',1), ('Expectimax', 2), ('Q-Learning', 3)],
                          onchange=self.__set_white_player, default= 2)
        menu.add_selector('Black player:',
                          [('Human', 1), ('Expectimax', 2), ('Q-Learning', 3)],
                          onchange=self.__set_black_player, default=0)
        menu.add_button('Play', self.__dummy_function)
        menu.add_button('About', self.__display_about)
        menu.add_button('Quit', pygame_menu.events.EXIT)

        menu.mainloop(self.screen)

        # self.__menu_buttons = []
        # self.__main_menu()
        # self.__create_menu_buttons()

        self.__gui_piece_buttons = []
        self.DiceButton, self.MessageBox = self.__draw_peripherals()
        self.create_empty_board()
        self.__cur_selected = (-1, -1)
        self.__black_player = B                 #TODO gets players at init (human, agents etc)
        self.__white_player = W
        self.__dice_rolled = False

    def __dummy_function(self):
        return 1

    def __set_white_player(self, value, index):
        self.__white_player = PLAYER_DICT[index]

    def __set_black_player(self, value, index):
        self.__black_player = PLAYER_DICT[index]

    def __display_about(self):
        return 1

    # def __main_menu(self):
    #
    #     done = False
    #     self.__create_menu_buttons()
    #     while not done:


    # def __create_menu_buttons(self):
    #
    #     self.__create_player_selection(SCREEN_SIZE[0] - 300, 80)
    #
    #
    # def __create_player_selection(self, x, y):
    #
    #     column_width = SCREEN_SIZE[0] // 4
    #     start_x = x + 20
    #     start_y = y + 80
    #     text = pg.font.SysFont('Impact', MESSAGE_FONT_SIZE).render('White player',True,BLACK)
    #     self.screen.blit(text, (start_x, start_y))
    #     start_x += text.get_size()[0] + 14
    #     draw_circle(self.screen, start_x, start_y, PIECE_RADIUS, WHITE)
    #     start_x = x + 20
    #     start_y += 30
    #     white_p_selection = MessageBox(start_x, start_y, self.screen)
    #     white_p_selection.update(self.screen, 'Expectimax')
    #
    #
    #
    #
    #     pg.display.update()



    def get_action(self, board):

        done = False

        while not done:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pygame_menu.events.EXIT
                    return None

                mouse_pos = pg.mouse.get_pos()
                mouse_clicked = pg.mouse.get_pressed()[0]  # checks LMB only
                if mouse_clicked:
                    if not self.__dice_rolled:
                        if self.DiceButton.is_clicked(mouse_pos):
                            cur_roll = self.DiceButton.roll(board)
                            self.__dice_rolled = True
                            roll_msg = BLACK_ROLLED + ' ' + str(cur_roll)
                            self.MessageBox.update(self.screen, roll_msg)
                            if cur_roll == 0:
                                time.sleep(1)
                                self.__dice_rolled = False
                                return None
                            continue
                    else: #dice was rolled
                        button = self.__get_piece_clicked(mouse_pos)
                        if button:
                            if button.is_hint():
                                self.__dice_rolled = False
                                return self.__cur_selected
                            else:
                                button_coords = button.get_coords()
                                self.__cur_selected = button_coords
                                next_pos = board.position_if_moved(*button_coords)
                                self.draw_board(board, self.__black_player, next_pos)

        self.__dice_rolled = False
        return None

    def __draw_peripherals(self):
        """
        draws the dice roll button and message window on the screen.
        """
        x, y = BOARD_START_XY
        x += PERIPHERAL_OFFSET_X
        y += PERIPHERAL_OFFSET_Y
        msg_box = MessageBox(x,y,self.screen)
        x += (MESSAGE_BOX_WIDTH // 2) - (DICE_BUTTON_WIDTH // 2)
        y += MESSAGE_BOX_HEIGHT + 10
        dice_button = DiceButton(x,y, self.screen)



        return dice_button, msg_box


    def __create_bases(self, screen, board):    #TODO coordinates according to players

        agent_base = Base((0,4), W)
        agent_base.draw_base(screen, board)

        human_base = Base((2,4), B)
        human_base.draw_base(screen,board)


        pg.display.update()
        return human_base, agent_base


    def draw_board(self, board, player, hint= (-1,-1)):   #TODO add '__' to name
        """
        Draws the board on the screen, with (x,y) being the position of the top left corner. To be used at the start of
        each game
        :param top_left: (x,y) of the top left corner of the desired board position
        :type (int, int)
        """
        if not self.__dice_rolled:  #new turn
            text_player = 'White player'
            if player == B:
                text_player = 'Black player'

            turn_message = text_player + "\'s turn"
            self.MessageBox.update(self.screen, turn_message)
        self.__gui_piece_buttons = []   #resets button list
        self.__draw_board_bg()  #board background color
        human_base, agent_base = self.__create_bases(self.screen, board.get_current_board())
        self.__gui_piece_buttons.append(human_base)
        end_tile = self.__create_end_tiles(board)
        self.__gui_piece_buttons.append(end_tile)
        self.__draw_tiles(board, player, hint)  #creates tiles with the relevant buttons


        pg.display.update()


    def __create_end_tiles(self, board):

        x, y = BOARD_START_XY
        x += 5 * MARGIN_SIZE + 4 * TILE_SIZE[0]
        y += 3 * MARGIN_SIZE + 2 * TILE_SIZE[1]
        board_matrix = board.get_current_board()
        self.__cur_selected = (1,1)     #TODO delete this. (for agent vs agent)
        hint = True if board.position_if_moved(*self.__cur_selected) == (2,5) else False
        end_tile = ExitTile((2,5),board_matrix[2][5], hint, rosette=False)
        end_tile.draw(self.screen, (x,y))

        return end_tile



    def __get_piece_clicked(self, mouse_pos):
        """
        Returns the button corresponding to the tile clicked on the screen.
        :param mouse_pos: (x,y) position of mouse when clicked
        :return: A Button object corresponding to the tile clicked.
        """
        for button in self.__gui_piece_buttons:
            if button.is_clicked(mouse_pos):
                return button
        return None



    def __draw_board_bg(self):
        """
        Draws the board background, with (x,y) being the position of the top left corner. The background is what fills
        the margins between tiles
        """
        x, y = BOARD_START_XY
        piece_one_size = (5 * MARGIN_SIZE + 4 * TILE_SIZE[0], 4 * MARGIN_SIZE + 3 * TILE_SIZE[1])
        piece_two_size = (1 * MARGIN_SIZE + 2 * TILE_SIZE[0], 2 * MARGIN_SIZE + 1 * TILE_SIZE[1])
        piece_three_size = (3 * MARGIN_SIZE + 2 * TILE_SIZE[0], 4 * MARGIN_SIZE + 3 * TILE_SIZE[1])

        piece_two_start = (x + piece_one_size[0], y + MARGIN_SIZE + TILE_SIZE[1])
        piece_three_start = (x + piece_one_size[0] + piece_two_size[0], y)

        pg.draw.rect(self.screen, MARGIN_COLOR, (x,y, piece_one_size[0], piece_one_size[1]), 0)
        pg.draw.rect(self.screen, MARGIN_COLOR, (piece_two_start[0], piece_two_start[1],
                                                 piece_two_size[0], piece_two_size[1]), 0)
        pg.draw.rect(self.screen, MARGIN_COLOR, (piece_three_start[0], piece_three_start[1],
                                                 piece_three_size[0], piece_three_size[1]), 0)


    def __draw_tiles(self, board_obj, player, hint):    #TODO DECIDE IF THERES NO HINT THEN ITS (-1, -1) OR NONE (game_engine)
        """
        Draws the tiles, creates clickable tiles as buttons (tiles with current player's pieces, move-to tiles)
        :param board: current board state
        :type: python list
        :param player: current player's color. 'W' or 'B' corresponding to white or black player.
        :type string
        :type: boolean

        """
        board = board_obj.get_current_board()
        for tile_num_y in range(len(board)):
            for tile_num_x in range(len(board[0])):
                mat_coords = (tile_num_y,tile_num_x)
                if mat_coords not in EMPTY_SQUARES:
                    #calculate top left of current tile
                    cur_x = BOARD_START_XY[0] + (MARGIN_SIZE + TILE_SIZE[0]) * tile_num_x + MARGIN_SIZE
                    cur_y = BOARD_START_XY[1] + (MARGIN_SIZE + TILE_SIZE[1]) * tile_num_y + MARGIN_SIZE

                    rosette = is_rosette(tile_num_y, tile_num_x)
                    piece = board[tile_num_y][tile_num_x]
                    white_move = player == W and piece == W
                    black_move = player == B and piece == B
                    capture_move_white = (player == W and piece == B)
                    capture_move_black = (player == B and piece == W)

                    if white_move or black_move:

                        button = self.__create_tile_button((cur_x, cur_y), player, mat_coords,
                                                           self.screen, rosette, is_hint=False)
                        self.__gui_piece_buttons.append(button)

                    elif  hint == mat_coords and capture_move_white:

                        button = self.__create_tile_button((cur_x,cur_y),B, mat_coords,
                                                           self.screen, rosette, is_hint=True)
                        self.__gui_piece_buttons.append(button)

                    elif hint == mat_coords and capture_move_black:
                        button = self.__create_tile_button((cur_x, cur_y), W, mat_coords,
                                                           self.screen, rosette, is_hint=True)
                        self.__gui_piece_buttons.append(button)

                    #empty hint tile
                    elif hint == mat_coords:
                        button = self.__create_tile_button((cur_x, cur_y), EMPTY, mat_coords,
                                                           self.screen, rosette, is_hint=True)
                        self.__gui_piece_buttons.append(button)

                    #empty tile. not clickable
                    else:
                        self.__create_normal_tile(rosette, piece, cur_x, cur_y)


        pg.display.update()

    def create_empty_board(self):

        board = [[EMPTY] * 8] + [[EMPTY] * 8] + [[EMPTY] * 8]
        board[0][4] = 7
        board[0][5] = 0
        board[2][4] = 7
        board[2][5] = 0
        self.__draw_board_bg()
        for tile_num_y in range(len(board)):
            for tile_num_x in range(len(board[0])):
                mat_coords = (tile_num_y, tile_num_x)
                if mat_coords not in EMPTY_SQUARES:
                    cur_x = BOARD_START_XY[0] + (MARGIN_SIZE + TILE_SIZE[0]) * tile_num_x + MARGIN_SIZE
                    cur_y = BOARD_START_XY[1] + (MARGIN_SIZE + TILE_SIZE[1]) * tile_num_y + MARGIN_SIZE
                    rosette = is_rosette(tile_num_y, tile_num_x)
                    self.__create_normal_tile(rosette,EMPTY,cur_x,cur_y)
        self.__create_bases(self.screen, board)
        pg.display.update()


    def __create_tile_button(self, top_left, player, mat_coords, screen, rosette=False, is_hint=False):
        """
        Creates a tile button with a white / black piece on it. Clicking on the button will call draw_board and will
        create hint tiles as buttons in order to move the piece there. The method also adds the button to gui_board
        :param top_left: top left coordinates of the button on the screen. (x,y)
        :type: (int, int)
        :param player: 'W" or 'B' corresponding to White or Black piece, or 'empty' for an empty tile
        :type: string
        :param is_hint: If the current tile can be moved into, creates red rectangle on the button.
        :type: boolean
        :param mat_coord: coordinates on the board matrix (for coloring rosettes) (y,x)
        """
        button = Button(mat_coords, player, is_hint, rosette)
        button.draw(screen, top_left)
        return button

    def __create_normal_tile(self, rosette, piece, x, y):
        """
        Creates a normal, non-clickable tile with the corresponding piece on it, and appropriate color
        :param rosette: True if the tile is a rosette tile, False otherwise
        :type: boolean
        :param piece: 'W' or 'B' corresponding to the piece on the tile, or '_' for empty tile
        :type: string
        """
        color = ROSETTE_COLOR if rosette else TILE_COLOR
        rect = pg.draw.rect(self.screen, color,(x, y, TILE_SIZE[0], TILE_SIZE[1]))
        # draw the piece
        if piece == W:
            # pg.draw.circle(self.screen, WHITE, rect.center, (TILE_SIZE[0] // 2) - 5)
            # pg.draw.circle(self.screen, BLACK, rect.center, (TILE_SIZE[0] // 2) - 5, 2)  # black outline
            draw_circle(self.screen,*rect.center, TILE_SIZE[0] // 2 - 5, WHITE)

        elif piece == B:
            # pg.draw.circle(self.screen, BLACK, rect.center, (TILE_SIZE[0] // 2) - 5)
            draw_circle(self.screen, *rect.center, TILE_SIZE[0] // 2 - 5, BLACK)


    def update_msg(self, message):

        self.MessageBox.update(self.screen, message)




class Button:

    def __init__(self, mat_coords, piece=EMPTY, hint=False, rosette=False):

        self.__mat_coords = mat_coords
        self.__rosette = rosette
        self.__hint = hint
        self.piece = piece
        self.tile = None


    def draw(self, screen, top_left):

        y, x = self.__mat_coords
        color = ROSETTE_COLOR if is_rosette(y, x) else TILE_COLOR

        # draw the tile
        self.tile = pg.draw.rect(screen, color, (top_left[0], top_left[1], TILE_SIZE[0], TILE_SIZE[1]))

        # draw the piece
        if self.piece == W:
            # pg.draw.circle(screen, WHITE, self.__tile.center, (TILE_SIZE[0] // 2) - 5)
            # pg.draw.circle(screen, BLACK, self.__tile.center, (TILE_SIZE[0] // 2) - 5, 2)  # black outline
            draw_circle(screen, *self.tile.center, PIECE_RADIUS, WHITE)

        elif self.piece == B:
            draw_circle(screen, *self.tile.center, PIECE_RADIUS, BLACK)

        if self.__hint:  # hint button
            pg.draw.rect(screen, RED, (top_left[0] + 2, top_left[1] + 2, TILE_SIZE[0] - 4, TILE_SIZE[1] - 4), 2)

    def get_center(self): return self.tile.center

    def get_coords(self): return self.__mat_coords

    def is_rosette(self): return self.__rosette

    def is_hint(self): return self.__hint

    def is_clicked(self, mouse_pos): return self.tile.collidepoint(mouse_pos)

    def get_piece(self): return self.piece

class ExitTile(Button):

    def draw(self, screen, top_left):
        x, y = top_left

        tile_size = (1 * MARGIN_SIZE + 2 * TILE_SIZE[0], 0.5 * MARGIN_SIZE + 1 * TILE_SIZE[1])
        self.tile = pg.draw.rect(screen, BLACK, (x, y, tile_size[0], tile_size[1]), 1)
        pg.display.update()


class Base(Button):

    def draw_base(self, screen, board):

        screen_size = screen.get_size()
        base_row = BLACK_BASE_ROW
        if self.piece == W:
            top_left = (0, 0)
            base_row = WHITE_BASE_ROW
        else:
            top_left = (0, screen_size[1] - BASE_HEIGHT)
        self.tile = pg.draw.rect(screen, GREY, (*top_left, BASE_WIDTH, BASE_HEIGHT))  #background
        num_of_pieces = board[base_row][4]
        offset = 30
        piece_color = WHITE if self.piece == W else BLACK
        for i in range(num_of_pieces):
            piece_center_x = offset + PIECE_RADIUS * 2 * i+1 + MARGIN_SIZE * i
            piece_center_y = top_left[1] + (BASE_HEIGHT // 2)
            draw_circle(screen, piece_center_x, piece_center_y, PIECE_RADIUS, piece_color)

        # draw_circle(screen, *self.__tile.center, (TILE_SIZE[0] // 2 - 5), WHITE)





class MessageBox:

    def __init__(self, x,y, screen):
        self.__msg_str = ''
        self.__draw(x,y, screen)
        self.__top_left = (x,y)

    def __draw(self,x,y, screen):

        pg.draw.rect(screen,GREY, (x,y,MESSAGE_BOX_WIDTH,MESSAGE_BOX_HEIGHT))
        pg.draw.rect(screen,BLACK,(x,y,MESSAGE_BOX_WIDTH,MESSAGE_BOX_HEIGHT),5)

    def update(self, screen, message):

        self.__draw(*self.__top_left, screen)   #reset box
        self.__msg_str = message
        text = pg.font.SysFont('Impact', MESSAGE_FONT_SIZE).render(message,True,BLACK)
        text_start_x = self.__top_left[0] + 3
        text_start_y = self.__top_left[1] + MESSAGE_BOX_HEIGHT // 3
        screen.blit(text, (text_start_x,text_start_y))
        pg.display.update()

    def get_message(self): return self.__msg_str

class DiceButton:

    def __init__(self,x,y, screen):

        self.__rect = None
        self.__last_roll = None
        self.__draw(x, y, screen)
        self.__was_clicked = False

    def __draw(self, x, y, screen):

        self.__rect = pg.draw.rect(screen, ORANGE, (x,y, DICE_BUTTON_WIDTH, DICE_BUTTON_HEIGHT))
        text = pg.font.SysFont('Impact', FONT_SIZE).render(ROLL_BUTTON_TEXT, True, WHITE)
        text_start_x = x + (DICE_BUTTON_WIDTH // 2) - (text.get_width() // 2)
        text_start_y = y - 3
        screen.blit(text,(text_start_x, text_start_y))
        pg.display.update()

    def roll(self, board):

        self.__last_roll = board.roll_dice()  #TODO copied roll_dice() from game_engine.Board to here

        return sum(self.__last_roll)

    def is_clicked(self, mouse_pos): return self.__rect.collidepoint(mouse_pos)

    def get_last_roll(self): return self.__last_roll

b = game_engine.Board()
guhi = GUI(b)
# b = game_engine.Board()
# # board = b._Board__board
# # board[1][0] = W
# print(b)
# # b.set_board(board)
# # b.run_game()
# gui = GUI()
# # # gui.draw_board(board,W, (1,2))
# print(gui.get_action(b))

# done = False
# while not done:
#     for event in pg.event.get():
#         if event.type == pg.QUIT:
#             done = True
#     pg.draw.rect(screen, (0,128,250), pg.Rect(30,30,60,60))
#     pg.display.flip()



# pauseUntil = time.time() + random.randint(5, 15) * 0.1
#
# while time.time() < pauseUntil:
#
#     pygame.display.update()

