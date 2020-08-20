import pygame as pg
import os, random, time, pygame_menu, sys
from pygame import gfxdraw
import Agents, expectiminimax_agent, simpleAgents
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
WINNER_DISPLAY_SIZE = (600, 300)

FONT_SIZE = 36
MESSAGE_FONT_SIZE = 24
WINNER_FONT_SIZE = 64

WHITE = (250,250,250)
BLACK = (5,5,5)
GREY = (212,210,209)
RED = (250, 5, 5)
ORANGE = (237,178,90)
OFF_WHITE = (240,240,240)
TILE_COLOR = (240,221,147)
ROSETTE_COLOR = (232,240,147)
MARGIN_COLOR = (224,172,98)
OUT_OF_PLAY_COLOR = (247, 241, 68)

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
NO_LEGAL_MOVES = 'No available moves!'


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
    """
    Draws an anti-aliased circle on the screen with the given parameters.
    :param surface: The surface on which to draw on.
    :type surface: pygame.display object
    :param x: x coordinate of the center
    :type x: int
    :param y: y coordinate of the center
    :type y: int
    :param radius: Radius of the circle
    :type radius: int
    :param color: Color of the circle
    :type color: (int, int, int)
    """

    gfxdraw.aacircle(surface, x, y, radius, color)
    gfxdraw.filled_circle(surface, x, y, radius, color)


def determine_human_players(p1, p2):
    """
    Determines if the given players are human or not.
    :param p1: player to check if human
    :type p1: object from Agents.py
    :param p2: player to check if human
    :type p2: object from Agents.py
    :return: True to each player that is a human player. False otherwise
    """

    p1_human = False
    p2_human = False
    if isinstance(p1,Agents.Human):
        p1_human = True
    if isinstance(p2, Agents.Human):
        p2_human = True

    return p1_human, p2_human


class GUI:
    """
    The main class to draw the game graphics. draws on the screen all relevant
    graphics and handles all human interaction. Able to provide an action from
    the human player to the game_engine to update the game mechanics.
    """

    def __init__(self, Board, main_menu = False):

        pg.init()
        pg.font.init()
        self.screen = pg.display.set_mode(SCREEN_SIZE)
        self.screen.fill(OFF_WHITE)
        self.board = Board
        self.__gui_buttons = []
        self.DiceButton, self.MessageBox = self.__draw_peripherals()
        self.__cur_selected = (-1, -1)
        self.__dice_rolled = False

        self.__black_player = Agents.Human()
        self.__white_player = expectiminimax_agent.ExpectiminimaxAgent(B, W)
        self.__delay = 0
        self.__has_menu = main_menu
        if main_menu:
            self.create_main_menu()


    def create_main_menu(self):
        """
        Creates a main menu with the appropriate buttons:
        -White player selection (Agents, Human)
        -Black player selection (Agents, Human)
        -Play button - initiates board.run_game with the selected players
        -Delay selection - Creates delay between agent actions, in order to be
                            better understood by human eyes.
        -Quit - Closes the pygame window.

        After creating the buttons, the menu is loaded and presented on screen
        in menu.mainloop().
        """

        MENU_SIZE = (SCREEN_SIZE[1] - 50, SCREEN_SIZE[0] - 200)
        theme = pygame_menu.themes.THEME_DARK.copy()
        theme.widget_alignment = pygame_menu.locals.ALIGN_LEFT
        theme.widget_font = pygame_menu.font.FONT_HELVETICA
        theme.widget_font_size = FONT_SIZE
        theme.widget_font_antialias = True

        menu = pygame_menu.Menu(*MENU_SIZE,
                                'The Royal Game of Ur - AI Project',
                                theme=theme)
        menu.add_selector('White player:',
                          [('Human', 1), ('Expectimax', 2), ('Q-Learning', 3),
                           ('Random Playing Agent', 4),('Greedy', 5)],
                          onchange=self.set_white_player, default=3)
        menu.add_selector('Black player:',
                          [('Human', 1), ('Expectimax', 2), ('Q-Learning', 3),
                           ('Random Playing Agent', 4),('Greedy', 5)],
                          onchange=self.set_black_player, default=0)
        menu.add_button('Play', self.__play)
        menu.add_selector('Delay (Seconds):',
                          [('0', 0),('0.5', 0.5), ('1', 1),('1.5', 1.5), ('2', 2), ('2.5', 2.5),
                           ('3', 3),('3.5', 3.5), ('4', 4)],
                          onchange=self.__set_delay, default=0)
        menu.add_button('Quit', pygame_menu.events.EXIT)

        self.main_menu = menu
        self.main_menu.mainloop(self.screen)

    def __display_winner(self,winner_str):

        x = SCREEN_SIZE[0] // 2 - WINNER_DISPLAY_SIZE[0] // 2
        y = SCREEN_SIZE[1] // 2 - WINNER_DISPLAY_SIZE[1] // 2
        box = pg.draw.rect(self.screen,WHITE,(x,y,WINNER_DISPLAY_SIZE[0],WINNER_DISPLAY_SIZE[1]))
        pg.draw.rect(self.screen,BLACK,(x,y,WINNER_DISPLAY_SIZE[0],WINNER_DISPLAY_SIZE[1]),5)
        text = pg.font.SysFont('Impact', WINNER_FONT_SIZE).render(winner_str, True, BLACK)
        text_x, text_y = text.get_size()
        x_coor = box.centerx - text_x // 2
        y_coor = box.centery - text_y // 2
        # text_y = box.centery

        self.screen.blit(text, (x_coor, y_coor))
        pg.display.update()



    def __play(self):
        """
        The callback function activated when pressing the Play button in the
        main menu. Fills the screen with background color and starts the game
        loop.
        """

        self.screen.fill(OFF_WHITE)
        self.__draw_peripherals()
        winner = self.board.run_game(self.__black_player, self.__white_player, self,
                            self.__delay)
        winner_str = "White Player Wins!" if winner == 'W' else "Black Player Wins!"
        # self.MessageBox.update(self.screen, "The Winner is: "+ winner_str)
        self.__display_winner(winner_str)
        time.sleep(2)
        exit(0)

    def __set_delay(self, value, delay):
        """
        Sets the delay between agent actions for better understanding under
        human eyes.
        :param value: Delay in seconds as string. (not used)
        :type value: string
        :param delay: Delay between agent actions.
        :type delay: int
        """

        self.__delay = delay

    def set_white_player(self, value, index):
        """
        Callback from White Player Selection in main menu. Sets the white player
        as the appropriate agent selected.
        :param value: Agent name. (not used)
        :type value: string
        :param index: Index deciding which agent to instantiate.
        :type index: int
        """

        player = None
        if index == 1:
            player = Agents.Human()
        elif index == 2:
            player = expectiminimax_agent.ExpectiminimaxAgent(W, B)
        elif index == 3:
            player = Agents.DeepQAgent(W)
            player.network.load_weights("best_player_yet")
        elif index == 4:
            player = simpleAgents.RandomAgent(W)
        elif index == 5:
            player = simpleAgents.GreedyAgent(W)

        self.__white_player = player

    def set_black_player(self, value, index):
        """
        Callback from Black Player Selection in main menu. Sets the black player
        as the appropriate agent selected.
        :param value: Agent name. (not used)
        :type value: string
        :param index: Index deciding which agent to instantiate.
        :type index: int
        """

        player = None
        if index == 1:
            player = Agents.Human()
        elif index == 2:
            player = expectiminimax_agent.ExpectiminimaxAgent(B, W)
        elif index == 3:
            player = Agents.DeepQAgent(B)
            player.network.load_weights("best_player_yet")
        elif index == 4:
            player = simpleAgents.RandomAgent(B)
        elif index == 5:
            player = simpleAgents.GreedyAgent(B)

        self.__black_player = player



    def get_action(self, board):
        """
        Main loop in order to get action from a human player. The function
        gets events happening on screen from pygame.events (Mouse clicked
        somewhere, close button pressed, etc..) and reacts accordingly. The
        human player has a couple of options interacting with the game:
            -Pressing the close (red X) button on the screen closes the window
            When the mouse is clicked on screen, the function evaluates where
            the click happened and determines if a button was pressed:
                -if the dice were not rolled yet, the user can press the roll
                button to roll the dice.
                - if the dice were already rolled, the user can choose a piece
                on the board in order to move (Including the user's base).
                - if the user already pressed on a piece they want to move, a
                red hint is drawn on the respective tile to move to. if the user
                clicks on that tile, the function returns the move decided by
                the user.
        The function draws appropriate buttons on the screen in regards to
        which actions the user can perform (roll dice, move piece etc...).
        :param board: Board object in order to get relevant information on the
        game state.
        :type board: Board object
        :return: Coordinates of the piece to be moved
        :rtype: (int, int)
        """

        done = False
        color = board.get_cur_turn()
        roll_msg_color = 'White' if color == 'W' else 'Black'
        while not done:
            for event in pg.event.get():
                if event.type == pg.QUIT:   #User pressed red X button
                    pg.quit()
                    return None

                mouse_pos = pg.mouse.get_pos()
                mouse_clicked = pg.mouse.get_pressed()[0]  # checks LMB only
                if mouse_clicked:
                    if not self.__dice_rolled:    #prevents multiple dice rolls
                        if self.DiceButton.is_clicked(mouse_pos):
                            cur_roll = self.DiceButton.roll(board)
                            self.__dice_rolled = True
                            roll_msg = roll_msg_color + ' rolled ' + str(cur_roll)
                            self.MessageBox.update(self.screen, roll_msg)
                            if cur_roll == 0:
                                time.sleep(1)
                                self.__dice_rolled = False
                                return None
                            continue
                    else:   #dice was rolled
                        if not board.get_legal_moves():
                            # time.sleep(1)
                            self.MessageBox.update(self.screen,NO_LEGAL_MOVES)
                            time.sleep(1)
                            self.__dice_rolled = False
                            return None
                        button = self.__get_piece_clicked(mouse_pos)
                        if button:
                            if button.is_hint():    #user selected a legal move
                                self.__dice_rolled = False
                                return self.__cur_selected
                            else:   #user selected a piece
                                button_coords = button.get_coords()
                                self.__cur_selected = button_coords
                                next_pos = board.position_if_moved(*button_coords)
                                self.draw_board(board, color, next_pos)

        self.__dice_rolled = False
        return None

    def __draw_peripherals(self):
        """
        Creates the Message box and Dice roll button and draws them on screen.
        :return: Dicebutton and MessageBox objects to be added to the button
        list
        :rtype: DiceButton and MessageBox objects
        """

        x, y = BOARD_START_XY
        x += PERIPHERAL_OFFSET_X
        y += PERIPHERAL_OFFSET_Y
        msg_box = MessageBox(x,y,self.screen)
        x += (MESSAGE_BOX_WIDTH // 2) - (DICE_BUTTON_WIDTH // 2)
        y += MESSAGE_BOX_HEIGHT + 10
        dice_button = DiceButton(x,y, self.screen)

        return dice_button, msg_box


    def __create_bases(self, screen, board):
        """
        Creates base buttons for both White and Black and draws the appropriate
        amount of pieces left inside each player's respective bases.
        :param screen: Screen object to draw the bases on. The screen object is
        passed into the bases' draw_base function
        :type screen: pygame.display object
        :param board: Board object in order to get how much pieces are left in
        each base.
        :type board: Board object
        :return: White and black bases
        :rtype: (Base object, Base object)
        """

        white_base = Base((0,4), W)
        white_base.draw_base(screen, board)

        black_base = Base((2,4), B)
        black_base.draw_base(screen,board)

        pg.display.update()
        return white_base, black_base


    def draw_board(self, board, player, hint= (-1,-1)):
        """
        Draws the relevant board on screen, creates the appropriate buttons
        according to where the human player's pieces are, creates a hint tile
        if needed, draws out of play pieces, updates message box with the
        current dice roll.
        The main function used when updating the screen with the current board.
        :param board: Board object to extract relevant information
        :type board: Board object
        :param player: Current player turn ('W' or 'B')
        :type player: string
        :param hint: The tile to draw the hint on. Updated according to if the
        player selected a certain piece to move with.
        :type hint: (int, int)
        """

        if not self.__dice_rolled:  #new turn
            text_player = 'White player'
            if player == B:
                text_player = 'Black player'

            turn_message = text_player + "\'s turn"
            self.MessageBox.update(self.screen, turn_message)
        board_mat = board.get_current_board()
        self.__gui_buttons = []   #resets button list
        self.__draw_board_bg()  #board background color
        white_base, black_base = self.__create_bases(self.screen, board_mat)
        white_human, black_human = determine_human_players(self.__white_player, self.__black_player)
        if white_human:
            self.__gui_buttons.append(white_base)
        if black_human:
            self.__gui_buttons.append(black_base)

        self.__create_out_of_play(board_mat)
        tiles = self.__create_end_tiles(board)  #invisible buttons to remove
                                                #pieces from play
        for end_tile in tiles:
            if end_tile:
                self.__gui_buttons.append(end_tile)

        self.__draw_tiles(board, player, hint)  #creates tiles with the relevant buttons

        pg.display.update()


    def __create_out_of_play(self, board):
        """
        Draws out of play pieces on each player's respective side.
        :param board: Board matrix to get how many pieces are out of play for
        each player.
        :type board: python list of lists
        """

        out_white = board[0][5]
        out_black = board[2][5]

        offset = 30

        center_y = BASE_HEIGHT // 2
        for i in range(out_white):
            center_x = SCREEN_SIZE[0] - (
                    offset + PIECE_RADIUS * 2 * i + 1 + MARGIN_SIZE * i)
            draw_circle(self.screen,center_x,center_y,PIECE_RADIUS,
                        OUT_OF_PLAY_COLOR)

        center_y = SCREEN_SIZE[1] - BASE_HEIGHT // 2
        for i in range(out_black):
            center_x = SCREEN_SIZE[0] - (
                        offset + PIECE_RADIUS * 2 * i + 1 + MARGIN_SIZE * i)
            draw_circle(self.screen, center_x, center_y, PIECE_RADIUS,
                        OUT_OF_PLAY_COLOR)

        pg.display.update()

    def __create_end_tiles(self, board):
        """
        Creates invisible buttons according to which player is Human. The
        buttons remove pieces from the board to the out_of_play area
        :param board: Board object to get relevant information on the end tiles
        :type board: Board object
        :return: White and black exit tiles according to which player is human.
        If a player is not human then None is returned isntead of the respective
        colored tile.
        :rtype: (ExitTile object, ExitTile object)
        """

        x, y = BOARD_START_XY
        end_tile_w = None
        end_tile_b = None
        if isinstance(self.__black_player, Agents.Human):
            x += 5 * MARGIN_SIZE + 4 * TILE_SIZE[0]
            y += 3 * MARGIN_SIZE + 2 * TILE_SIZE[1]
            board_matrix = board.get_current_board()
            hint = True if board.position_if_moved(*self.__cur_selected) == (2,5) else False
            end_tile_b = ExitTile((2,5),board_matrix[2][5], hint, rosette=False)
            end_tile_b.draw(self.screen, (x,y))

        x, y = BOARD_START_XY
        if isinstance(self.__white_player, Agents.Human):
            x += 5 * MARGIN_SIZE + 4 * TILE_SIZE[0]

            board_matrix = board.get_current_board()
            hint = True if board.position_if_moved(*self.__cur_selected) == (0,5) else False
            end_tile_w = ExitTile((0,5),board_matrix[0][5], hint, rosette=False)
            end_tile_w.draw(self.screen, (x,y))


        return end_tile_w, end_tile_b



    def __get_piece_clicked(self, mouse_pos):
        """
        Returns the button clicked on the screen.
        :param mouse_pos: (x,y) position of mouse when clicked
        :type mouse_pos: (int, int)
        :return: A Button object that was clicked, None otherwise
        :rtype: Button object, or None
        """

        for button in self.__gui_buttons:
            if button.is_clicked(mouse_pos):
                return button
        return None



    def __draw_board_bg(self):
        """
        Draws the board background, with (x,y) being the position of the top
        left corner. The background is what fills the margins between tiles.
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


    def __draw_tiles(self, board_obj, player, hint):
        """
        Creates clickable tiles (Buttons) and non-clickable tiles (empty, or
        opponent's pieces) draws them on the screen.
        Clickable tiles: Tiles with the players pieces on them, or hint tiles
        Non clickable tiles: All other tiles
        :param board_obj: Board object to get relevant information on tiles
        :type board_obj: Board object
        :param player: Current player's turn
        :type player: string
        :param hint: The tile with hint on them
        :type hint: (int, int)
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
                        self.__gui_buttons.append(button)

                    elif  hint == mat_coords and capture_move_white:

                        button = self.__create_tile_button((cur_x,cur_y),B, mat_coords,
                                                           self.screen, rosette, is_hint=True)
                        self.__gui_buttons.append(button)

                    elif hint == mat_coords and capture_move_black:
                        button = self.__create_tile_button((cur_x, cur_y), W, mat_coords,
                                                           self.screen, rosette, is_hint=True)
                        self.__gui_buttons.append(button)

                    #empty hint tile
                    elif hint == mat_coords:
                        button = self.__create_tile_button((cur_x, cur_y), EMPTY, mat_coords,
                                                           self.screen, rosette, is_hint=True)
                        self.__gui_buttons.append(button)

                    #empty tile. not clickable
                    else:
                        self.__create_normal_tile(rosette, piece, cur_x, cur_y)


        pg.display.update()

    def create_empty_board(self):
        """
        Draws a blank board with all the pieces in their bases
        """

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
        Creates a tile button with a white / black piece on it. Clicking on the
        button will call draw_board and will create hint tiles as buttons in
        order to move the piece there. The method also adds the button to
        gui_board.
        :param top_left: top left coordinates of the button on the screen. (x,y)
        :type: (int, int)
        :param player: 'W" or 'B' corresponding to White or Black piece, or
        'empty' for an empty tile
        :type: string
        :param is_hint: If the current tile can be moved into, creates red
        rectangle on the button.
        :type: boolean
        :param mat_coord: coordinates on the board matrix
        (for coloring rosettes) (y,x)
        :return: A clickable tile with relevant art on it
        :rtype: Button object
        """

        button = Button(mat_coords, player, is_hint, rosette)
        button.draw(screen, top_left)
        return button

    def __create_normal_tile(self, rosette, piece, x, y):
        """
        Draws a normal, non-clickable tile with the corresponding piece on
        it, and appropriate color
        :param rosette: True if the tile is a rosette tile, False otherwise
        :type: boolean
        :param piece: 'W' or 'B' corresponding to the piece on the tile, or '_'
        for empty tile
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
        """
        Updates the message inside the MessageBox to be displayed on screen.
        :param message: Message to be displayed.
        :type message: string
        """

        self.MessageBox.update(self.screen, message)




class Button:
    """
    Button class to be saved inside GUI object.
    """

    def __init__(self, mat_coords, piece=EMPTY, hint=False, rosette=False):

        self.__mat_coords = mat_coords
        self.__rosette = rosette
        self.__hint = hint
        self.piece = piece
        self.tile = None


    def draw(self, screen, top_left):
        """
        Draws the button on the screen
        :param screen: The screen to draw the button on.
        :type screen: pygame.display object
        :param top_left: Top left coordinates of the button inside screen
        :type top_left: (int, int)
        """

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
    """
    Invisible button to remove pieces from the board to the out_of_play area.
    """
    def draw(self, screen, top_left):
        """
        Draws the button on screen. The button drawn is the same color as the
        screen in order to be invisible.
        :param screen: Screen object to draw on.
        :type screen: pygame.display object
        :param top_left: Top left coordinates of the button on the screen.
        :type top_left: (int, int)
        """

        x, y = top_left
        tile_size = (1 * MARGIN_SIZE + 2 * TILE_SIZE[0], 0.5 * MARGIN_SIZE + 1 * TILE_SIZE[1])
        self.tile = pg.draw.rect(screen, OFF_WHITE, (x, y, tile_size[0], tile_size[1]), 2)
        if self.is_hint():
            pg.draw.rect(screen,RED, (x, y, tile_size[0], tile_size[1]), 2)
        pg.display.update()


class Base(Button):
    """
    Handles all operations regarding the base of a player.
    """

    def draw_base(self, screen, board):
        """
        Draws the base on the screen.
        :param screen: Screen object to draw the base on.
        :type screen: pygame.display object
        :param board: Board object to get relevant information on the base from
        """

        screen_size = screen.get_size()
        base_row = BLACK_BASE_ROW
        if self.piece == W:
            top_left = (0, 0)
            base_row = WHITE_BASE_ROW
        else:
            top_left = (0, screen_size[1] - BASE_HEIGHT)
        self.tile = pg.draw.rect(screen, GREY, (*top_left, SCREEN_SIZE[0], BASE_HEIGHT))  #background
        num_of_pieces = board[base_row][4]
        offset = 30
        piece_color = WHITE if self.piece == W else BLACK
        for i in range(num_of_pieces):
            piece_center_x = offset + PIECE_RADIUS * 2 * i+1 + MARGIN_SIZE * i
            piece_center_y = top_left[1] + (BASE_HEIGHT // 2)
            draw_circle(screen, piece_center_x, piece_center_y, PIECE_RADIUS, piece_color)







class MessageBox:
    """
    Displays messages on the screen
    """

    def __init__(self, x,y, screen):
        self.__msg_str = ''
        self.__draw(x,y, screen)
        self.__top_left = (x,y)

    def __draw(self,x,y, screen):
        """
        Draws the an empty message box on screen
        :param x: x coordinate of the message box on screen
        :type x: int
        :param y: y coordinate of the message box on screen
        :type y: int
        :param screen: The screen on which to draw on
        :type screen: pygame.display object
        """

        pg.draw.rect(screen,GREY, (x,y,MESSAGE_BOX_WIDTH,MESSAGE_BOX_HEIGHT))
        pg.draw.rect(screen,BLACK,(x,y,MESSAGE_BOX_WIDTH,MESSAGE_BOX_HEIGHT),5)

    def update(self, screen, message):
        """
        Updates the messagebox with the given message
        :param screen: Screen object to draw the message on
        :type screen: pygame.display object
        :param message: The message to be displayed
        """

        self.__draw(*self.__top_left, screen)   #reset box
        self.__msg_str = message
        text = pg.font.SysFont('Impact', MESSAGE_FONT_SIZE).render(message,True,BLACK)
        text_start_x = self.__top_left[0] + 3
        text_start_y = self.__top_left[1] + MESSAGE_BOX_HEIGHT // 3
        screen.blit(text, (text_start_x,text_start_y))
        pg.display.update()

    def get_message(self): return self.__msg_str


class DiceButton:
    """
    The button to handle dice operations
    """

    def __init__(self,x,y, screen):

        self.__rect = None
        self.__last_roll = None
        self.__draw(x, y, screen)
        self.__was_clicked = False

    def __draw(self, x, y, screen):
        """
        Draws the button on screen
        :param x: x coordinate of the button on the screen
        :type x: int
        :param y: y coordinate of the button on the screen
        :type y: int
        :param screen: Screen object to draw the button on
        :type screen: pygame.display object
        """

        self.__rect = pg.draw.rect(screen, ORANGE, (x,y, DICE_BUTTON_WIDTH, DICE_BUTTON_HEIGHT))
        text = pg.font.SysFont('Impact', FONT_SIZE).render(ROLL_BUTTON_TEXT, True, WHITE)
        text_start_x = x + (DICE_BUTTON_WIDTH // 2) - (text.get_width() // 2)
        text_start_y = y - 3
        screen.blit(text,(text_start_x, text_start_y))
        pg.display.update()

    def roll(self, board):
        """
        Rolls the dice.
        :param board: Board object to roll the dice in.
        :type board: Board object
        """

        self.__last_roll = board.roll_dice()

        return sum(self.__last_roll)

    def is_clicked(self, mouse_pos): return self.__rect.collidepoint(mouse_pos)

    def get_last_roll(self): return self.__last_roll

