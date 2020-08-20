# The Royal Game of Ur - AI Project

## Introduction

The Royal Game of Ur is an ancient Mesopotamian board game, considered to be the spiritual predecessor to Backgammon and similar games.  The game was relatively obscure, up until the British Museum’s curator Dr. Irving Finkel reconstructed the rules of the game from a cuneiform tablet, and posted a Youtube video of him playing the game against another youtuber. 
Stumbling upon this video, we decided that creating agents for this game is a worthy project idea, since it has some different aspects to it than other games:
It has 2 players, it is stochastic, each game is very different from another, and the state of the game may change quickly. In addition, there is little research and experimentation with this game regarding AI.
The goal of this project is to pioneer AI agents for the game (Since there is little data on it, and no open source learning projects that are available to the public) using knowledge acquired from the course, in order to create the best possible agent and hopefully beat human opponents. Another goal is to present observations in the creation of these agents that reflect upon the nature of search and learning algorithms. 

### Rules of the game

The rules of the game are fairly simple. Each of the 2 players has 7 pieces which have to be moved  across the board in a given path.

The goal of each player is to remove all the pieces off the board by completing the path with all of his pieces. The pieces can only move in the forward direction of the path, single piece in each turn.
Pieces are moved by rolling 4 binary dice.
The amount of tiles each piece can move ranges from 0 (loss of a turn) to 4 according to the dice rolled. 
The board is split between player bases and a main alley. Each player can move inside their respective base, and both players share the main alley. Each tile may contain only one piece at a time. Within the main alley players can capture opponent pieces by landing on the same tile as them. Captured pieces are returned to the start of the respective player’s base. 
Special tiles (“Rosettes”) are placed across the board. Landing on a rosette tile grants the current player an additional roll. The rosette within the main alley has an additional rule tied to it; Enemies cannot capture pieces on this tile.
Lastly, in order to remove a piece from the board, the player must roll the exact number corresponding to the amount of tiles left between the piece and the exit.


## How to run 

### requirements:

To run the code you must have python 3.7.4 installed.

Create a virtual environment with the following command:

**_Virtualenv path/virtual_env_name --python python 3 --system-site-packages_**

Activate the virtual environment 

**_source path/virtual_env_name/bin/activate.csh_**

Run our Makefile to install the required python libraries with make install
with the virtual environment enabled follow 5.2 to start the game.


### Instructions

To run the game simply access the game files through the cmd wherever you saved them and run the command **_python game_engine.py_**

Once you press enter, a game menu should pop up with character selection, you can put any two players against each other.
If you’d like to see the computer’s actions more clearly, opt to add a delay in the main menu.

Once ready hit play.

If you choose to use a human player you must press the roll button to roll the dice.
Once you have a roll, you can then press on a piece and you’ll have a hint pop up where you can then click once more and place your piece on the correct spot.
To move pieces that are out of the board simply select them.

The game also offers a way to run it without the main menu for training purposes
optional arguments.
These are the existing flags:

 [-h] [-mm] [-b BLACK_PLAYER] [-w WHITE_PLAYER][-depth_b EXPECTIMAX_DEPTH_B]
 
 [-depth_w EXPECTIMAX_DEPTH_W] [-num_of_games NUM_OF_GAMES] [-delay DELAY]
 
 [-gm GAME_MODE] [-lpb LOAD_PATH_BLACK][-lpw LOAD_PATH_WHITE] [-spb SAVE_PATH_BLACK] [-spw SAVE_PATH_WHITE] [-lw LEARNING_WHITE] [-lb LEARNING_BLACK] [-gui GUI_ON]
 
For more details on what each flag does please run python game_engine.py --help in the command prompt.

The flags that are most relevant are:

**_-gm 0/ 1_** to run the game either with or without the main menu option

**_-b / -w (2-6)_** select the player for black or white.

**_-gui 0/1_** either turn the gui on or off

**_-lpb, -spb, -lb_** control parameters for from where the black player  will load his weights, where he will save his weights, and whether he will learn or not at all (by default the black player will not learn and will load the default DQN weights). Equivalent flags have been made for the white player.

Note that a human player cannot be selected in the game mode without the main menu.

## Created by

Tom Marom

Daniel Deygin

Edan Patt
