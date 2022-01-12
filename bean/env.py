from enum import Enum
from random import randint, choice
from copy import copy

# environment possible actions: swipe to left, right, up, down
class Action(Enum):
    def __str__(self):
        return self.name

    Left = 1
    Right = 2
    Up = 3
    Down = 4


class GameEnvironment:
    def __init__(self, board_size=3, target=64, initial_state=None):
        if initial_state == None:
            # start with empty board
            self.__initial_state = [-1 for n in range(board_size * board_size)]
        else:
            # copy to prevent aliassing
            self.__initial_state = copy(initial_state)

        # dynamic board size
        self.__board_size = board_size
        self.__target = target
        self.__state = self.__initial_state
        self.__possible_states = []
        # maybe to remove
        # self.__calculate_possible_states(self.__initial_state)

    # maybe to remove - iterate over all possible states
    def __calculate_possible_states(self, state):
        actions = self.get_possible_actions(state)
        for action in actions:
            new_state = copy(state)
            if state.count(X) == state.count(O):
                new_state[action] = X
            else:
                new_state[action] = O
            self.__possible_states.append(new_state)
            if not self.is_done(new_state):
                self.__calculate_possible_states(new_state)

    def reset(self):
        self.__state = self.__initial_state
        return self.__state

    # perform action on environment
    def __calculate_transition(self, action):
        if self.is_done():
            return self.__state

        # 1. change the state to reflect the move by the agent,
        # 2. merge same value tiles

        # swipe to left
        if action == Action.Left:
            self.__state = self.swipeToLeft(self.__state)
            self.__state = self.mergeToLeft(self.__state)
        # swipe to right
        elif action == Action.Right:
            self.__state = self.swipeToRight(self.__state)
            self.__state = self.mergeToRight(self.__state)
        elif action == Action.Up:
            # take transpose, swipe, then re-take transpose
            temp_state = self.transpose(self.__state)
            temp_state = self.swipeToLeft(temp_state)
            self.__state = self.transpose(temp_state)

            # take transpose, merge same tiles, then re-take transpose
            temp_state = self.transpose(self.__state)
            temp_state = self.mergeToLeft(temp_state)
            self.__state = self.transpose(temp_state)
        elif action == Action.Down:
            # take transpose
            temp_state = self.transpose(self.__state)
            temp_state = self.swipeToRight(temp_state)
            self.__state = self.transpose(temp_state)

            # take transpose, merge same tiles, then re-take transpose
            temp_state = self.transpose(self.__state)
            temp_state = self.mergeToRight(temp_state)
            self.__state = self.transpose(temp_state)

        # 3. generate a new tile on empty cells
        empty_state = self.get_possible_actions(self.__state)
        # possible generated tile values
        possible_gen_tiles = [2, 4]
        # generate new tile at random empty cell
        self.__state[choice(empty_state)] = possible_gen_tiles[randint(0, 1)]

        return self.__state

    def swipeToLeft(self, state):
        for i in range(self.__board_size):
            for j in range(self.__board_size - 1):
                # [0,2,2]: if current cell is empty, swap with the right one
                if state[(i * self.__board_size) + j] == -1:

                    # k is the offset of the first found tile
                    for k in range(1, self.__board_size - j):
                        if state[(i * self.__board_size) + (j + k)] != -1:

                            self.swap(
                                state,
                                (i * self.__board_size) + j,
                                (i * self.__board_size) + (j + k),
                            )
                            break
        return state

    def mergeToLeft(self, state):
        # merge same tiles together
        for i in range(self.__board_size):
            for j in range(self.__board_size - 1):
                current_tile = state[(i * self.__board_size) + j]
                if current_tile != -1:
                    right_tile = state[(i * self.__board_size) + (j + 1)]
                    if right_tile == current_tile:
                        # merge same tiles together
                        state[(i * self.__board_size) + j] = current_tile * 2
                        state[(i * self.__board_size) + (j + 1)] = -1
                        # shift to the left other tiles
                        for k in range(j + 1, self.__board_size - 1):
                            # current tile equal right tile
                            state[(i * self.__board_size) + (j + k)] = state[
                                (i * self.__board_size) + (j + k + 1)
                            ]
                        # last cell is empty
                        state[(i * self.__board_size) + self.__board_size - 1] = -1

        return state

    def swipeToRight(self, state):
        for i in range(self.__board_size):
            for j in reversed(range(1, self.__board_size)):
                # [2,2,0]: if current cell is empty, swap with the left one
                if state[(i * self.__board_size) + j] == -1:

                    # k is the offset of the first found tile
                    for k in range(1, j + 1):
                        if state[(i * self.__board_size) + (j - k)] != -1:

                            self.swap(
                                state,
                                (i * self.__board_size) + j,
                                (i * self.__board_size) + (j - k),
                            )
                            break
        return state

    def mergeToRight(self, state):
        # merge same tiles together
        for i in range(self.__board_size):
            for j in reversed(range(1, self.__board_size)):
                current_tile = state[(i * self.__board_size) + j]
                if current_tile != -1:
                    left_tile = state[(i * self.__board_size) + (j - 1)]
                    if left_tile == current_tile:
                        # merge same tiles together
                        state[(i * self.__board_size) + j] = current_tile * 2
                        state[(i * self.__board_size) + (j - 1)] = -1
                        # shift to the right other tiles
                        for k in reversed(range(1, j - 1)):
                            # current tile equal right tile
                            state[(i * self.__board_size) + (j - k)] = state[
                                (i * self.__board_size) + (j - k - 1)
                            ]
                        # last cell is empty (in this case first cells, e.g. state[0], state[3], state[6])
                        state[(i * self.__board_size)] = -1

        return state

    def transpose(self, array):
        row, column = self.__board_size, self.__board_size
        transposed_array = [-1 for n in range(row * column)]
        for i in range(row):
            for j in range(column):
                transposed_array[(i * row) + j] = array[i + (j * row)]

        return transposed_array

    def swap(self, state, x, y):
        z = state[x]
        state[x] = state[y]
        state[y] = z

    # unit step on environment
    def step(self, action):
        old_state = self.__state
        # state after agent action
        self.__state = self.__calculate_transition(action)
        observation = self.__state  # environment is fully observable
        done = self.is_done()
        reward = self.get_reward(self.__state)
        info = {}  # optional debug info
        return observation, done, reward, info

    # render environment (board) on CLI
    def render(self):
        print("┼───┼────┼───┼")

        for i in range(self.__board_size):
            print("│", end="")
            for j in range(self.__board_size):
                tile = ""
                if self.__state[(i * self.__board_size) + j] != -1:
                    tile = self.__state[(i * self.__board_size) + j]
                if (j == 0) or (j == self.__board_size - 1):
                    print("{0:>3}".format(tile) + "│", end="")
                else:
                    print("{0:>4}".format(tile) + "│", end="")

            print()
            print("┼───┼────┼───┼")
        print()

    # =========================================================
    # public functions for agent to calculate optimal policy
    # =========================================================

    def get_possible_states(self):
        return self.__possible_states

    # get index of empty cells
    def get_possible_actions(self, state=None):
        if state is None:
            state = self.__state
        return [n for n in range(9) if state[n] == -1]

    # determine wheter the game is over
    # either: when all cells are occupied and no more merging is possible,
    # or 2048 tile is generated
    def is_done(self, state=None):
        if state is None:
            state = self.__state

        # detect if a tile has target value (e.g. 2048)
        for n in range(self.__board_size * self.__board_size):
            if state[n] == self.__target:
                return True

        # check if all cells are occupied and no more merging is possible
        if -1 not in state:
            # no more merging is possible
            for i in range(self.__board_size - 1):
                for j in range(self.__board_size - 1):
                    if (
                        state[(i * self.__board_size) + j]
                        == state[((i + 1) * self.__board_size) + j]
                    ) or (
                        state[(i * self.__board_size) + j]
                        == state[(i * self.__board_size) + (j + 1)]
                    ):
                        return False
            # check bottom row
            for j in range(self.__board_size - 1):
                if (
                    state[((self.__board_size - 1) * self.__board_size) + j]
                    == state[((self.__board_size - 1) * self.__board_size) + (j + 1)]
                ):
                    return False

            # check rightmost column
            for i in range(self.__board_size - 1):
                if (
                    state[(i * self.__board_size) + self.__board_size - 1]
                    == state[((i + 1) * self.__board_size) + self.__board_size - 1]
                ):
                    return False

            return True

        return False

    # Reward R(s) for every possible state
    def get_reward(self, state):
        # detect tile with target value (e.g. 2048 tile)
        for n in range(self.__board_size * self.__board_size):
            if state[n] == self.__target:
                return 1
        return -1

    # returns the Transition Probability P(s'| s, a)
    # def get_transition_prob(self, action, new_state, old_state=None):
    # if old_state is None:
    #     old_state = self.__state
    # # returns the Transition Probability P(s'| s, a)
    # # with s = old_state, a = action and s' = new_state

    # # if the game is over, no transition can take place
    # if self.is_done(old_state):
    #     return 0.0

    # # the position of the action must be empty
    # if old_state[action] != E:
    #     return 0.0

    # # state after placing X
    # state_after_X = copy(old_state)  # avoid unwanted changed by reference
    # state_after_X[action] = X

    # # check if game is done
    # if self.is_done(state_after_X) and state_after_X == new_state:
    #     return 1.0

    # # game is not done: calculate all possible states of the opponent
    # possible_new_states = []
    # possible_opponent_actions = self.get_possible_actions(state_after_X)
    # for action in possible_opponent_actions:
    #     possible_new_state = copy(state_after_X)
    #     possible_new_state[action] = O
    #     possible_new_states.append(possible_new_state)
    # if new_state not in possible_new_states:
    #     return 0.0

    # # transition is possible, apply strategy:
    # # random opponent, probability is 1 / (# of E before placing the new O)
    # prob = 1 / (len(possible_new_states))
    # return prob
