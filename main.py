# EUROPEAN LOOKBACK PUT OPTIONS PRICING WITH CRR BINOMIAL TREE

import numpy as np
import math

# Initial parameters for our tree
PROBABILITY_UP = 0.5073
TREE_DEPTH = 4
DISCOUNT_FACTOR = 0.06
INITIAL_PRICE = 50
UP_FACTOR = 1.1224
TIME = 0.25  # Time to maturity in years

# Calculate derived parameters
N = TREE_DEPTH - 1
DOWN_FACTOR = 1 / UP_FACTOR
D_T = TIME / N

# Debug mode flag (0 off, 1 shows a few steps, 2 EXTREME)
DEBUG_MODE = 0


def create_tree_prices(tree_depth, initial_price, up_factor):
    """
    Create the price tree for a binomial model.

    Args:
        tree_depth (int): The depth of the tree.
        initial_price (float): The initial price.
        up_factor (float): The factor by which the price increases.

    Returns:
        numpy.ndarray: The price tree as a 2D numpy array.
    """
    tree = np.zeros((tree_depth, tree_depth))
    tree[0, 0] = initial_price

    # Loop over tree rows and columns to calculate prices
    for i in range(1, tree_depth):
        for j in range(i + 1):
            if j == 0:
                tree[i, 0] = tree[i - 1, 0] / up_factor
            else:
                tree[i, j] = tree[i - 1, j - 1] * up_factor

    return tree


def calculate_S_max(tree, td):
    """
    Calculate the maximum stock price at each step.

    Args:
        tree (numpy.ndarray): The price tree.
        td (int): The depth of the tree.

    Returns:
        list: List of maximum stock prices at each step.
    """
    S_max = [tree[0][0]]
    for i in range(1, td-1):
        S_max.append(tree[i-1][i-1])
        S_max.append(tree[i][i])
    S_max.append(tree[td-1][td-1])
    return S_max


def calculate_payoffs(tree, S_max):
    """
    Calculate the possible payoffs at the last period.

    Args:
        tree (numpy.ndarray): The price tree.
        S_max (list): List of maximum stock prices at each step.

    Returns:
        list: List of payoffs at the last period.
            Is also the last column of the value tree actually. But we don't include it.
    """
    tree_depth = len(tree)

    payoffs = [max(0, S_max[0] - tree[tree_depth-1][0])]
    for i in range(1, tree_depth-1):
        for j in range(2):
            payoffs.append(max(0, S_max[(2*i-1)+j] - tree[tree_depth-1][i]))
    payoffs.append(max(0, S_max[len(S_max)-1] -
                   tree[tree_depth-1][tree_depth-1]))
    return payoffs


def backward_induction(tree, td, payoffs, p, r, dt):
    """
    Create the value tree from the price tree and payoffs.

    Args:
        tree (numpy.ndarray): The price tree.
        td (int): The depth of the tree.
        payoffs (list): List of payoffs at each step.
        p (float): The probability of an upward price movement.
        r (float): The discount rate.
        dt (float): The time step.

    Returns:
        numpy.ndarray: The value tree as a 3D numpy array.
    """
    tree = np.zeros((td-1, td, 2))
    # tree [period] [node] [2 possible values]

    # Calculate with payoffs
    tree[td-2][0][0] = (payoffs[0]*(1-p)+payoffs[1]*p)*math.exp(-r*dt)
    for i in range(1, td-2):
        # oi : offset index because for each node we need 2 values from payoffs
        oi = 2*i-1
        tree[td-2][i][0] = (payoffs[oi]*(1-p) +
                            payoffs[oi+2]*p)*math.exp(-r*dt)
        tree[td-2][i][1] = (payoffs[oi+1]*(1-p) +
                            payoffs[oi+2]*p)*math.exp(-r*dt)
    tree[td-2][td-2][0] = (payoffs[len(payoffs)-2]*(1-p) +
                           payoffs[len(payoffs)-1] * p)*math.exp(-r*dt)

    if DEBUG_MODE == 2:
        print("value_tree (period= ", td-2, "):\n", tree[td-2])

    for i in range(td-3, -1, -1):
        tree[i][0][0] = (tree[i+1][0][0]*(1-p)+tree[i+1]
                         [1][0]*p)*math.exp(-r*dt)
        for j in range(1, i):
            tree[i][j][0] = (tree[i+1][j][0]*(1-p) +
                             tree[i+1][j+1][0]*p)*math.exp(-r*dt)
            tree[i][j][1] = (tree[i+1][j][1]*(1-p) +
                             tree[i+1][j+1][0]*p)*math.exp(-r*dt)
        if (i != 0):
            tree[i][i][0] = (tree[i+1][i][1]*(1-p) +
                             tree[i+1][i+1][0] * p)*math.exp(-r*dt)

        if DEBUG_MODE == 2:
            print("value_tree (period= ", i, "):\n", tree[i])
    return tree


def create_tree_values(price_tree, r, p, dt, tree_depth):
    """
    Generate the values tree from the price tree.

    Args:
        price_tree (numpy.ndarray): The price tree.
        r (float): The discount rate.
        p (float): The probability of an upward price movement.
        dt (float): The time step.
        tree_depth (int): The depth of the tree.

    Returns:
        numpy.ndarray: The values tree as a 3D numpy array.
    """
    S_max = calculate_S_max(price_tree, tree_depth)
    if DEBUG_MODE == 2:
        print("S_max: ", S_max)
    payoffs = calculate_payoffs(price_tree, S_max)
    if DEBUG_MODE == 2:
        print("payoffs: ", payoffs)
    values_tree = backward_induction(
        price_tree, tree_depth, payoffs, p, r, dt)

    return values_tree


price_tree = create_tree_prices(TREE_DEPTH, INITIAL_PRICE, UP_FACTOR)

if DEBUG_MODE:
    print(f'Price Tree:\n{price_tree}\n')

values_tree = create_tree_values(
    price_tree, DISCOUNT_FACTOR, PROBABILITY_UP, D_T, TREE_DEPTH)

if DEBUG_MODE:
    print(f'Values Tree:\n{values_tree}')
else:
    print(values_tree[0][0][0])
