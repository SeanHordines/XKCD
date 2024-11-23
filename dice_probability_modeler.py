from math import comb, gcd
from itertools import combinations_with_replacement, product
from collections import Counter
from functools import lru_cache

RPG_DICE_SET = [4, 6, 8, 10, 12, 20]

def hypergeometric_pmf(total_population, total_successes, draws, target_successes):
    """
    Calculate the probability mass function for the hypergeometric distribution.
    https://en.wikipedia.org/wiki/Hypergeometric_distribution
    p(k) = KCk * (N-K)C(n-k) / NCn

    Parameters:
        total_population (int): Total number of items in the population (N).
        total_successes (int): Number of successful items in the population (K).
        draws (int): Number of items drawn (n).
        target_successes (int): Number of successful items drawn (k).

    Returns:
        p_value (float): Probability of observing exactly target_successes number
            of successes.
        numerator (int): Reduced top half of p_value as a fraction.
        denominator (int): Reduced bottom half of p_value as a fraction.
    """

    # input validation
    if (total_successes > total_population) or \
        (draws > total_population) or \
        (total_population < 1):
        raise ValueError("Invalid Parameters")

    if (target_successes > draws) or \
        (target_successes > total_successes) or \
        (target_successes < 0):
        return 0.0

    # using the formula from above
    term_1 = comb(total_successes, target_successes)
    term_2 = comb(total_population - total_successes, draws - target_successes)
    term_3 = comb(total_population, draws)
    p_value = term_1 * term_2 / term_3

    # reduce the fraction form
    fraction_gcd = gcd(term_1 * term_2, term_3)
    numerator = (term_1 * term_2) // fraction_gcd
    denominator = term_3 // fraction_gcd

    return (p_value, numerator, denominator)

def hypergeometric_cdf(total_population, total_successes, draws, target_successes, inequality="eq"):
    """
    Calculate the cumulative probability for the hypergeometric distribution.

    Parameters:
        total_population (int): Total number of items in the population (N).
        total_successes (int): Number of successful items in the population (K).
        draws (int): Number of items drawn (n).
        target_successes (int): Number of successful items drawn (k).
        inequality (str): Inequality condition. Options are:
        "eq": P(X = k)
        "lte": P(X <= k)
        "lt":  P(X < k)
        "gte": P(X >= k)
        "gt":  P(X > k)
        "neq": P(X != k)

    Returns:
        p_value (float): Cumulative probability based on the inequality condition.
    """

    max_successes = min(draws, total_successes)
    k_values = []

    if inequality == "eq":
        k_values = [target_successes]
    elif inequality == "lte":
        k_values = range(0, target_successes + 1)
    elif inequality == "lt":
        k_values = range(0, target_successes)
    elif inequality == "gte":
        k_values = range(target_successes, max_successes + 1)
    elif inequality == "gt":
        k_values = range(target_successes + 1, max_successes + 1)
    elif inequality == "neq":
        k_values = list(range(0, target_successes)) + \
            list(range(target_successes + 1, max_successes + 1))
        # ugly compared to the others but whatever
    else:
        raise ValueError("Invalid Inequality Option")

    outcomes = [hypergeometric_pmf(total_population, total_successes, draws, k)[0]
        for k in k_values]
    p_value = sum(outcomes)
    return p_value

def validate_distro(numerator, denominator, dice):
    """
    Returns a threshold sum where the cumulative number of dice roll outcomes
        matches a target proportion of the total possible outcomes. If no such
        threshold exists, returns 0.

    Parameters:
        numerator (int): Represents the proportion of cases to check.
        denominator (int): Representing the proportion of cases to check.
        dice (list of int): A list where each element represents the number of
            sides on a particular die.

    Returns:
        threshold (int): The threshold sum (or 0 if no threshold exists) that
        divides the distribution in the specified proportion.
    """

    num_cases = 1
    for d in dice:
        num_cases *= d

    if num_cases % denominator:
        return 0

    target_cases = numerator * num_cases / denominator

    rolls = product(*[range(1, d + 1) for d in dice])
    distro = Counter([sum(roll) for roll in rolls])
    distro = sorted(distro.items(), reverse=True)

    cum_cases = 0
    for threshold, freq in distro:
        cum_cases += freq
        if cum_cases == target_cases:
            return threshold
        elif cum_cases > target_cases:
            return 0

def find_dice(p_value, numerator, denominator, dice_set = RPG_DICE_SET, limit = 20):
    """
    Finds the smallest combination of dice that are capable of modeling the
    given probability. The probability must be expressed as fraction

    Parameters:
        p_value (float): The probability to be modeled by dice
        numerator (int): Reduced top half of p_value as a fraction.
        denominator (int): Reduced bottom half of p_value as a fraction.
        dice_set (List): Dice availible for use expressed as the number of sides
            for each type of die. Defaults to standard RPG dice.
        limit (int): The maximum size of the dice combinations to check before
            canceling the operation.

    Returns:
        solutions (List): Contains tuples of dice with their corresponding thresholds.
    """

    N = 1
    solutions = []
    while N <= limit and not solutions:
        for dice in combinations_with_replacement(dice_set, N):
            threshold = validate_distro(numerator, denominator, dice)
            if threshold:
                solutions.append((dice, threshold))
        N += 1

    return solutions

if __name__ == "__main__":
    solutions = find_dice(*hypergeometric_pmf(10, 5, 2, 2))

    print("Smallest combinations of dice that yeild the matching hypergeometric distribution:")
    if solutions == []:
        print("None")
    else:
        n = 1
        for s in solutions:
            print(f"Solution {n}")
            print(f"    Dice: {s[0]}")
            print(f"    Threshold: {s[1]}")
            n += 1
