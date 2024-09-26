from itertools import permutations

def is_valid(solution, letters, word1, word2, result):
    """Check if the current solution is valid."""
    letter_to_digit = {letters[i]: solution[i] for i in range(len(letters))}
    
    # Convert letters to digits
    num1 = int(''.join(str(letter_to_digit[char]) for char in word1))
    num2 = int(''.join(str(letter_to_digit[char]) for char in word2))
    num_result = int(''.join(str(letter_to_digit[char]) for char in result))

    return num1 + num2 == num_result

def solve_cryptarithmetic(word1, word2, result):
    """Solve the cryptarithmetic puzzle given two words and their result."""
    letters = set(word1 + word2 + result)  # Unique letters from both words and the result
    if len(letters) > 10:
        return None  # More than 10 unique letters means no solution

    letters = list(letters)  # Convert to list for indexing
    for perm in permutations(range(10), len(letters)):
        # Check leading digit constraints
        if (perm[letters.index(word1[0])] == 0 or
            perm[letters.index(word2[0])] == 0 or
            perm[letters.index(result[0])] == 0):
            continue
        if is_valid(perm, letters, word1, word2, result):
            return {letters[i]: perm[i] for i in range(len(letters))}
    return None  # No solution found

if __name__ == "__main__":
    # Input two words and their sum
    word1 = input("Enter the first word: ").strip()
    word2 = input("Enter the second word: ").strip()
    result = input("Enter the result word: ").strip()

    solution = solve_cryptarithmetic(word1, word2, result)
    if solution:
        print("Solution found:")
        for letter, digit in solution.items():
            print(f"{letter} = {digit}")
    else:
        print("No solution exists.")















def solve_cryptarithm(words, result):unique_letters = ''.join(set(''.join(words) + result))if len(unique_letters) > 10:print("Too many unique letters for a single-digit solution.")returndigits = '0123456789'for perm in itertools.permutations(digits, len(unique_letters)):letter_to_digit = dict(zip(unique_letters, perm))def word_to_number(word):return int(''.join(letter_to_digit[letter] for letter in word))C B I TLaboratory Record Roll No. 160122749034Of AIML LAB Experiment No. Sheet No. 15Date. if any(letter_to_digit[word[0]] == '0' for word in words + [result]):continuesum_words = sum(word_to_number(word) for word in words)if sum_words == word_to_number(result):print("Solution found!")for word in words:print(f"{word} = {word_to_number(word)}")print(f"{result} = {word_to_number(result)}")print(f"Letter to Digit Mapping: {letter_to_digit}")returnprint("No solution found.")input_words = input("Enter the words to sum (space-separated): ").upper().split()input_result = input("Enter the result word: ").upper()solve_cryptarithm(input_words, input_result)
