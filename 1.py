import random

def temperature_converter():
    print("\nTask 1: Temperature Converter")
    temp = float(input("Enter temperature: "))
    choice = input("Convert to (C/F): ").upper()

    if choice == "F":
        result = temp * 9 / 5 + 32
        print("Result:", result)
    elif choice == "C":
        result = (temp - 32) * 5 / 9
        print("Result:", result)
    else:
        print("Invalid choice")

def guess_the_number():
    print("\nTask 2: Guess the Number")
    number = random.randint(1, 100)

    while True:
        guess = int(input("Guess the number (1-100): "))
        if guess < number:
            print("Too low")
        elif guess > number:
            print("Too high")
        else:
            print("Correct!")
            break

def fibonacci_sequence():
    print("\nTask 3: Fibonacci Sequence")
    n = int(input("Enter number of terms: "))
    a, b = 0, 1

    for _ in range(n):
        print(a, end=" ")
        a, b = b, a + b
    print()

def dictionary_loop():
    print("\nTask 4: Dictionary and Loops")
    products = {
        "apple": 2,
        "banana": 3,
        "milk": 4,
        "bread": 5
    }

    total = 0
    for key, value in products.items():
        print(key, "=", value)
        total += value

    print("Total cost:", total)

def unique_words():
    print("\nTask 5: Unique Word Counter")
    text = input("Enter a sentence: ").lower().split()
    unique = set(text)
    print("Unique words:", len(unique))

def prime_checker():
    print("\nTask 6: Prime Number Checker")
    num = int(input("Enter a number: "))

    if num <= 1:
        print("Not prime")
    else:
        prime = True
        for i in range(2, num):
            if num % i == 0:
                prime = False
                break

        if prime:
            print("Prime")
        else:
            print("Not prime")

def simple_menu():
    print("\nTask 7: Simple Menu")
    while True:
        print("\n1. Add")
        print("2. Subtract")
        print("3. Multiply")
        print("4. Exit")

        choice = input("Choose: ")

        if choice == "4":
            print("Goodbye")
            break

        if choice in ["1", "2", "3"]:
            a = float(input("Enter first number: "))
            b = float(input("Enter second number: "))

            if choice == "1":
                print("Result:", a + b)
            elif choice == "2":
                print("Result:", a - b)
            elif choice == "3":
                print("Result:", a * b)
        else:
            print("Invalid option")

def main():
    temperature_converter()
    guess_the_number()
    fibonacci_sequence()
    dictionary_loop()
    unique_words()
    prime_checker()
    simple_menu()

main()