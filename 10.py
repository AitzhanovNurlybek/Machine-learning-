# Task 1: Average Grade Calculator

def calculate_average(grades):

    if len(grades) == 0:
        return 0
    return sum(grades) / len(grades)


def task1():
    print("=== Task 1: Average Grade Calculator ===")
    grades = []

    for i in range(5):
        while True:
            try:
                grade = float(input(f"Enter grade for student {i + 1}: "))
                grades.append(grade)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    average = calculate_average(grades)
    print(f"Group average grade: {average}\n")


# Task 2: Product Manager


def add_product(products, name, price):
    """Adds a new product with its price to the dictionary."""
    products[name] = price
    print("Product added.")


def remove_product(products, name):
   
    if name in products:
        del products[name]
        print("Product removed.")
    else:
        print("Product not found.")


def view_products(products):
    
    if not products:
        print("No products available.")
    else:
        print("Products list:")
        for name, price in products.items():
            print(f"{name}: {price}")


def task2():
    print("=== Task 2: Product Manager ===")
    products = {}

    while True:
        action = input("Choose action: add, remove, view, quit\n> ").lower()

        if action == "add":
            name = input("Enter product name: ")
            while True:
                try:
                    price = float(input("Enter product price: "))
                    add_product(products, name, price)
                    break
                except ValueError:
                    print("Invalid price. Please enter a number.")

        elif action == "remove":
            name = input("Enter product name to remove: ")
            remove_product(products, name)

        elif action == "view":
            view_products(products)

        elif action == "quit":
            print("Exiting Product Manager.\n")
            break

        else:
            print("Invalid action. Please choose add, remove, view, or quit.")


# Task 3: Refactored Maximum Finde

def find_max(numbers):
    
    if len(numbers) == 0:
        return None

    maximum = numbers[0]
    for num in numbers:
        if num > maximum:
            maximum = num
    return maximum


def task3():
    print("=== Task 3: Find Maximum Number ===")
    numbers = []

    while True:
        user_input = input("Enter numbers separated by spaces: ")

        try:
            numbers = list(map(float, user_input.split()))
            if len(numbers) == 0:
                print("Please enter at least one number.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter only numbers.")

    maximum = find_max(numbers)
    print(f"Maximum number: {maximum}\n")


# Main Program

def main():
    while True:
        print(" Assignment 5 Menu")
        print("1 - Task 1: Average Grade Calculator")
        print("2 - Task 2: Product Manager")
        print("3 - Task 3: Find Maximum Number")
        print("4 - Exit")

        choice = input("Choose task: ")

        if choice == "1":
            task1()
        elif choice == "2":
            task2()
        elif choice == "3":
            task3()
        elif choice == "4":
            print("Program finished.")
            break
        else:
            print("Invalid choice. Try again.\n")


main()