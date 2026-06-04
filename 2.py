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
dictionary_loop()