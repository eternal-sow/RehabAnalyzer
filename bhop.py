# Простая программа для вычисления 1 + 1
# Создано Grok Build

def calculate_one_plus_one():
    """Вычисляет 1 + 1"""
    a = 1
    b = 1
    result = a + b
    return result


if __name__ == "__main__":
    print("Вычисление: 1 + 1")
    result = calculate_one_plus_one()
    print(f"Результат: {result}")
    print("Готово!")
