class Something:
    def __init__(self, age):
        self.age = age


if __name__ == "__main__":
    s = Something(10)
    breakpoint()
    s.age = 20
    print(s.age)
    s.age = "Hello"
    print(s.age)

    # Output:
    # 10
    # 20
    # Hello
    # The age is now a string, which is not what we want. We can use property to fix this.
