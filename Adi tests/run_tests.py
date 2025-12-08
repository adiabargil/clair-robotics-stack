import test_2

def main():
    # Run the test from test_2.py
    for i in range(1, 4):
        test_2.main(i)
        print(f"Test {i} completed.")

if __name__ == "__main__":
    main()