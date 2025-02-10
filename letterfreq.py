def count_letter_frequencies():
    first_letter_freq = {}
    last_letter_freq = {}
    
    with open('abc.txt', 'r') as file:
        for line in file:
            if line.startswith('  - '):
                country = line[4:].strip()  
                first = country[0].upper()
                first_letter_freq[first] = first_letter_freq.get(first, 0) + 1
                last = country[-1].upper()
                last_letter_freq[last] = last_letter_freq.get(last, 0) + 1

    print("First Letter Frequencies:")
    print("------------------------")
    for letter, count in sorted(first_letter_freq.items()):
        print(f"{letter}: {count}")
    
    print("\nLast Letter Frequencies:")
    print("------------------------")
    for letter, count in sorted(last_letter_freq.items()):
        print(f"{letter}: {count}")

if __name__ == "__main__":
    count_letter_frequencies()