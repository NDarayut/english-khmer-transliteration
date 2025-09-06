import inference

def print_banner():
    print("=" * 50)
    print(" Khmer-English Transliteration Shell ".center(50, " "))
    print("=" * 50)

def main():
    print_banner()
    print("Type 'exit' to quit.\n")
    while True:
        eng_text = input("Transliterate> ")
        if eng_text.strip().lower() == 'exit':
            print("\nExiting...!")
            break
        khm_text = inference.transliterate_with_dict(eng_input=eng_text)
        print("Transliterated Text: \033[92m{}\033[0m\n".format(khm_text[0] if khm_text else ""))

if __name__ == "__main__":
    main()