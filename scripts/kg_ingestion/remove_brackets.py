import os

def remove_outer_square_brackets(folder_path: str):
    """
    Removes only the outer square brackets from JSON files.
    Does NOT parse JSON — preserves formatting exactly.
    Does NOT create backup files.
    """

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, filename)
        print(f"Processing: {file_path}")

        # Read file
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Remove blank lines
        clean_lines = [line.rstrip("\n") for line in lines if line.strip() != ""]

        # Ensure brackets exist
        if (
            len(clean_lines) >= 2 
            and clean_lines[0].strip() == "[" 
            and clean_lines[-1].strip() == "]"
        ):
            # Remove the first '[' and last ']'
            new_lines = clean_lines[1:-1]

            # Write modified file (overwrite)
            with open(file_path, "w", encoding="utf-8") as f:
                for line in new_lines:
                    f.write(line + "\n")

            print(" → Outer brackets removed.")
        else:
            print(" → No outer brackets found. Skipped.")


if __name__ == "__main__":
    folder = input("Enter folder path: ").strip()
    remove_outer_square_brackets(folder)
    print("\nDone.")
