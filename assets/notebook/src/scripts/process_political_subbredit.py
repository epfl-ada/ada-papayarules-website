def turn_txt_file_to_lowercase(input_file: str = "data/subreddits.txt", output_file: str = "data/pol_sb.txt"):
    """
    Reads a text file containing subreddit names, converts all text to lowercase,
    and writes the result to a new text file.
    """
    # Read the content of the input file
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Convert to lowercase
    lower_text = text.lower()

    # Write the lowercase text to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(lower_text)

    print(f"Converted '{input_file}' to lowercase and saved as '{output_file}'.")