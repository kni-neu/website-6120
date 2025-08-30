import argparse
import csv
import sys

def create_alias(infile, outfile=None):
    """
    Reads a CSV file and creates a list of formatted aliases.
    
    Args:
        infile (str): The path to the input CSV file.
        outfile (str, optional): The path to the output file.
                                 If None, output is written to stdout.
    """
    aliases = []
    
    try:
        with open(infile, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            for row in reader:
                # Assuming the CSV columns are: _, name, _, alias, email
                # Adjust indices if your CSV has a different structure
                name, alias, email = row[1], row[3], row[4]
                
                if alias:
                    alias_line = f'{name} {alias} <{email}>'
                else:
                    alias_line = f'{name} <{email}>'
                aliases.append(alias_line)
    except FileNotFoundError:
        print(f"Error: The file '{infile}' was not found.", file=sys.stderr)
        return
    
    if outfile:
        try:
            with open(outfile, 'w', encoding='utf-8') as f:
                f.write('\n'.join(aliases) + '\n')
            print(f"Aliases successfully written to '{outfile}'.")
        except IOError as e:
            print(f"Error: Could not write to file '{outfile}'. Reason: {e}", file=sys.stderr)
    else:
        # Write to standard output
        print('\n'.join(aliases))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate a list of aliases from a CSV file.'
    )
    
    # Required argument for the input file
    parser.add_argument(
        'infile',
        help='The path to the input CSV file.'
    )
    
    # Optional argument for the output file
    parser.add_argument(
        '-o', '--outfile',
        help='The path to the output file. If not specified, output goes to stdout.'
    )
    
    args = parser.parse_args()
    
    create_alias(args.infile, args.outfile)
