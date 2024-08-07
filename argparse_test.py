import argparse


def get_args():

    description="""Easy-to-use, pixel quantification """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-w", "--math",     dest="math",        action="store", required=False,  type=str, nargs="*", default=['mean'],   help='Set the math operation (default: mean)')
    inputs.add_argument("-q", "--quantile", dest="quantile",    action="store", required=False,  type=str, nargs="*", default=None,     help='Set the quantile')
    return parser.parse_args()


def checks(args):
    if isinstance(args.math, str):
        print("Math is a str")
    elif isinstance(args.math, list):
        print("Math is a list")

    #quantile
    if args.quantile is not None:
        if isinstance(args.quantile, str):
            print("Quantile is a str")
        elif isinstance(args.quantile, list):
            print("Quantile is a list")
    else:
        print("No quantile")

    args.quantile = [float(string) for string in args.quantile]
    for q in args.quantile:
        print(type(q))

def main():
    args = get_args()
    checks(args)
    for q in args.quantile:
        print(type(q))

if __name__ == "__main__":
    main()