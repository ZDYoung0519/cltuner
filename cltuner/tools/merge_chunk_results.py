import argparse



def main():
    parser = argparse.ArgumentParser(description="Extract vision hidden states")
    parser.add_argument("config", help="config file name or path.")
    parser.add_argument("--results-folder", default="", type=str, help="path to save data")
    parser.add_argument("--outputfile", default="", type=str, help="path to save data")



if __name__ == "__main__":
    main()