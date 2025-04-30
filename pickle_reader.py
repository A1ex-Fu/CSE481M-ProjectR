import pickle
import argparse

def main(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    for paper, result in data.items():
        print(f"Paper: {paper}")
        print(f"Result: {result}\n{'-' * 40}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', required=True, help='Path to processed_docs.pkl')
    args = parser.parse_args()
    main(args.pkl_path)

