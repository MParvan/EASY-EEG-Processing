import argparse
from load_data import load_eeg_data
from preprocessing import preprocess_data
from models import get_model
from evaluation import evaluate_model
import torch

def main():
    parser = argparse.ArgumentParser(
        description="EEG Classification Framework"
    )
    parser.add_argument('--data-path', type=str, required=True, help='Path to EEG data')
    parser.add_argument('--model', type=str, default='cnn', help='Model type (e.g., cnn, lstm)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    args = parser.parse_args()

    print("Loading data...")
    X_train, y_train, X_test, y_test = load_eeg_data(args.data_path)
    
    print("Preprocessing data...")
    X_train, X_test = preprocess_data(X_train, X_test)
    
    print(f"Building model: {args.model}")
    model = get_model(args.model)
    model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Training...")
    for epoch in range(args.epochs):
        model.train()
        # Add your training loop here (batching, forward, backward, optimizer step)
        # This is a placeholder for modularity
        pass

    print("Evaluating...")
    evaluate_model(model, X_test, y_test, device=args.device)

if __name__ == "__main__":
    main()