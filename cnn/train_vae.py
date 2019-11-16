import vae
import utils
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Convolutional VAE for CelebA')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--report_freq', type=int, default=100,
                        help='how many batches to wait before showing progress (default: 100)')
    parser.add_argument('--data_path', type=str, default='~',
                        help='path to downloaded dataset (default: ~)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='path to a saved model, for resuming training (default: None)')
    args = parser.parse_args()

    train_loader, test_loader = utils.load_celebA(data_path = args.data_path)
    vae.train(
        train_loader,
        model_path = args.model_path,
        num_epochs = args.epochs,
        seed = args.seed,
        report_freq = args.report_freq,
    )
