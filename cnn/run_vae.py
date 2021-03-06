import vae
import utils
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Convolutional VAE for CelebA')
    parser.add_argument('--train', action='store_true', default=False,
                        help='whether to run training (default: False)')
    parser.add_argument('--test', action='store_true', default=False,
                        help='whether to run testing (default: False)')
    parser.add_argument('--batch_size', type=int, default=64,
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
                        help=f'path to a saved model for resuming training and/or performing testing (default: None)')
    parser.add_argument('--save_examples', action='store_true', default=False,
                        help='whether to save example images to files during training (default: False)')
    args = parser.parse_args()

    if args.train or args.test:
        train_loader, test_loader = utils.load_celebA(data_path = args.data_path, batch_size = args.batch_size)
    else:
        print('Neither --train nor --test specified. Quitting.')
        exit(0)

    if args.train:
        vae.train(
            train_loader,
            model_path = args.model_path,
            num_epochs = args.epochs,
            seed = args.seed,
            report_freq = args.report_freq,
            save_examples = args.save_examples,
        )

    if args.test:
        vae.test(
            test_loader,
            args.model_path if args.model_path else vae.DEFAULT_MODEL_PATH,
            report_freq = args.report_freq,
            save_examples = args.save_examples,
        )
