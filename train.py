import argparse

def train(lr, momentum, num_hidden, sizes, activation, loss, opt, batch_size, anneal, save_dir, expt_dir, train_file, test_file):
    # Your training code here
    
    
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.5, help="Momentum value")
    parser.add_argument("--num_hidden", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("--sizes", type=str, default="100,100,100", help="Hidden layer sizes")
    parser.add_argument("--activation", type=str, default="sigmoid", help="Activation function")
    parser.add_argument("--loss", type=str, default="sq", help="Loss function")
    parser.add_argument("--opt", type=str, default="adam", help="Optimizer")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--anneal", type=bool, default=True, help="Annealing")
    parser.add_argument("--save_dir", type=str, default="pa1/", help="Save directory")
    parser.add_argument("--expt_dir", type=str, default="pa1/exp1/", help="Experiment directory")
    parser.add_argument("--train", type=str, default="train.csv", help="Training file")
    parser.add_argument("--test", type=str, default="test.csv", help="Testing file")

    args = parser.parse_args()

    train(
        args.lr,
        args.momentum,
        args.num_hidden,
        [int(size) for size in args.sizes.split(",")],
        args.activation,
        args.loss,
        args.opt,
        args.batch_size,
        args.anneal,
        args.save_dir,
        args.expt_dir,
        args.train,
        args.test
    )
