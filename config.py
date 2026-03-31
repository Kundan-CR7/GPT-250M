class Config:
    #model architecture
    vocab_size = 50000
    block_size = 512

    n_layer = 24
    n_head = 16
    n_embd = 1024

    #training parameters
    batch_size = 32
    learning_rate = 3e-4
    max_iters = 200000

    #device
    device = "cuda"

    #evaluation
    eval_interval = 500


