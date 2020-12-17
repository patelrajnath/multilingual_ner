class HParamSet(object):
    def __init__(self, max_sts_score=5, balance_data=False, output_size=None,
                 activation='relu', hidden_layer_size=512, num_hidden_layers=1,
                 embedding_dim=256, batch_size=32, dropout=0.1, optimizer='sgd',
                 learning_rate=0.01, lr_decay_pow=1, epochs=100, seed=999,
                 max_steps=1500, patience=100, eval_each_epoch=True, number_of_tags=0,
                 vocab_size=0):
        self.max_sts_score = max_sts_score
        self.balance_data = balance_data
        self.output_size = output_size
        self.batch_size = batch_size
        self.activation = activation
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_decay_pow = lr_decay_pow
        self.epochs = epochs
        self.seed = seed
        self.max_steps = max_steps
        self.patience = patience
        self.eval_each_epoch = eval_each_epoch
        self.number_of_tags = number_of_tags
        self.vocab_size = vocab_size
