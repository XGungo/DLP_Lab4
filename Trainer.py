class Trainer(object):

    def __init__(self, encoder, decoder, data_transformer, learning_rate, use_cuda,
                 checkpoint_name=config.checkpoint_name,
                 teacher_forcing_ratio=config.teacher_forcing_ratio):

        self.encoder = encoder
        self.decoder = decoder

        # record some information about dataset
        self.data_transformer = data_transformer
        self.vocab_size = self.data_transformer.vocab_size
        self.PAD_ID = self.data_transformer.PAD_ID
        self.use_cuda = use_cuda

        # optimizer setting
        self.learning_rate = learning_rate
        self.optimizer= torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # self.criterion = torch.nn.NLLLoss(ignore_index=self.PAD_ID, size_average=True)

        self.checkpoint_name = checkpoint_name

    def train(self, num_epochs, batch_size, pretrained=False):


        for epoch in range(0, num_epochs):
            mini_batches = self.data_transformer.mini_batches(batch_size=batch_size)
            for input_batch, target_batch in mini_batches:
                self.optimizer.zero_grad()
                decoder_outputs, decoder_hidden = self.model(input_batch, target_batch)
                # calculate the loss and back prop.
                cur_loss = self.get_loss(decoder_outputs, target_batch[0])
                cur_loss.backward()
                # optimize
                self.optimizer.step()

        self.save_model()

    def get_loss(self, decoder_outputs, targets):
        b = decoder_outputs.size(1)
        t = decoder_outputs.size(0)
        targets = targets.contiguous().view(-1)  # S = (B*T)
        decoder_outputs = decoder_outputs.view(b * t, -1)  # S = (B*T) x V
        return self.criterion(decoder_outputs, targets)