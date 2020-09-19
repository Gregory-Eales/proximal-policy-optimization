from pytorch_lightning import Trainer, seed_everything




class AlgorithmTrainer(Trainer):

    """
    this class overides the trainer training loop in order to add custom
    reinforcement learning functionality
    """

    def __init__(self):
        super(AlgorithmTrainer, self).__init__()

    @override
    def train(self):
        self.run_sanity_check(self.get_model())

        # enable train mode
        model = self.get_model()
        model.train()
        torch.set_grad_enabled(True)

        # reload data when needed
        self.train_loop.reset_train_val_dataloaders(model)

        # hook
        self.train_loop.on_train_start()

        try:
            # run all epochs
            for epoch in range(self.current_epoch, self.max_epochs):

                # reset train dataloader
                if self.reload_dataloaders_every_epoch:
                    self.reset_train_dataloader(model)

                # hook
                self.train_loop.on_train_epoch_start(epoch)

                # run train epoch
                self.train_loop.run_training_epoch()

                if self.max_steps and self.max_steps <= self.global_step:

                    # hook
                    self.train_loop.on_train_end()
                    return

                # update LR schedulers
                self.optimizer_connector.update_learning_rates(interval='epoch')

                # early stopping
                met_min_epochs = epoch >= self.min_epochs - 1
                met_min_steps = self.global_step >= self.min_steps if self.min_steps else True

                if self.should_stop:
                    if (met_min_epochs and met_min_steps):
                        self.train_loop.on_train_end()
                        return
                    else:
                        log.info('Trainer was signaled to stop but required minimum epochs'
                                 f' ({self.min_epochs}) or minimum steps ({self.min_steps}) has'
                                 ' not been met. Training will continue...')

            # hook
            self.train_loop.on_train_end()

        except KeyboardInterrupt:
            rank_zero_warn('Detected KeyboardInterrupt, attempting graceful shutdown...')

            # user could press ctrl+c many times... only shutdown once
            if not self.interrupted:
                self.interrupted = True
                self._state = TrainerState.INTERRUPTED
                self.on_keyboard_interrupt()

                # hook
                self.train_loop.on_train_end()