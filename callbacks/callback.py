class CallBacks:
    def on_train_begin(self, model, optimizer):
        pass

    def on_train_end(self, model, optimizer):
        pass

    def on_epoch_begin(self, epoch_idx, model, optimizer):
        pass

    def on_epoch_end(self, epoch_idx, model, optimizer, val_loss=None, val_metric=None):
        pass

    def on_batch_begin(self, batch_idx, model, optimizer):
        pass

    def on_batch_end(self, batch_idx, model, optimizer, batch_loss, batch_metric):
        pass
