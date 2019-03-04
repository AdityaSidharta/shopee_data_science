from fastai.text import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils.logger import logger


def get_accuracy_score(learn_class, ds_type, df):
    true_dict = {key: value for key, value in enumerate(learn_class.data.classes)}
    preds = learn_class.get_preds(ds_type=ds_type, ordered=True, with_loss=True)
    preds = to_np(preds[0]).argmax(axis=1)
    final_preds = np.vectorize(true_dict.get)(preds)
    return accuracy_score(df['label'], final_preds)


def get_prediction(learn_class, ds_type):
    preds = learn_class.get_preds(ds_type=ds_type, ordered=True, with_loss=True)
    preds = to_np(preds[0])
    pred_df = pd.DataFrame(preds, columns=learn_class.data.classes)
    return pred_df


def fastai_prediction(train_df, test_df, columns, data_lm, RESULT_PATH):
    result_dict = {}
    for column in columns:
        logger.info('Performing prediction on Topic : {}'.format(column))
        df_trn = train_df[[column, 'title']].copy()

        df_tst = test_df.copy()
        df_tst[column] = np.nan
        df_tst = df_tst[[column, 'title']]

        df_trn.columns = ['label', 'text']
        df_tst.columns = ['label', 'text']

        df_trn = df_trn.dropna()

        df_trn['label'] = df_trn['label'].astype(str)
        df_tst['label'] = df_tst['label'].astype(str)

        df_trn, df_val = train_test_split(df_trn, test_size=0.2)
        data_class = TextClasDataBunch.from_df('', train_df=df_trn, valid_df=df_val, test_df=df_tst,
                                               vocab=data_lm.vocab)

        learn_class = text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.5)
        learn_class.load_encoder(RESULT_PATH / 'encoder_lm.pkl')

        learn_class.fit_one_cycle(1, 1e-1, moms=(0.8, 0.7))
        learn_class.fit_one_cycle(10, 1e-2, moms=(0.8, 0.7))
        learn_class.unfreeze()
        learn_class.fit_one_cycle(20, 1e-2, moms=(0.8, 0.7))

        train_metric = learn_class.validate(learn_class.data.train_dl)
        val_metric = learn_class.validate(learn_class.data.valid_dl)
        train_loss = train_metric[0]
        val_loss = val_metric[0]
        train_acc = train_metric[1].item()
        val_acc = val_metric[1].item()

        logger.info('Train Loss on Topic {} : {}'.format(column, train_loss))
        logger.info('Validation Loss on Topic {} : {}'.format(column, val_loss))
        logger.info('Train Accuracy on Topic {} : {}'.format(column, train_acc))
        logger.info('Validation Accuracy on Topic {} : {}'.format(column, val_acc))

        pred_df = get_prediction(learn_class, DatasetType.Test)
        result_dict[column] = pred_df

    return result_dict
