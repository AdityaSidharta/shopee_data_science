from fastai.text import *
from sklearn.model_selection import train_test_split


def create_data_lm(full_text_df):
    full_text_df.columns = ['text']
    full_text_df['label'] = 0
    full_text_df = full_text_df[['label', 'text']]
    train_df, val_df = train_test_split(full_text_df, test_size=0.10)
    data_lm = TextLMDataBunch.from_df(path='', train_df=train_df, valid_df=val_df)
    return data_lm

def create_model_lm(data_lm):
    learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

    learn_lm.fit_one_cycle(1, 0.1, moms=(0.8, 0.7))
#    learn_lm.fit_one_cycle(10, 0.01, moms=(0.8, 0.7))
#    learn_lm.unfreeze()
#    learn_lm.fit_one_cycle(10, 0.01, moms=(0.8, 0.7))

    return learn_lm


def load_lm(data_lm, RESULT_PATH):
    learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
    learn_lm.load(RESULT_PATH / 'model_lm.pkl')
    learn_lm.load_encoder(RESULT_PATH / 'encoder_lm.pkl')
    return learn_lm
