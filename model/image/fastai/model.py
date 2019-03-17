from fastai import *
from fastai.vision import *

from utils.logger import logger


def fastai_prediction(train_df, test_folder, column_list, path):
    result_dict = {}
    for column in column_list:
        df_trn = train_df[['image_path', column]].copy()
        df_trn.columns = ['name', 'label']
        df_trn = df_trn[~pd.isnull(df_trn['label'])]
        df_trn['label'] = df_trn['label'].astype(str)

        tfms = get_transforms()

        data = ImageDataBunch.from_df(path, df=df_trn, size=224, num_workers=6)
        data.normalize()

        learn = create_cnn(data, models.resnet34, metrics=accuracy)
        learn.fit(5)
        learn.unfreeze()
        learn.fit(5)

        train_metric = learn.validate(learn.data.train_dl)
        val_metric = learn.validate(learn.data.valid_dl)
        train_loss = train_metric[0]
        val_loss = val_metric[0]
        train_acc = train_metric[1].item()
        val_acc = val_metric[1].item()

        logger.info('Train Loss on Topic {} : {}'.format(column, train_loss))
        logger.info('Validation Loss on Topic {} : {}'.format(column, val_loss))
        logger.info('Train Accuracy on Topic {} : {}'.format(column, train_acc))
        logger.info('Validation Accuracy on Topic {} : {}'.format(column, val_acc))

        learn.export()

        test = ImageList.from_folder(path / test_folder)
        final_learn = load_learner(path, test=test)
        test_preds = final_learn.get_preds(ds_type=DatasetType.Test)
        pred_df = pd.DataFrame(to_np(test_preds[0]), columns=learn.data.classes)

        result_dict[column] = pred_df
    return result_dict

