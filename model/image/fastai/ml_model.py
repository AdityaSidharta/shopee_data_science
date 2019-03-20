from fastai.vision import *

from utils.logger import logger


def fastai_prediction(train_df, test_df, test_folder, column_list, path, topic):
    result_dict = {}
    for column in column_list:
        df_trn = train_df[['image_path', column]].copy()
        df_trn.columns = ['name', 'label']
        df_trn = df_trn[~pd.isnull(df_trn['label'])]
        df_trn['label'] = df_trn['label'].astype(str)

        tfms = get_transforms()

        data = ImageDataBunch.from_df(path, df=df_trn, size=224, num_workers=6, ds_tfms=tfms, valid_pct=0.0001)
        data.normalize()

        learn = create_cnn(data, models.resnet34, metrics=accuracy)
        learn.fit(1)
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

        learn.export(fname='{}_{}_export.pkl'.format(topic, column))

        test = ImageList.from_folder(path / test_folder)
        final_learn = load_learner(path, test=test, fname='{}_{}_export.pkl'.format(topic, column))
        test_items = final_learn.data.label_list.test.items
        test_preds = final_learn.get_preds(ds_type=DatasetType.Test)
        pred_df = pd.DataFrame(to_np(test_preds[0]), columns=learn.data.classes)

        pred_df['image_path'] = test_items
        pred_df['image_path'] = pred_df['image_path'].apply(lambda y: str(y).split('/')[-1])

        test_df['image_path'] = test_df['image_path'].apply(lambda y: str(y).split('/')[-1])

        merge_df = test_df[['image_path']].merge(pred_df, on='image_path', how='left', validate='1:1')
        merge_df = merge_df.drop(columns=['image_path'])

        result_dict[column] = merge_df
    return result_dict

