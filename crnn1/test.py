import sys, os
import time

sys.path.append(os.getcwd())

# crnn packages
import torch
from torch.autograd import Variable
from crnn1 import utils
from crnn1 import alphabets
from crnn1 import dataset
from crnn1.models import crnn as crnn

str1 = alphabets.alphabet

# crnn params
crnn_model_path = 'trained_models/mixed_second_finetune_acc97p7.pth'
alphabet = str1
nclass = len(alphabet)+1


# crnn文本信息识别
def crnn_recognition(cropped_image, converter, tmodel):

    # # crnn network
    # crnn_model_path = 'crnn1/trained_models/mixed_second_finetune_acc97p7.pth'
    # tmodel = crnn.CRNN(32, 1, nclass, 256)
    # if torch.cuda.is_available():
    #     tmodel = tmodel.cuda()
    # print('loading pre trained model from {0}'.format(crnn_model_path))
    # # 导入已经训练好的crnn模型
    # tmodel.load_state_dict(torch.load(crnn_model_path))
    #
    # converter = utils.strLabelConverter(alphabet)
  
    image = cropped_image.convert('L')

    w = int(image.size[0] / (280 * 1.0 / 160))
    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    tmodel.eval()
    preds = tmodel(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return sim_pred


