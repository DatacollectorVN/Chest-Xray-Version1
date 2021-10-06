import urllib
import requests
import argparse
class Cfg(object):
  
    def __init__(self):
        super(Cfg, self).__init__()
        self.dim = 512
        self.batch= 8
        self.steps= 500
        self.epochs= 10
        self.model_url = ['https://github.com/DatacollectorVN/Chest_Xray_ver_1/releases/download/V1/resnet101_csv_85.h5']
    
    def down_model_ver_1(self, destination):
        model_url = self.model_url[0]
        print ('Start to download, this process take a few minutes')
        urllib.request.urlretrieve(model_url, destination)
        print("Downloaded pretrained model- {} to-'{}'".format(model_url, destination))

def main(dest):
    cfg = Cfg()
    cfg.down_model_ver_1(destination = dest)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', help= 'Destination to save model', type= str,
                        default= 'Keras_retinanet/snapshots/pretrain_model.h5')

    args = parser.parse_args()
    main(dest= args.dest)
