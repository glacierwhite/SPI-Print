import sys
if './' not in sys.path:
	sys.path.append('./')
     
import logging
import time
import os

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
	
from models.vision_transformer import VisionTransformer
from torch.utils.tensorboard import SummaryWriter
from options import Options
import torch
from running import setup, SupervisedRunner, NEG_METRICS, validate, check_progress
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.loss import get_loss_module
from optimizers import get_optimizer
from tqdm import tqdm
import numpy as np
import utils
import scipy.io as sio

def main(config):

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device.type == "cuda":
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Define transformations for preprocessing the images
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),   # resize images
        transforms.ToTensor(),           # convert PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]) # normalize like ImageNet
    ])

    # Load dataset
    logger.info("Loading and preprocessing data ...")
    test_dataset = datasets.ImageFolder(root= config['data_dir']+"/test", transform=transform)

    logger.info("{} samples may be used for testing".format(len(test_dataset)))

    # Create model
    logger.info("Loading model from checkpoint ...")
    model = VisionTransformer(
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            hidden_dim=config['hidden_dim'],
            mlp_dim=config['mlp_dim'],
            num_classes=config['num_classes']
        )
    
    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))
    
    # Initialize optimizer

    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr = config['lr']  # current learning step
    # Load model and optimizer state
    if args.load_model:
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
    
    model.to(device)

    # # Using multiple gpus if available
    # model = torch.nn.DataParallel(model)
    # model.to(device)

    loss_module = get_loss_module(config)

    # Create DataLoaders
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=config['batch_size'],  # number of images per batch
                            shuffle=False,   # shuffle for training
                            num_workers=config['num_workers'])  # adjust for your CPU
    
    test_evaluator = SupervisedRunner(model, test_loader, device, loss_module,
                                       print_interval=config['print_interval'], console=config['console'])
    
    with torch.no_grad():
        aggr_metrics_test, per_batch_test = test_evaluator.evaluate(keep_all=True)
    del aggr_metrics_test['epoch']
    print_str = 'Test Summary: '
    for k, v in aggr_metrics_test.items():
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)

    targets = np.concatenate(per_batch_test['targets'],axis=0)
    embeddings = np.concatenate(per_batch_test['embeddings'],axis=0)

    # iterate through different threshold
    tprs = []
    fprs = []
    accs = []
    for i in range(201):
        similarity_theshold = 0.01*i - 1
        tpr = []
        fpr = []
        acc = []
        # iterate through each enrolling driver
        for enrolling_driver_index in np.unique(targets):
            number_of_enrolling_instances = config['num_instances'] if config['num_instances'] < len(np.where(targets==enrolling_driver_index)[0])//2 else len(np.where(targets==enrolling_driver_index)[0])//2
            # logger.info("Select driver {} as the legitimate driver".format(enrolling_driver_index))
            # logger.info("Using {} instances for enrollment".format(number_of_enrolling_instances))
            
            # enrollment
            template = np.mean(embeddings[np.where(targets==enrolling_driver_index)[0][:number_of_enrolling_instances]],axis=0)

            # authentication
            num_true_positive = 0
            num_true_negative = 0
            num_false_positive = 0
            num_false_negative = 0
            # iterate through each test driver
            for test_driver_index in np.unique(targets):
                similarity_score = []
                for test_instance_index in np.where(targets==test_driver_index)[0][len(np.where(targets==test_driver_index)[0])//2:]:
                    similarity_score.append(np.dot(template, embeddings[test_instance_index]) / (np.linalg.norm(template) * np.linalg.norm(embeddings[test_instance_index])))
                similarity_score = np.array(similarity_score)
                num_positive = np.sum(similarity_score>similarity_theshold)
                num_negative = similarity_score.shape[0] - num_positive
                if test_driver_index == enrolling_driver_index:
                    num_true_positive += num_positive
                    num_false_negative += num_negative
                else:
                    num_false_positive += num_positive
                    num_true_negative += num_negative
            
            tpr.append(num_true_positive/(num_true_positive+num_false_negative))
            fpr.append(num_false_positive/(num_false_positive+num_true_negative))
            acc.append((num_true_positive+num_true_negative)/(num_true_positive+num_true_negative+num_false_positive+num_false_negative))

        tprs.append(tpr)
        fprs.append(fpr)
        accs.append(acc)

    sio.savemat("./output/test_results.mat", {"tprs": tprs, "fprs": fprs, "accs": accs})

    return

if __name__ == '__main__':
     
     args = Options().parse()
     config = setup(args)
     main(config)