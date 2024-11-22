# region libraries
import os, time, socket, random, copy, traceback, logging, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision.models import vision_transformer
# endregion


from datas.dataset_getter import get_datasets
from models.model_getter import get_wrapped_model


def train(rank: int, config: dict):
    prefix = config['prefix']
    logging.basicConfig(
        filename=f"{config['prefix']}.log",
        filemode='a',
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        level=logging.INFO
    )

    # region seeds
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # endregion

    # region DataLoader
    d = get_datasets(dataset_name=config['dataset_name'], fold=config['fold'])
    trainset, valset, num_classes = d['trainset'], d['valset'], d['num_classes']

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    trainloader = DataLoader(
        trainset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], 
        pin_memory=True, worker_init_fn=seed_worker
    )
    if config.get('test_number', False):
        from torch.utils.data import RandomSampler
        sampler = RandomSampler(data_source=trainset, replacement=False, num_samples=config['test_number'])
        trainloader = DataLoader(trainset, batch_size=16, sampler=sampler)
        print(f"--子集{config['test_number']}个样本")

    valloader = DataLoader(
        valset, batch_size=config['batch_size'], shuffle=False, 
        num_workers=config['num_workers'], pin_memory=True, worker_init_fn=seed_worker
    )
    # endregion

    # region model
    model = get_wrapped_model(model_name=config['model_name'], pretrain=config['pretrain'], num_classes=num_classes)
    n_classifiers = len(model.classifiers)

    cweights = torch.nn.Parameter(torch.ones(n_classifiers))
    alpha, temperature = torch.tensor(0.8), torch.tensor(2.0)

    device = torch.device(f'cuda:{rank}')
    model.to(device)
    # endregion

    # region optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + [cweights], lr=1e-4)
    ce_loss = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    scaler = GradScaler() if config.get('amp', False) else None
    # endregion

    # writer
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f"runs/{prefix}/Accuracy")

    # Train
    patience, progress = 0, []
    bepoch, bacc, bmodel_state, bN, bvacc = 0, 0, None, n_classifiers - 1, 0
    t0 = time.time()

    try:
        for epoch in range(config['max_epochs']):
            model.train()
            tlosses, tcorrects, ttotal = [0] * n_classifiers, [0] * n_classifiers, 0

            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with autocast(device_type='cuda', enabled=config.get('amp', False)):
                    toutputs = model(inputs)

                    closses = [ce_loss(output, labels) for output in toutputs]
                    closs = sum(w * l for w, l in zip(cweights.softmax(dim=0), closses))

                    # Distillation loss
                    if epoch > 0 and config.get('distill', False):
                        i = closses.index(min(closses))

                        soft_targets = F.softmax(toutputs[i] / temperature, dim=1).detach()
                        soft_predictions = F.log_softmax(toutputs[max(0, bN-1)] / temperature, dim=1)
                        dloss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (temperature ** 2)

                        gloss = alpha * closs + (1-alpha) * dloss
                    else:
                        gloss = closs

                if scaler:
                    scaler.scale(gloss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    gloss.backward()
                    optimizer.step()

                tlosses = [tlosses[i] + closses[i].item() for i in range(n_classifiers)]
                tcorrects = [tcorrects[i] + (torch.argmax(toutputs[i], dim=-1) == labels).sum().item() for i in range(n_classifiers)]
                ttotal += inputs.size(0)

            avg_tlosses = [tloss / ttotal for tloss in tlosses]
            avg_taccs = [tcorrect / ttotal for tcorrect in tcorrects]

            # Validation
            model.eval()
            vlosses, vcorrects, vtotal = [0] * n_classifiers, [0] * n_classifiers, 0
            with torch.no_grad():
                for inputs, labels in valloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    batch_vlosses = [ce_loss(output, labels).item() for output in outputs]
                    vlosses = [vloss + batch_vloss for vloss, batch_vloss in zip(vlosses, batch_vlosses)]
                    vcorrects = [vcorrect + (torch.argmax(output, dim=-1) == labels).sum().item() for vcorrect, output in zip(vcorrects, outputs)]
                    vtotal += inputs.size(0)

            avg_vlosses = [vloss / vtotal for vloss in vlosses]
            avg_vaccs = [vcorrect / vtotal for vcorrect in vcorrects]

            if max(avg_vaccs) > bacc:
                bepoch, bacc = epoch, max(avg_vaccs)
                bN = [bacc - vacc < 0.02 for vacc in avg_vaccs].index(True)
                bvacc = avg_vaccs[bN]
                bmodel_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1

            metrics = {
                'prefix': prefix, 'gloss': gloss.item(),
                'taccs': avg_taccs, 'vaccs': avg_vaccs,
                'bepoch': bepoch, 'bacc': bacc, 'bN': bN, 'bvacc': bvacc,
                'time': time.time() - t0
            }
            progress.append(metrics)
            
            tacc_dict, vacc_dict = {f"acc_{i}": acc for i, acc in enumerate(avg_taccs)}, {f"acc_{i}": acc for i, acc in enumerate(avg_vaccs)}
            writer.add_scalars(f"tAcc", tacc_dict, epoch)
            writer.add_scalars(f"vAcc", vacc_dict, epoch)

            message = (
                f"--{prefix}, Epoch:{epoch+1}/{config['max_epochs']}, gloss:{gloss:.4f}, "
                f"taccs:{max(avg_taccs):.4f}, vaccs:{max(avg_vaccs):.4f}, "
                f"bacc:{bacc:.4f}({bvacc}-{bN}/{n_classifiers}), bepoch:{bepoch}, patience:{patience}, "
                f"lr:{optimizer.param_groups[0]['lr']:.1e}, time:{time.time()-t0:.0f}s"
            )
            logging.info(f"cweights:{cweights}, alpha:{alpha}, temperature:{temperature}")
            print(message)
            logging.info(message)
            scheduler.step(min(avg_vlosses))

            if patience > 10:
                model.load_state_dict(bmodel_state)
                optimizer.param_groups[0]['lr'] = config['lr']
                patience = 0

        progress_df = pd.DataFrame(progress)
        model.load_state_dict(bmodel_state)
        pruned_model = model.retain_layers(bN)

        torch.save(pruned_model, f"{prefix}={bacc}({bvacc}-{bN}|{n_classifiers}).pth")
        progress_df.to_csv(f"{prefix}={bacc}({bvacc}-{bN}|{n_classifiers}).csv", index=False)
        print(f"Training finished in {time.time()-t0:.0f}s")

    except Exception as e:
        print(e)
        traceback.print_exc()


if __name__ == '__main__':
    config = {
        # data
        'dataset_name': 'cifar10',
        'num_workers': 0,
        'fold': 1,            
        # model
        'model_name': 'vit_b_16',
        'pretrain': True, 
        # train
        'batch_size': 128,
        'lr': 1e-4,
        'if_checkpoint': False,
        'amp': True,
        'max_epochs': 120
    }
    config['seed'] = 1
    # config['test_number'] = 30
    config['distill'] = False
    config['prefix'] = 'as=cifar10=vit=prune=single'
    train(rank=0, config=config)

    # tensorboard --logdir=runs/