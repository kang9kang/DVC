import time
import os
import torch
from torch.utils.data import DataLoader
import logging


class Trainer:
    def __init__(
        self, dataset, model, loss, args, device=torch.device("cuda:0"), devices=[0, 1]
    ):
        self.dataset = dataset
        self.train_dataloader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )
        self.model = model.to(device)
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model, device_ids=devices)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.base_lr)
        self.loss = loss.to(device)
        self.device = device
        self.devices = devices
        self.args = args
        self.current_lr = args.base_lr

        self.train_logger = logging.getLogger("Train")
        self.test_logger = logging.getLogger("Test")

        logging.basicConfig(
            filename="DVC.log",
            filemode="a",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    def train(self, epochs):
        begin_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        modelspath = "./checkpoints/{}".format(begin_time)
        self.train_logger.info("-------------------Start training-------------------")
        self.train_logger.info("Start training in time: {}".format(begin_time))
        args_message = ' | '.join([f'{k}: {v}' for k, v in vars(self.args).items()])
        self.train_logger.info(args_message)
        self.train_logger.info("Total epochs: {}".format(epochs))
        batches = len(self.train_dataloader)
        self.train_logger.info("Total batches: {}".format(batches))

        os.makedirs(modelspath, exist_ok=True)

        self.model.train()
        for epoch in range(epochs):
            self.adjust_learning_rate(epoch, self.args)
            self.train_logger.info(
                "Epoch: {}, Cur_lr: {}".format(epoch, self.current_lr)
            )
            losses = []
            psnrs = []
            bpps = []
            for i, (input_img, ref_img) in enumerate(self.train_dataloader):
                input_img = input_img.to(self.device)
                ref_img = ref_img.to(self.device)

                self.optimizer.zero_grad()
                _, recon_image, warpframe, prediction, bpp = self.model(
                    input_img, ref_img
                )
                loss, psnr = self.loss(
                    input_img, recon_image, warpframe, prediction, bpp.mean()
                )
                
                losses.append(loss.item())
                psnrs.append(psnr.item())
                bpps.append(bpp.mean().item())
                loss.backward()
                self.clip_grad(clip_norm=0.5)
                self.optimizer.step()

                if i % 100 == 0:
                    self.train_logger.info(
                        "Epoch: {}/{}, Batch: {}/{}, Loss: {}, PSNR: {}, bpp: {}".format(
                            epoch,
                            epochs,
                            i,
                            batches,
                            loss,
                            psnr,
                            bpp.mean(),
                        )
                    )

                
            loss = sum(losses) / len(losses)
            psnr = sum(psnrs) / len(psnrs)
            bpp = sum(bpps) / len(bpps)
            self.train_logger.info(
                "Epoch: {}/{}, Loss: {}, PSNR: {}, bpp: {}".format(
                    epoch, epochs, loss, psnr, bpp
                )
            )
            modelpath = os.path.join(modelspath, "epoch_{}.pth".format(epoch))
            self.save(modelpath)
            self.train_logger.info("Model saved in {}".format(modelpath))

            if psnr > 37:
                self.train_logger.info("PSNR > 37, training finished!")
                return

        self.train_logger.info("All epochs finished!")

    def test(self):
        self.test_logger.info("Start testing!")
        self.test_logger.info("Finish testing!")
        pass

    def adjust_learning_rate(self, epoch, args):
        if epoch > args.lr_decay_epoch:
            self.current_lr = self.current_lr * args.lr_decay
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.current_lr

    def clip_grad(self, clip_norm=0.5):
        for p in self.model.parameters():
            p.grad.data.clamp_(-clip_norm, clip_norm)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)
