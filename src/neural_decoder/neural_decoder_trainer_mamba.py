import os
import pickle
import time
import wandb

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder, MambaDecoder
from .dataset import SpeechDataset


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"
    
    if args["USE_WANDB"]:
        # Make wandb config dictionary
        wandb.init(project=args["wandb_project"], job_type='model_training', config=args, entity=args["wandb_entity"])
    else:
        wandb.init(mode='offline')

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )
    print("Training Mamba...")

    model = MambaDecoder(
        neural_dim=args["nInputFeatures"],
        d_model=args["d_model"],
        d_state=args["d_state"], 
        d_conv=args["d_conv"],
        expand_factor=args["expand_factor"],
        n_classes=args["nClasses"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, args['adamBeta2']),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    # scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=1.0,
    #     end_factor=args["lrEnd"] / args["lrStart"],
    #     total_iters=args["nBatch"],
    # )

    # warmup learning rate for nWarmup iters (min = 1)
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / args["nWarmup"],
        end_factor=1.0,
        total_iters=args["nWarmup"],
    )
    # if args['cosine_anneal']:
    #     scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer,
    #         T_max=args['nBatch'] - args['nWarmup'],
    #         eta_min=args['lrMin'],
    #     )
    # else:
    #     scheduler2 = torch.optim.lr_scheduler.LinearLR(
    #         optimizer,
    #         start_factor=1.0,
    #         end_factor=args["lrEnd"] / args["lrStart"],
    #         total_iters=args["nBatch"] - args['nWarmup'],
    #    )

    scheduler2 = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"] - args['nWarmup'],
   )
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[scheduler1, scheduler2], 
        milestones=[args["nWarmup"]]
    )


    # --train--
    testLoss = []
    testCER = []
    early_end = 0
    best_train_loss = 0.0
    best_train_CER = 1.0
    best_batch = 0
    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        # Compute prediction error
        pred = model.forward(X, dayIdx)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                # get train batch CER
                adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                    torch.int32
                )
                total_edit_distance = 0
                total_seq_length = 0
                for iterIdx in range(pred.shape[0]):
                    decodedSeq = torch.argmax(
                        torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                        dim=-1,
                    )  # [num_seq,]
                    decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                    decodedSeq = decodedSeq.cpu().detach().numpy()
                    decodedSeq = np.array([i for i in decodedSeq if i != 0])

                    trueSeq = np.array(
                        y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                    )

                    matcher = SequenceMatcher(
                        a=trueSeq.tolist(), b=decodedSeq.tolist()
                    )
                    total_edit_distance += matcher.distance()
                    total_seq_length += len(trueSeq)
                train_cer = total_edit_distance / total_seq_length
                
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                
                print(
                    f"batch {batch/100}, ctc loss: {avgDayLoss:>7f}, train_cer: {train_cer:>7f}, test_cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                train_loss = avgDayLoss
                train_CER = train_cer
                raw_CER = cer
                
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_batch = batch / 100
                
                if train_CER < best_train_CER:
                    best_train_CER = train_CER
                    best_batch = batch / 100                
                
                startTime = time.time()
                wandb.log(
                    {
                        "Training Loss": train_loss,
                        "Training CER": train_CER,
                        "Test CER": raw_CER,
                        "Batch": batch / 100,
                        "time/batch": (endTime - startTime)/100,
                        # "Learning rate count": lr_count,
                        # "Opt acc": opt_acc,
                        # "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                        # "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
                    }
                )
                wandb.run.summary["Training Loss"] = train_loss
                wandb.run.summary["Training CER"] = train_CER
                wandb.run.summary["Best Batch"] = best_batch
                wandb.run.summary["Best Training Loss"] = best_train_loss
                wandb.run.summary["Best Training CER"] = best_train_CER    

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
                early_end = 0
            else:
                early_end = early_end + 1
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)

            # Make this a hyperparameter in the future
            if early_end > 15:
                break

def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = MambaDecoder(
        neural_dim=args["nInputFeatures"],
        d_model=args["d_model"],
        d_state=args["d_state"],
        d_conv=args["d_conv"],
        expand_factor=args["expand_factor"],
        n_classes=args["nClasses"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()
