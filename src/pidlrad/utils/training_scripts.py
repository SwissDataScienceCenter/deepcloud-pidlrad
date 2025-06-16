import os
import wandb
import pickle
import torch
import torch.optim as optim
from dotenv import load_dotenv


from .custom_losses import get_loss
from .heating_rate import extrapolate_pressures, calculate_heating_rates

seed = 42
wandb_config = {"seed": seed}
torch.manual_seed(seed)


# train function
def train_model(
    model,
    train_dataloader,
    val_dataloader,
    logger,
    last_checkpoint,
    best_checkpoint,
    save_id,
    device,
    args,
):
    load_dotenv()
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)

        logger.info("Start the training loop...")
        wandb.init(
            project="ig-transformers",
            entity="deepcloud",
            name=save_id,
            id=save_id,
            config={**wandb_config, **args.__dict__},
            sync_tensorboard=True,
            save_code=True,
            resume="allow",
            tags=[
                "icon grid",
            ],
            mode=args.wandb_mode,
        )
        wandb.watch(model, log_freq=100)

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.learning_rate, eps=1e-8, weight_decay=0.01
        )
    else:
        raise NameError("optimizer not supported.")

    train_loss = get_loss(args.loss).to(device)
    valid_loss = get_loss(args.loss).to(device)
    train_mae = get_loss("mae").to(device)
    valid_mae = get_loss("mae").to(device)

    if os.path.exists(last_checkpoint):
        checkpoint = torch.load(last_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        init_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
    else:
        init_epoch = 0
        best_loss = 1e9999999
    logger.info(f"Training will start from epoch: {init_epoch}")
    for epoch in range(init_epoch, args.num_epochs):
        # Training step
        model.train(True)
        for i, data in enumerate(train_dataloader):
            batch_x3d, batch_x2d, batch_y = (d.to(device) for d in data)
            outputs = model(batch_x3d, batch_x2d)
            if args.loss in ["eclv2", "eclv4"]:
                pres = extrapolate_pressures(batch_x3d, batch_x2d)
                loss = train_loss(outputs, batch_y, pres)
            else:
                loss = train_loss(outputs, batch_y)
            batch_mae = train_mae(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()
            if i % args.log_interval == args.log_interval - 1:
                logger.info(f"batch {i+1} loss: {loss:.4f}, MAE: {batch_mae:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            for i, v_data in enumerate(val_dataloader):
                vx3d, vx2d, v_labels = (d.to(device) for d in v_data)
                v_outputs = model(vx3d, vx2d)

                if args.loss in ["eclv2", "eclv4"]:
                    pres = extrapolate_pressures(vx3d, vx2d)
                    valid_loss.update(v_outputs, v_labels, pres)
                else:
                    valid_loss.update(v_outputs, v_labels)
                valid_mae.update(v_outputs, v_labels)

        total_train_loss = train_loss.compute()
        total_valid_loss = valid_loss.compute()
        total_train_mae = train_mae.compute()
        total_valid_mae = valid_mae.compute()

        print(
            f"{epoch}/{args.num_epochs}: ",
            f"Train loss: {total_train_loss:.4f} ",
            f"mean_absolute_error: {total_train_mae:.4f}, ",
            f"Validation loss: {total_valid_loss:.4f}, ",
            f"mean_absolute_error: {total_valid_mae:.4f}",
        )

        if wandb_key:
            wandb.log(
                {
                    "epoch": epoch,
                    "loss": total_train_loss,
                    "val_loss": total_valid_loss,
                    "mean_absolute_error": total_train_mae,
                    "val_mean_absolute_error": total_valid_mae,
                }
            )

        # saving checkpoints
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": valid_loss,
            "best_loss": best_loss,
        }
        torch.save(checkpoint, last_checkpoint)
        if total_valid_loss < best_loss:
            torch.save(checkpoint, best_checkpoint)
            best_loss = total_valid_loss

        train_mae.reset()
        valid_mae.reset()
        train_loss.reset()
        valid_loss.reset()
    return model


def test_model(model, test_set, test_dir, logger, device, args):
    logger.info(f"Start the test..., result will be saved at: {test_dir}")
    # if not os.path.exists(os.path.join(test_dir, 'mean_abs_err.pickle')):
    test_loss = get_loss(args.loss).to(device)
    test_mae = get_loss("mae").to(device)
    test_column_mae = get_loss("column_mae").to(device)

    with torch.no_grad():
        for i, data in enumerate(test_set):
            batch_x3, batch_x2, batch_y = [d.to(device) for d in data]

            # batch_y = batch_y * args.y_std + args.y_mean

            outputs = model(batch_x3, batch_x2)

            test_column_mae.update(batch_y, outputs, batch_x3, batch_x2)

            if args.loss in ["eclv2", "eclv4"]:
                pres = extrapolate_pressures(batch_x3, batch_x2)
                loss = test_loss(outputs, batch_y, pres)
            else:
                loss = test_loss(outputs, batch_y)
            mae = test_mae(outputs, batch_y)
            if i % args.log_interval == args.log_interval - 1:
                logger.info(f"batch {i+1} loss: {loss:.4f}, MAE: {mae:.4f},")

        mean_err, heat_err = [e.cpu() for e in test_column_mae.compute()]
        with open(os.path.join(test_dir, "mean_abs_err.pickle"), "wb") as handle:
            pickle.dump(mean_err, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(test_dir, "heating_rate_mae.pickle"), "wb") as handle:
            pickle.dump(heat_err, handle, protocol=pickle.HIGHEST_PROTOCOL)

        total_test_loss = test_loss.compute()
        total_test_mae = test_mae.compute()

        logger.info(f"Test loss: {total_test_loss:.4f}, MAE: {total_test_mae:.4f}")
