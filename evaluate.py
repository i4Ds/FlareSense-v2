    # Evaluate on Test set
    test_dataloader = DataLoader(
        ds_test,
        batch_size=config["general"]["batch_size"],
        num_workers=8,
        shuffle=False,
        persistent_workers=True,
    )

    trainer.test(model, dataloaders=test_dataloader)
