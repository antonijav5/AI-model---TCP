import torch
from data.events_loader import EventsLoader

torch.set_float32_matmul_precision('high')
device = torch.device("cuda")


import time
from datetime import datetime
import os

from modeling.models.model import ModelTrainer

from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

base_path = os.path.dirname(os.path.abspath(__file__))


def load_config(config_path=f'{base_path}/training_config.json'):
    with open(config_path, 'r') as f:
        import json
        config = json.load(f)
    return config

def simple_collate(x):
    return x[0]

def main():
    config = load_config()
    print(config)
    full_dataset = EventsLoader(
        batch_size=config['batch_size'],
        src_seq_length=config['src_seq_len'],
        tgt_seq_length=config['tgt_seq_len'],
        id_category_size=config['id_category_size'],
        input_size=config['model']['d_model'],
        output_size=config['id_category_size'],
    )
    print(full_dataset.category_fields.values())
    output_names = full_dataset.category_fields.keys()
    model_g = ModelTrainer(
        d_input=full_dataset.input_size,
        d_categories=list(full_dataset.category_fields.values()),
        d_output=full_dataset.output_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['num_heads'],
        encoder_layers=config['model']['encoder_layers'],
        decoder_layers=config['model']['decoder_layers'],
        encoders=full_dataset.encoder_streams
    )

    start_epoch = 0
    if config['model_checkpoint'] is not None:
        model_loaded = torch.load(
            config['model_checkpoint'],
            weights_only=False,
            map_location=device
        )
        model_g.load_state_dict(
            model_loaded['model_state_dict'],
        )
        start_epoch = model_loaded['epoch']

    model_g = model_g.to('cuda', dtype=torch.bfloat16)
   # device = torch.device('cpu')
#model_g = model_g.to(device)

    optimizer_g = torch.optim.Adam(
        model_g.parameters(),
        lr=3e-4
    )

    model_g.train()
    # full_dataset.load_dataset()
    training_data, test_data = torch.utils.data.random_split(full_dataset, [0.9, 0.1])

    train_dataloader = DataLoader(
        training_data,  # Changed from full_dataset to training_data
        num_workers=24,
        batch_size=1,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=lambda x: x[0]
    )
    test_dataloader = DataLoader(
        test_data,
        num_workers=8,
        batch_size=1,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=lambda x: x[0]
    )

    current_datetime = datetime.now()
    run_directory = f'/runs/{current_datetime.strftime("%Y-%m-%d")}/{current_datetime.strftime("%H_%M")}'
    test_writer = SummaryWriter(log_dir=f'{run_directory}/test')
    train_writer = SummaryWriter(log_dir=f'{run_directory}/train')
    train(
        model_g,
        optimizer_g,
        train_writer,
        test_writer,
        train_dataloader,
        test_dataloader,
        start_epoch,
        config['epochs'],
        config['grad_accum'],
        run_directory,
        output_names,
        config['batch_size']
    )


def execute(
        accum_iter,
        epoch,
        model_g: "ModelTrainer",
        dataloader,
        writer,
        output_names,
        batch_size,
        optimizer_g=None,
        train=False,
):
    last_time = time.time()
    total_steps = len(dataloader)

    mode = "Train" if train else "Test"
    print(f"\n[{mode}] Epoch {epoch} - Total steps: {total_steps}")

    total_loss = 0.0
    category_totals = [0.0] * len(output_names)

    for data_step, (tgt, srcs, masks) in enumerate(dataloader):

        full_progress = count_steps(epoch, data_step, total_steps, batch_size)
        writer.add_scalar(f"Time/Data", (time.time() - last_time), full_progress)
        last_time = time.time()

        ''' Train Forcaster '''
        if optimizer_g is not None:
            (category_loss, category_loss_separated) = model_g(srcs, tgt, masks, train)
            category_loss = sum(category_loss_separated)

            # FIX: Konvertuj u tensor SAMO ako nije tensor
            if not isinstance(category_loss, torch.Tensor):
                category_loss = torch.tensor(category_loss, device=device, requires_grad=True)
        else:
            with torch.no_grad():
                (category_loss, category_loss_separated) = model_g(srcs, tgt, masks)
            category_loss = sum(category_loss_separated)

        total_loss += category_loss

        # FIX: DODAJTE OVO - ažuriranje category_totals
        for i in range(len(category_totals)):
            category_totals[i] += category_loss_separated[i]

        if data_step % 10 == 0 or data_step == total_steps - 1:
            progress_pct = (data_step + 1) / total_steps * 100
            print(f"  Step {data_step + 1}/{total_steps} ({progress_pct:.1f}%) - Loss: {category_loss:.6f}")

            if data_step % 50 == 0 and data_step > 0:
                print(f"    Category losses: {[f'{loss:.4f}' for loss in category_loss_separated]}")

        writer.add_scalar(f"Model/category_loss", category_loss, full_progress)
        for i, name in enumerate(output_names):
            writer.add_scalar(f"Model/{name}", category_loss_separated[i], full_progress)

        if optimizer_g is not None:
            if ((data_step + 1) % accum_iter == 0) or (data_step + 1 == total_steps):
                # FIX: Dodajte backward pass!
                category_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_g.parameters(), max_norm=1.0)
                optimizer_g.step()
                optimizer_g.zero_grad()

        writer.add_scalar(f"Time/Compute", (time.time() - last_time), full_progress)
        last_time = time.time()

    # FIX: Konvertuj u float za računanje proseka
    if isinstance(total_loss, torch.Tensor):
        total_loss = total_loss.item()

    avg_loss = total_loss / total_steps
    avg_categories = [total / total_steps for total in category_totals]

    print(f"[{mode}] Epoch {epoch} Summary - Avg Loss: {avg_loss:.6f}")
    print(f"  Category averages: {[f'{name}: {avg:.4f}' for name, avg in zip(output_names, avg_categories)]}")

    return avg_loss


def train(
        model_g,
        optimizer_g,
        train_writer,
        test_writer,
        train_dataloader,
        test_dataloader,
        start_epoch,
        epochs,
        grad_accum,
        run_directory,
        output_names,
        batch_size
):

    #LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g,
        mode='min',
        factor=0.5,
        patience=3,
    )

    print("\n" + "=" * 80)
    print(f"Starting training for {epochs} epochs")
    print(f"Batch size: {batch_size}, Gradient accumulation: {grad_accum}")
    print("=" * 80)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, epochs):
        print(f"\n{'=' * 60}")
        print(f'EPOCH {epoch}/{epochs - 1}')
        print('=' * 60)

        epoch_start = time.time()

        val_loss = execute(
            grad_accum,
            epoch,
            model_g,
            test_dataloader,
            test_writer,
            output_names,
            batch_size
        )

        optimizer_g.zero_grad()
        train_loss = execute(
            grad_accum,
            epoch,
            model_g,
            train_dataloader,
            train_writer,
            output_names,
            batch_size,
            optimizer_g, True
        )

        # UPDATE LEARNING RATE BASED ON VAL LOSS
        scheduler.step(val_loss)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"✓ New best validation loss: {val_loss:.6f}")
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_g.state_dict(),
                'optimizer_state_dict': optimizer_g.state_dict(),
                'best_val_loss': best_val_loss,
            }, f'{run_directory}/best_model.pt')

        test_writer.flush()
        train_writer.flush()

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s")
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        try:
            if epoch % 10 == 0:
                save_path = f'{run_directory}/forcaster_checkpoint_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_g.state_dict(),
                    'optimizer_state_dict': optimizer_g.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                print(f"Checkpoint saved: {save_path}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model_g.state_dict(),
                'optimizer_state_dict': optimizer_g.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'{run_directory}/forcaster_checkpoint_latest.pt')
        except Exception as e:
            print(e)
            pass

    # Added: Training complete message
    print("\n" + "=" * 80)
    print(f"Training completed! Total epochs: {epochs - start_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 80)


def count_steps(epoch: int, batch_num, num_batches, batch_size):
    return (epoch * num_batches + batch_num) * batch_size


if __name__ == '__main__':
    main()