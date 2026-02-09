# train.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import config
from data_loader import load_and_preprocess_data, CascadeDataset, collate_fn
from model import GRIDModel 
from embeddings import CombinedEmbedder, TextEmbedder, ProfileEmbedder, TimeEmbedder
from utils import calculate_metrics, compute_early_stopping_score
import time
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train GRIDModel')
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs to use (e.g., "0" or "0,1")')
    return parser.parse_args()

def set_seed(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def msle_loss_fn(y_pred, y_true):
    eps = 1e-6
    y_pred = y_pred.float()
    y_true = y_true.float()
    y_pred = torch.clamp(y_pred, min=0.0)
    y_true = torch.clamp(y_true, min=0.0)
    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any() or torch.isnan(y_true).any() or torch.isinf(y_true).any():
        return torch.tensor(0.0, device=y_pred.device)
    return torch.mean((torch.log1p(y_pred + eps) - torch.log1p(y_true + eps))**2)

def train_epoch(model, combined_embedder_instance, dataloader, optimizer, criterion_user, criterion_size_msle, criterion_group, device, epoch):
    model.train()
    total_loss, total_bce_loss, total_size_loss, total_group_loss = 0.0, 0.0, 0.0, 0.0
    start_time = time.time()
    processed_batches, processed_samples = 0, 0

    for i, batch in enumerate(dataloader): 
        if batch is None: continue
        optimizer.zero_grad()

        if 'target_user_vector' not in batch or 'target_size' not in batch: continue
        target_user_vectors_full = batch['target_user_vector'].to(device)
        target_sizes_full = batch['target_size'].to(device)
        
        outputs = model(batch, combined_embedder_instance=combined_embedder_instance) 

        if outputs is None or 'user_scores' not in outputs or outputs['user_scores'].shape[0] == 0: continue
        
        valid_indices = outputs.get('valid_indices')
        if valid_indices is None or not valid_indices : continue
        
        try: valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long, device=device)
        except Exception: continue 
        if valid_indices_tensor.numel() == 0: continue

        target_user_vectors = target_user_vectors_full.index_select(0, valid_indices_tensor)
        target_sizes = target_sizes_full.index_select(0, valid_indices_tensor) 
        
        actual_batch_size = outputs['user_scores'].shape[0]
        if actual_batch_size == 0: continue

        pred_user_scores = outputs['user_scores']
        bce_loss = criterion_user(pred_user_scores, target_user_vectors)

        pred_sizes_raw = outputs['size_predictions']
        pred_sizes_non_negative = F.relu(pred_sizes_raw) 
        size_loss = criterion_size_msle(pred_sizes_non_negative, target_sizes)

        pred_group_logits = outputs['group_logits']
        target_group_dist = outputs['target_group_dist'] 
        
        group_loss = torch.tensor(0.0, device=device) 
        if pred_group_logits.numel() > 0 and target_group_dist.numel() > 0 and \
           pred_group_logits.shape[0] == target_group_dist.shape[0] and \
           pred_group_logits.shape[1] == target_group_dist.shape[1] and \
           pred_group_logits.shape[1] > 0: 
            log_pred_group_probs = F.log_softmax(pred_group_logits, dim=1)
            group_loss = criterion_group(log_pred_group_probs, target_group_dist)
        elif pred_group_logits.numel() > 0 and target_group_dist.numel() > 0 : 
            print(f"Train Warning: Shape mismatch for group loss. Logits: {pred_group_logits.shape}, TargetDist: {target_group_dist.shape}")

        current_loss_value = (
                config.BCE_LOSS_WEIGHT * bce_loss +
                config.SIZE_LOSS_WEIGHT * size_loss +
                config.GROUP_LOSS_WEIGHT * group_loss
        )

        if torch.isnan(current_loss_value): continue

        current_loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)
        optimizer.step()

        total_loss += current_loss_value.item()
        total_bce_loss += bce_loss.item()
        total_size_loss += size_loss.item() 
        total_group_loss += group_loss.item()
        processed_batches += 1
        processed_samples += actual_batch_size

        if processed_batches > 0 and processed_batches % 20 == 0 :
             elapsed = time.time() - start_time
             print(f'Epoch [{epoch+1}/{config.EPOCHS}], Batch [{processed_batches}/{len(dataloader)}], '
                   f'Avg Loss: {total_loss/processed_batches:.4f}, UserL(BCE): {total_bce_loss/processed_batches:.4f}, '
                   f'SizeL(MSLE): {total_size_loss/processed_batches:.4f}, GroupL: {total_group_loss/processed_batches:.4f}, Time: {elapsed:.2f}s')
    
    avg_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
    avg_bce = total_bce_loss / processed_batches if processed_batches > 0 else 0.0
    avg_sl = total_size_loss / processed_batches if processed_batches > 0 else 0.0 
    avg_gl = total_group_loss / processed_batches if processed_batches > 0 else 0.0

    print(f'--- Epoch {epoch+1} Train Summary ---')
    print(f'Avg Loss: {avg_loss:.4f}, Avg UserL(BCE): {avg_bce:.4f}, Avg SizeL(MSLE): {avg_sl:.4f}, Avg GroupL: {avg_gl:.4f}')
    print(f'Processed Samples: {processed_samples}')
    print(f'--------------------------------')
    return avg_loss

def evaluate_epoch(model, combined_embedder_instance, dataloader, device, epoch):
    model.eval()
    topk = config.TOP_K
    metric_sums = {metric: {k: 0.0 for k in topk} for metric in ['Recall', 'NDCG']}
    total_size_sq_log_error = 0.0 
    total_items_for_metrics = 0
    total_group_samples_for_acc = 0
    total_correct_top1 = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader): 
            if batch is None: continue
            if 'target_user_vector' not in batch or 'target_size' not in batch: continue
            target_user_vectors_full = batch['target_user_vector'].to(device)
            target_sizes_full = batch['target_size'].to(device)
            
            outputs = model(batch, combined_embedder_instance=combined_embedder_instance)
            
            if outputs is None or 'user_scores' not in outputs or outputs['user_scores'].shape[0] == 0: continue
            valid_indices = outputs.get('valid_indices')
            if valid_indices is None or not valid_indices: continue
            try: valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long, device=device)
            except Exception: continue
            if valid_indices_tensor.numel() == 0: continue

            target_user_vectors = target_user_vectors_full.index_select(0, valid_indices_tensor)
            target_sizes = target_sizes_full.index_select(0, valid_indices_tensor) 

            pred_user_scores = outputs['user_scores'].sigmoid()
            pred_sizes_raw = outputs['size_predictions']
            pred_sizes_non_negative = F.relu(pred_sizes_raw) 

            pred_group_logits = outputs.get('group_logits')
            target_group_dist_from_model = outputs.get('target_group_dist')

            if pred_user_scores.shape[0] != target_user_vectors.shape[0] or \
               pred_sizes_non_negative.shape[0] != target_sizes.shape[0] or \
               target_user_vectors.shape[0] == 0: 
                continue
            current_batch_item_count = pred_user_scores.shape[0]
            if current_batch_item_count == 0: continue
            batch_metrics = calculate_metrics(pred_user_scores.cpu(), target_user_vectors.cpu(), topk)
            for metric_name in metric_sums:
                for k in topk:
                    metric_sums[metric_name][k] += batch_metrics.get(f'{metric_name}@{k}', 0.0) * current_batch_item_count
            log_true = torch.log1p(target_sizes)
            log_pred = torch.log1p(pred_sizes_non_negative) 
            sq_log_error = (log_pred - log_true)**2
            total_size_sq_log_error += torch.sum(sq_log_error).item()
            
            if pred_group_logits is not None and pred_group_logits.numel() > 0 and \
               target_group_dist_from_model is not None and target_group_dist_from_model.numel() > 0 and \
               pred_group_logits.shape[0] == target_group_dist_from_model.shape[0] and \
               pred_group_logits.shape[1] == target_group_dist_from_model.shape[1] and \
               pred_group_logits.shape[1] > 0:
                target_group_indices = torch.argmax(target_group_dist_from_model, dim=1)
                top1_pred = torch.argmax(pred_group_logits, dim=1)
                correct_top1 = (top1_pred == target_group_indices).sum().item()
                total_correct_top1 += correct_top1
                total_group_samples_for_acc += pred_group_logits.size(0)

            total_items_for_metrics += current_batch_item_count

    final_metrics = {}
    if total_items_for_metrics == 0:
        for metric in metric_sums:
            for k in topk:
                final_metrics[f'{metric}@{k}'] = 0.0

        final_metrics['MSLE_Size'] = float('inf')
        final_metrics['GroupAcc'] = 0.0
        return final_metrics

    for metric in metric_sums:
        for k in topk:
            final_metrics[f'{metric}@{k}'] = metric_sums[metric][k] / total_items_for_metrics
    
    msle = total_size_sq_log_error / total_items_for_metrics if total_items_for_metrics > 0 else float('inf')
    final_metrics['MSLE_Size'] = np.sqrt(msle) if msle != float('inf') else float('inf')
    final_metrics['GroupAcc'] = total_correct_top1 / total_group_samples_for_acc if total_group_samples_for_acc > 0 else 0.0

    return final_metrics


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print(f"Using GPUs: {args.gpus}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    set_seed(config.SEED)
    print("--- Initializing and Loading Data ---")
    data_package = load_and_preprocess_data()
    train_data = data_package["train"]; val_data = data_package["val"]; test_data = data_package["test"] 
    profiles_df = data_package["profiles_df"]; user_to_idx = data_package["user_to_idx"] 
    idx_to_user = data_package["idx_to_user"]; all_user_indices = data_package["all_user_indices"] 
    num_users = data_package["num_users"] 
    if not train_data or num_users == 0: print("Error: No training data or no users. Exiting."); exit()
    
    print("--- Initializing Embedders ---")
    text_embedder_main = TextEmbedder().to(config.DEVICE) if config.TEXT_EMBEDDING_DIM > 0 else None
    profile_embedder_main = ProfileEmbedder(
        profiles_df if profiles_df is not None else pd.DataFrame(), 
        config.SELECTED_PROFILE_FEATURES,
        output_dim=config.PROFILE_EMBEDDING_DIM 
    ).to(config.DEVICE) if config.PROFILE_EMBEDDING_DIM > 0 else None
    time_embedder_main = TimeEmbedder().to(config.DEVICE) if config.TIME_EMBEDDING_DIM > 0 else None

    combined_embedder_main = CombinedEmbedder(
        text_embedder_main, profile_embedder_main, time_embedder_main,
        num_users=num_users,
        user_to_idx=user_to_idx
    ).to(config.DEVICE)
    
    print(f"CombinedEmbedder initialized. Output Z Dim: {combined_embedder_main.final_combined_dim}")
    if combined_embedder_main.final_combined_dim != config.INITIAL_EMBEDDING_DIM:
        print(f"CRITICAL WARNING: CombinedEmbedder output Z dim ({combined_embedder_main.final_combined_dim}) "
              f"!= config.INITIAL_EMBEDDING_DIM ({config.INITIAL_EMBEDDING_DIM}). ")

    train_dataset = CascadeDataset(train_data, user_to_idx, combined_embedder_main)
    val_dataset = CascadeDataset(val_data, user_to_idx, combined_embedder_main) if val_data else None
    test_dataset = CascadeDataset(test_data, user_to_idx, combined_embedder_main) if test_data else None

    train_b_size = min(config.BATCH_SIZE, len(train_dataset)) if train_dataset and len(train_dataset) > 0 else 1
    val_b_size = min(config.BATCH_SIZE, len(val_dataset)) if val_dataset and len(val_dataset) > 0 else 1
    test_b_size = min(config.BATCH_SIZE, len(test_dataset)) if test_dataset and len(test_dataset) > 0 else 1

    train_loader = DataLoader(train_dataset, batch_size=train_b_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=val_b_size, shuffle=False, collate_fn=collate_fn, num_workers=0) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=test_b_size, shuffle=False, collate_fn=collate_fn, num_workers=0) if test_dataset else None
    
    print("--- Initializing Model ---")
    model = GRIDModel(
        num_users, profiles_df if profiles_df is not None else pd.DataFrame(), 
        user_to_idx, idx_to_user, all_user_indices
    ).to(config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion_user = nn.BCEWithLogitsLoss()
    criterion_size_msle_main = msle_loss_fn 
    criterion_group = nn.KLDivLoss(reduction='batchmean', log_target=False)

    print(f"--- Starting Training (Embedding: {config.EMBEDDING_COMBINATION}, Aggregation: {config.CASCADE_REP_AGGREGATION}) ---")

    start_epoch = 0
    best_val_metric = float('-inf') 
    epochs_no_improve = 0

    if config.RESUME_TRAINING and os.path.exists(config.MODEL_SAVE_PATH):
        print(f"==> Loading checkpoint from {config.MODEL_SAVE_PATH}")
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0)
        best_val_metric = checkpoint.get('best_val_metric', best_val_metric)
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)

        print(f"Resumed training from epoch {start_epoch}, best_val_metric={best_val_metric:.4f}")

    for epoch in range(start_epoch, config.EPOCHS):
        start_ep = time.time()
        train_loss = train_epoch(model, combined_embedder_main, train_loader, optimizer, 
                                 criterion_user, criterion_size_msle_main, criterion_group, 
                                 config.DEVICE, epoch)
        if val_loader:
            val_metrics = evaluate_epoch(model, combined_embedder_main, val_loader, config.DEVICE, epoch)
            val_score = val_metrics.get('NDCG@100', 0.0)
            scheduler.step(val_score)

            end_ep = time.time()
            print(f'--- Epoch {epoch+1}/{config.EPOCHS} Validation Results ---')
            val_metrics_str_list = []
            print(
                f"Validation: Acc={val_metrics.get('GroupAcc', 0.0):.4f}, MSLE={val_metrics.get('MSLE_Size', 0.0):.4f}")
            for k in sorted(set(config.TOP_K_RECALL + config.TOP_K_NDCG)):
                print(f"  @K={k} | "
                      f"NDCG={val_metrics.get(f'NDCG@{k}', 0.0):.4f}, "
                      f"Recall={val_metrics.get(f'Recall@{k}', 0.0):.4f}, ")

            print(f'--- Epoch {epoch + 1}/{config.EPOCHS} Validation Ends ---')
            print("-----------------------------------------------------------")
            
            key_metric_name = f'Recall@{max(config.TOP_K_RECALL)}' if config.TOP_K_RECALL else 'GroupAcc'
            is_higher_better = 'Recall' in key_metric_name or 'NDCG' in key_metric_name or 'GroupAcc' in key_metric_name
            if key_metric_name == 'MSLE_Size': best_val_metric = float('inf')
            current_val_metric = compute_early_stopping_score(val_metrics)
            is_higher_better = True
            improved = (is_higher_better and current_val_metric > best_val_metric)
            
            if improved:
                best_val_metric = current_val_metric
                epochs_no_improve = 0
                print(f"New best validation score: {best_val_metric:.4f}. Saving model...")
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_metric': best_val_metric,
                        'epochs_no_improve': epochs_no_improve
                    }, config.MODEL_SAVE_PATH)
                except Exception as e: print(f"Error saving model: {e}")
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE: print(f"Early stopping after {epochs_no_improve} epochs."); break
        else: 
            print(f"--- Epoch {epoch+1}/{config.EPOCHS} Finished (No Validation) ---"); print(f'Train Avg Loss: {train_loss:.4f}')
            try: torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            except Exception as e: print(f"Error saving model: {e}")
    print("--- Training Finished ---")
    if test_loader:
        print("Loading best model for test evaluation...");
        if os.path.exists(config.MODEL_SAVE_PATH):
            try:
                model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
                test_metrics = evaluate_epoch(model, combined_embedder_main, test_loader, config.DEVICE, epoch=-1) 
                print("--- Test Set Results ---")
                test_metrics_str_list = []
                print(
                    f"Validation: Acc={val_metrics.get('GroupAcc', 0.0):.4f}, MSLE={val_metrics.get('MSLE_Size', 0.0):.4f}")
                for k in sorted(set(config.TOP_K_RECALL + config.TOP_K_NDCG)):
                    print(f"  @K={k} | "
                          f"NDCG={test_metrics.get(f'NDCG@{k}', 0.0):.4f}, "
                          f"Recall={test_metrics.get(f'Recall@{k}', 0.0):.4f}, ")
            except Exception as e: print(f"Error during testing: {e}")
        else: print(f"Model file not found at {config.MODEL_SAVE_PATH}. Skipping test.")
    print("--- Run Finished ---")