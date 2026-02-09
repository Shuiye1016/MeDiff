# config.py
import torch
import os

SEED = 42
RESUME_TRAINING = False
DATA_DIR = "../dataset/Ours"
CASCADE_DIR = os.path.join(DATA_DIR, "topic_cascades")
USER_PROFILE_FILE = os.path.join(DATA_DIR, "merged_topic_user_infos.csv")
OUTPUT_DIR = "output"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "checkpoints.pth")

PRETRAINED_MODEL_NAME = "../bge_local_small"

TEXT_EMBEDDING_DIM = 64
PROFILE_EMBEDDING_DIM = 64
TIME_EMBEDDING_DIM = 64
EMBEDDING_COMBINATION = 'hetero' # {'concat', 'weighted_sum', 'hetero'}

HETERO_GNN_HIDDEN_DIM = 64
HETERO_GNN_LAYERS = 2      
HETERO_DROPOUT_RATE = 0.3  
USER_NODE_INIT_DIM = 64

PROFILE_CAT_AGG_DIM = 64
PROFILE_NUM_AGG_DIM = 64
PROFILE_TEXTUAL_AGG_DIM = 64
PROFILE_DATETIME_AGG_DIM = 64

if EMBEDDING_COMBINATION == 'concat':
    INITIAL_EMBEDDING_DIM = TEXT_EMBEDDING_DIM + PROFILE_EMBEDDING_DIM + TIME_EMBEDDING_DIM
elif EMBEDDING_COMBINATION == 'weighted_sum':
    INITIAL_EMBEDDING_DIM = max(TEXT_EMBEDDING_DIM, PROFILE_EMBEDDING_DIM, TIME_EMBEDDING_DIM, 64) 
elif EMBEDDING_COMBINATION == 'hetero':
    INITIAL_EMBEDDING_DIM = HETERO_GNN_HIDDEN_DIM 
else:
    raise ValueError(f"Unknown EMBEDDING_COMBINATION: {EMBEDDING_COMBINATION}")

GNN_EMBEDDING_DIM = 64
GNN_LAYERS = 2
NUM_CLUSTERS = 5       
FCM_M = 1.5            

ODE_TIMESTEPS = 5

GROUP_ATTENTION_DIM = GNN_EMBEDDING_DIM 
ATTENTION_HEADS = 4                   
CASCADE_REP_AGGREGATION = "group_attention"
CASCADE_MODEL_DIM = GNN_EMBEDDING_DIM 

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 200
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.4 
ODE_DROPOUT_RATE = 0.1
PREDICTION_HORIZON_RATIO = 0.2 
GROUP_LOSS_WEIGHT = 1.5
SIZE_LOSS_WEIGHT = 0.7
BCE_LOSS_WEIGHT = 1.0
BPR_LOSS_WEIGHT = 0.3
GRADIENT_CLIP_VAL = 1.0
EARLY_STOPPING_PATIENCE = 20
RETRIEVAL_TEMPERATURE = 0.1

MAX_CASCADE_LENGTH = 20000000
MIN_CASCADE_LENGTH = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SELECTED_PROFILE_FEATURES = {
    'categorical': [
        '性别', '认证状态', '认证类型', '信用等级',
    ],
    'numerical': [
        '粉丝数', '关注数', '微博数',       
        '获评转赞总数', '获评论数', '获转发数', '获点赞数',
        '微博会员等级',
    ],
    'textual': [
        '用户名', '所在地', '认证理由', '个人描述', '公司', '学校',
    ],
    'datetime': [ 
        '生日', '创建时间',
    ],
}
PROFILE_PADDING_IDX = 0 

TOP_K_RECALL = [5, 10, 20, 50, 100, 200, 500]
TOP_K_NDCG = [5, 10, 20, 50, 100, 200, 500]
TOP_K = [5, 10, 20, 50, 100, 200, 500]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"--- Configuration Loaded (config.py) ---")
print(f"DEVICE: {DEVICE}")
print(f"EMBEDDING_COMBINATION: {EMBEDDING_COMBINATION}")
print(f"INITIAL_EMBEDDING_DIM (Z - Target for CombinedEmbedder output): {INITIAL_EMBEDDING_DIM}")
if EMBEDDING_COMBINATION == 'hetero':
    print(f"  HETERO_GNN_HIDDEN_DIM (also Z for hetero): {HETERO_GNN_HIDDEN_DIM}")
    print(f"  USER_NODE_INIT_DIM (for hetero user type): {USER_NODE_INIT_DIM}")
print(f"TextEmbedder Output Dim: {TEXT_EMBEDDING_DIM}")
print(f"ProfileEmbedder Final Output Dim ('final_profile_embedding'): {PROFILE_EMBEDDING_DIM}")
print(f"TimeEmbedder Output Dim: {TIME_EMBEDDING_DIM}")
print(f"Cascade Representation Aggregation: {CASCADE_REP_AGGREGATION}")
print(f"Fuzzy-C-Means m: {FCM_M}")
print(f"Fuzzy-C-Means cluster: {NUM_CLUSTERS}")
print(f"RETRIEVAL_TEMPERATURE: {RETRIEVAL_TEMPERATURE}")
print("--------------------------------------------")
