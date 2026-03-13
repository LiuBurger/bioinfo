from tqdm import tqdm
from dataclasses import dataclass
import torch as pt
import numpy as np
from torch.utils.data import DataLoader
from utils.data import ProteinDataset, QueryHomologyDataset, query_homo_collate_fun, collate_fun_emb
from utils.model import DualEncoderRetriever
from utils.tools import gen_embeddings, build_idx, calculate_remote_homology_score, save_model
from sklearn.model_selection import train_test_split


data = pt.load(f'./data/pbond0_hbond0.pt', weights_only=False)
datalib = ProteinDataset(data)
lib_map = np.arange(len(datalib), dtype=np.int64)
print('number of proteins:', len(lib_map))
pdb2idx = [(data[2][i], i) for i in range(len(data[2]))] # pdb name -> idx
pdb2idx = dict(pdb2idx)
query_homo_data = QueryHomologyDataset(datalib, './data/tmalign.out', pdb2idx)
query_homo_map = np.arange(len(query_homo_data), dtype=np.int64)
batch_size = 256
libloader = DataLoader(datalib, batch_size=batch_size, shuffle=False, 
                       collate_fn=collate_fun_emb(mode='lib'), num_workers=6)
_, test_map = train_test_split(lib_map, test_size=1024, random_state=42)
test_set = ProteinDataset(datalib, test_map)
train_loader = DataLoader(query_homo_data, batch_size=batch_size, shuffle=True, num_workers=6,
                          collate_fn=query_homo_collate_fun(datalib), drop_last=True, )
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=6,
                         collate_fn=collate_fun_emb('query'), drop_last=False)

@dataclass
class TrainConfig:
    epochs: int = 10
    gpu: int = 6
    model_name: str = "gcn"
    lr: float = 1e-3
    pdb_root: str = "../../data/pdb"
    tmalign_path: str = "./TMalign"
    topk: int = 12
    tmalign_reference: int = 1
    tmalign_workers: int = None  # None 表示自动选择


class RegressionTrainer:
    def __init__(self, model, config: TrainConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.log_file = open(f"{config.model_name}.txt", "w", encoding="utf-8")
        self.max_score = float("-inf")
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = pt.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = pt.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.1
        )

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        lib_loader: DataLoader,
        test_set,
        datalib,
    ):
        try:
            for epoch in range(self.config.epochs):
                print(f"======================= Train epoch {epoch + 1} =======================")
                self.log_file.write(f"\nEpoch {epoch + 1} Training\n")
                train_loss = []
                self.model.train()
                for i, batch in enumerate(tqdm(train_loader, unit="batch")):
                    prot, inv, score = batch
                    prot = [x.to(self.config.gpu) for x in prot]
                    inv = [x.to(self.config.gpu) for x in inv]
                    score = score.to(self.config.gpu)
                    output = self.model((prot, inv))
                    tm_score, seq_score = output
                    tm_loss = self.criterion(tm_score, score[:, 0])
                    seq_loss = self.criterion(seq_score, score[:, 1])
                    loss = tm_loss + seq_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_loss.append(loss.item())
                    if (i + 1) % 500 == 0:
                        avg_loss = float(pt.tensor(train_loss).mean())
                        msg = f"Epoch [{epoch + 1}/{self.config.epochs}], Train Loss: {avg_loss:.4f}"
                        print(msg)
                        self.log_file.write(msg + "\n")
                        train_loss = []

                self.scheduler.step()
                print(f"======================= Test epoch {epoch + 1} =======================")
                self.log_file.write(f"\nEpoch {epoch + 1} Testing\n")
                pt.cuda.empty_cache()
                self.model.eval()
                with pt.no_grad():
                    embs_lib = gen_embeddings(self.model, lib_loader, self.config.gpu)
                    embs_test = gen_embeddings(self.model, test_loader, self.config.gpu)
                I, _ = build_idx(embs_lib, embs_test, self.config.gpu)
                score = calculate_remote_homology_score(
                    query=test_set,
                    database=datalib,
                    idx=I,
                    k=self.config.topk,
                    pdb_root=self.config.pdb_root,
                    tmalign_path=self.config.tmalign_path,
                    reference=self.config.tmalign_reference,
                    num_workers=self.config.tmalign_workers,
                )
                msg = f"Remote homologous score: {score:.6f}"
                print(msg)
                self.log_file.write(msg + "\n")
                if score > self.max_score:
                    self.max_score = score
                    save_model(self.model, self.config.model_name, epoch)
                print("======================================================================")
        finally:
            self.close()


if __name__ == "__main__":
    config = TrainConfig()
    pt.cuda.set_device(config.gpu)
    model = ProteinGCN().cuda(config.gpu)
    trainer = RegressionTrainer(model, config)
    trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        lib_loader=libloader,
        test_set=test_set,
        datalib=datalib,
    )
    pt.cuda.empty_cache()