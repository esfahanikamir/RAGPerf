import time
import numpy as np
from encoder.BaseEncoder import BaseEncoder
import torch, gc

import os
from typing import List, cast
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device

# ---- ADDED: adapter guard (transformers >=5 discards the LoRA checkpoint keys silently) ----
def _lora_stats(model):
    """(n lora_B tensors, max|lora_B|, fingerprint). Trained adapter: max > 0 and the fingerprint
    is identical on every load. Failed load: 0, or a small value that changes between loads."""
    lb = [p.detach().float() for n, p in model.named_parameters() if "lora_B" in n]
    if not lb:
        return 0, 0.0, 0.0
    return len(lb), max(p.abs().max().item() for p in lb), sum(p.abs().sum().item() for p in lb)
 
 
def _load_adapter_by_name(model, repo_or_path, adapter_name="default"):
    """Load adapter_model.safetensors by matching the layer-relative suffix of each checkpoint key
    to the model's parameter names (checkpoint ...language_model.model.layers... vs model
    ...model.language_model.layers...). Idempotent. Returns (n_loaded, unmatched_keys)."""
    from safetensors.torch import load_file
    if os.path.isdir(repo_or_path):
        path = os.path.join(repo_or_path, "adapter_model.safetensors")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_or_path, "adapter_model.safetensors")
    sd = load_file(path)
 
    def norm(k):
        k = k.replace("base_model.model.", "", 1).replace(f".{adapter_name}.", ".")
        return k[k.index("layers."):] if "layers." in k else ".".join(k.split(".")[-3:])
 
    target = {}
    for name, _ in model.named_parameters():
        if "lora_" in name:
            key = norm(name)
            if key in target:
                raise RuntimeError(f"ambiguous LoRA key {key}")
            target[key] = name
    new_sd, unmatched = {}, []
    for k, v in sd.items():
        name = target.get(norm(k))
        if name is None:
            unmatched.append(k)
        else:
            new_sd[name] = v
    res = model.load_state_dict(new_sd, strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"unexpected keys while loading adapter: {res.unexpected_keys[:3]}")
    return len(new_sd), unmatched
# ---- END ADDED ----

# TODO make this to abstactmethods
class ColPaliEncoder(BaseEncoder):
    def __init__(
        self,
        device,
        model_name,
        embedding_batch_size=64,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.embedding_batch_size = embedding_batch_size
        self.encoder = None
        return

    def __del__(self):
        self.free_encoder()

    def load_encoder(self) -> None:

        model = ColPali.from_pretrained(
            self.model_name,
            device_map=self.device,
        ).eval()
        # ---- ADDED: make sure the retrieval adapter is really applied ----
        if getattr(model, "peft_config", None):
            _, b_before, _ = _lora_stats(model)
            n_loaded, unmatched = _load_adapter_by_name(model, self.model_name)
            _, b_after, fingerprint = _lora_stats(model)
            if unmatched or n_loaded == 0 or b_after == 0.0:
                raise RuntimeError(
                    f"[ColPaliEncoder] adapter NOT applied: loaded={n_loaded}, unmatched={unmatched[:2]}, "
                    f"max|lora_B|={b_after}")
            print(f"[ColPaliEncoder] adapter OK: {n_loaded} LoRA tensors loaded, max|lora_B| "
                  f"{b_before:.4f} -> {b_after:.4f}, fingerprint {fingerprint:.1f} (must be identical every run)")
        # ---- END ADDED ----
        self.dim = model.config.hidden_size
        self.encoder = model
        self.processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(self.model_name))

        return

    # TODO fix this
    def free_encoder(self) -> None:
        if self.encoder is not None:
            del self.encoder
            self.encoder = None
        if self.processor:
            del self.processor
            self.processor = None

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        return

    def embedding(self, pages):

        images = [Image.open(name) for name in pages]

        dataloader = DataLoader(
            dataset=ListDataset[str](images),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x),
        )

        ds: List[torch.Tensor] = []
        for batch_doc in tqdm(dataloader, "embedding pdf's images"):
            with torch.no_grad():
                batch_doc = {k: v.to(self.encoder.device) for k, v in batch_doc.items()}
                embeddings_doc = self.encoder(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

        filepaths = [name for name in pages]
        data = []
        for i in range(len(filepaths)):
            data.append(
                {
                    "colbert_vecs": ds[i].float().numpy(),
                    "doc_id": i,
                    "filepath": filepaths[i],
                }
            )

        dict_list = []
        for pdf in tqdm(data, "insert pdf's image"):
            # Insert ColBERT embeddings and metadata for a document into the collection.
            colbert_vecs = [vec for vec in pdf["colbert_vecs"]]
            seq_length = len(colbert_vecs)
            doc_ids = [pdf["doc_id"] for i in range(seq_length)]
            seq_ids = list(range(seq_length))
            dict_list.extend(
                [
                    {
                        "vector": colbert_vecs[i],
                        "seq_id": seq_ids[i],
                        "doc_id": doc_ids[i],
                        "filepath": pdf["filepath"],
                    }
                    for i in range(seq_length)
                ]
            )
        return dict_list

    def embedding_query(self, queries) -> List[torch.Tensor]:
        dataloader = DataLoader(
            dataset=ListDataset[str](queries),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_queries(x),
        )

        qs: List[torch.Tensor] = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(self.encoder.device) for k, v in batch_query.items()}
                embeddings_query = self.encoder(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))
        return qs

    # @property
    # def dataset_name(self):
    # return self.__dataset_name

    # TODO add a dataset free
    def multi_gpus_embedding(self, texts):
        pass
