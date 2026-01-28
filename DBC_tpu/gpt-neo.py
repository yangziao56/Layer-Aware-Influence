import os
import time
import warnings

import torch
from datasets import DownloadConfig, load_dataset
from ghost_trainer import Trainer
from transformers import GPT2Tokenizer, GPTNeoConfig, GPTNeoForCausalLM, TrainingArguments


def _configure_tpu_topology_if_needed() -> None:
    """Ensure torchrun-managed TPU ranks configure PJRT before touching XLA."""
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return

    try:
        rank_int = int(local_rank)
    except ValueError:
        return

    local_world_size = os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("WORLD_SIZE")
    if not local_world_size:
        return

    try:
        world_size_int = int(local_world_size)
    except ValueError:
        return

    if world_size_int <= 1:
        return

    if os.environ.get("XLA_PJRT_TOPOLOGY_INITIALIZED") == "1":
        return

    try:
        from torch_xla._internal import pjrt  # pytype: disable=import-error
    except ImportError:
        return

    try:
        pjrt.initialize_multiprocess(rank_int, world_size_int)
    except RuntimeError as exc:
        warnings.warn(f"Skipping PJRT multiprocess init: {exc}")
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.warn(f"Failed to configure PJRT topology: {exc}")
    else:
        os.environ["XLA_PJRT_TOPOLOGY_INITIALIZED"] = "1"


_configure_tpu_topology_if_needed()

warnings.filterwarnings("ignore")

try:
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
except ImportError:
    xm = None
    XLA_AVAILABLE = False


def _print_once(*args, **kwargs):
    if XLA_AVAILABLE:
        try:
            xm.master_print(*args, **kwargs)
            return
        except RuntimeError:
            pass
    print(*args, **kwargs)


def main() -> None:
    start_time = time.time()

    config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
    model = GPTNeoForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-125M",
        config=config,
        use_safetensors=False,
    )

    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    streaming = os.environ.get("STREAM_DATASET", "1") == "1"
    if streaming:
        dataset = load_dataset(
            "apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2",
            streaming=True,
        )["train"]
        _print_once("Streaming dataset without local cache.")
    else:
        download_config = DownloadConfig(max_retries=20, timeout=120)
        dataset = load_dataset(
            "apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2",
            download_config=download_config,
        )["train"]
        _print_once("Dataset columns:", dataset.column_names)

    def tokenize_function(examples):
        examples["labels"] = examples["input_ids"]
        return examples

    map_kwargs = {"batched": True}
    if not streaming:
        map_kwargs["num_proc"] = min(32, os.cpu_count() or 1)

    tokenized_dataset = dataset.map(tokenize_function, **map_kwargs)
    if not streaming:
        _print_once("Tokenized columns:", tokenized_dataset.column_names)

    training_args_kwargs = dict(
        output_dir="./ours",
        evaluation_strategy="no",
        per_device_train_batch_size=16,
        logging_steps=10,
        save_steps=1000,
        num_train_epochs=1,
        report_to="none",
        logging_dir="./ours/logs",
        save_total_limit=100,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )

    if XLA_AVAILABLE:
        training_args_kwargs.update(
            dict(
                bf16=True,
                dataloader_pin_memory=False,
                tpu_num_cores=4,
            )
        )
    else:
        training_args_kwargs["fp16"] = torch.cuda.is_available()

    if streaming:
        training_args_kwargs["max_steps"] = int(os.environ.get("MAX_STEPS", "1000"))

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()

    removed_list = trainer.get_remove_list()
    _print_once("FINISHED!!!!!!", time.time() - start_time)
    _print_once(removed_list)


if __name__ == "__main__":
    main()
