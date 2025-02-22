"""
TorchDynamicShardInferenceEngine
A sharded inference engine using PyTorch-based torchtune models.
"""

import os
from concurrent.futures import ThreadPoolExecutor
import asyncio
import uuid
from typing import Optional
import logging

import numpy as np
import torch
import torchtune.generation as ttg

from exo.inference.inference_engine import InferenceEngine
from exo.download.shard_download import ShardDownloader
from exo.inference.shard import Shard
from exo.inference.tokenizers import _resolve_tokenizer
from exo.helpers import DEBUG
from exo.inference.torch.models.llm_utils import (
    load_model_config,
    load_model_weights_torchtune,
    ShardInferenceState,
)
from exo.inference.torch.models.general_mha import ShardedGeneralModel

# Default configuration constants
DEFAULT_TEMP = 0.6
DEFAULT_TOP_K = 35

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TorchDynamicShardInferenceEngine(InferenceEngine):
    """
    A PyTorch-based inference engine for sharded models.

    This engine efficiently manages sharded models, providing methods for encoding prompts,
    decoding tokens, sampling, and performing tensor inference. It optimizes resource usage,
    enhances performance with parallelism, and includes robust error handling.
    """

    def __init__(self, shard_downloader: ShardDownloader):
        """
        Initializes the sharded inference engine.

        Args:
            shard_downloader (ShardDownloader): Object responsible for downloading model shards.
        """
        self.current_shard = None
        self.shard = self.current_shard
        self.shard_downloader = shard_downloader
        self.model_instance = None
        self.current_request_id = None
        self.executor = ThreadPoolExecutor(max_workers=4)  # Optimized for parallelism
        self.instance_id = str(uuid.uuid4())
        self.model_directory = None
        self.model_config = None
        self.inference_state = None
        self.out_of_memory_count = 0

        # Cache configuration
        self.enable_cache = os.getenv("TORCH_USE_CACHE", "True").lower() == "true"
        self.cache_initialized = False

        # Device configuration
        self.device = self._determine_device()

        # Random number generator for sampling
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(1234)

    def _determine_device(self) -> torch.device:
        """
        Determines the appropriate device for computation based on environment and availability.

        Returns:
            torch.device: The selected device (CUDA, MPS, or CPU).
        """
        if os.environ.get("TORCH_DEVICE"):
            return torch.device(os.environ["TORCH_DEVICE"])
        elif torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")

    def setup_cache(self, batch_size: int = 1, total_response_length: int = 1024):
        """
        Configures the model cache if required and not already set up.

        Args:
            batch_size (int): Batch size for cache configuration. Defaults to 1.
            total_response_length (int): Total response length for cache setup. Defaults to 1024.
        """
        if not self.model_instance.model.caches_are_enabled() and self.enable_cache:
            with self.device:
                self.model_instance.model.setup_caches(
                    batch_size=batch_size,
                    dtype=self.model_config["torch_dtype"],
                    decoder_max_seq_len=total_response_length,
                )
            self.cache_initialized = True
            logger.info(
                f"Cache setup completed for batch_size={batch_size}, seq_len={total_response_length}"
            )

    def clear_model(self):
        """
        Frees the model, inference state, and clears GPU cache to prevent memory leaks.
        """
        if self.model_instance and self.model_instance.model.caches_are_enabled():
            self.model_instance.model.reset_caches()
        del self.model_instance
        self.model_instance = None
        if self.inference_state:
            del self.inference_state
            self.inference_state = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.current_shard = None
        self.cache_initialized = False
        logger.info("Model and resources cleared successfully")

    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        """
        Encodes a prompt into tensors for inference.

        Args:
            shard (Shard): The model shard to use for encoding.
            prompt (str): The input text to encode.

        Returns:
            np.ndarray: Encoded tensors ready for inference.

        Raises:
            RuntimeError: If the model cannot be loaded or configured.
        """
        if DEBUG >= 4:
            logger.debug(f"Encoding prompt: {prompt}")

        if self.model_instance is not None:
            logger.info("Clearing existing shard and model before encoding")
            self.clear_model()

        await self.ensure_shard(shard)

        def encode_task() -> np.ndarray:
            if self.model_instance is None:
                raise RuntimeError(
                    "Model instance is not loaded. Ensure the shard is loaded correctly."
                )

            tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            if DEBUG >= 4:
                logger.debug(f"Encoded tokens: {tokens}")

            max_gen_tokens = self.model_instance.max_generated_tokens
            if tokens.size(1) > max_gen_tokens:
                tokens = tokens[:, -max_gen_tokens:]

            self.inference_state.tokens = tokens
            batch_size, token_length = tokens.size()
            total_response_length = token_length + max_gen_tokens
            self.setup_cache(batch_size, total_response_length)

            max_seq_len = (
                total_response_length
                if not self.model_instance.model.caches_are_enabled()
                else self.model_instance.model.decoder_max_cache_seq_len
            )

            pad_id = getattr(self.tokenizer, "pad_id", 0)
            padding_masks = tokens != pad_id
            if not padding_masks.all():
                padding_masks = torch.nn.functional.pad(
                    padding_masks, (0, max_gen_tokens), value=True
                )
                self.inference_state.mask = ttg.get_causal_mask_from_padding_mask(
                    padding_masks, target_seq_len=max_seq_len
                )
                self.inference_state.input_pos = ttg.get_position_ids_from_padding_mask(
                    padding_masks
                )
            else:
                self.inference_state.mask = torch.tril(
                    torch.ones(
                        total_response_length,
                        max_seq_len,
                        dtype=torch.bool,
                        device=self.device,
                    )
                ).unsqueeze(0)
                self.inference_state.input_pos = torch.arange(
                    0, total_response_length, device=self.device
                ).unsqueeze(0)

            return tokens.cpu().numpy()

        return await asyncio.get_running_loop().run_in_executor(
            self.executor, encode_task
        )

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """
        Decodes tensors into human-readable text.

        Args:
            shard (Shard): The model shard to use for decoding.
            tokens (np.ndarray): The tensor data to decode.

        Returns:
            str: Decoded text.
        """
        if DEBUG >= 4:
            logger.debug(f"Decoding tokens: {tokens}")

        await self.ensure_shard(shard)
        tokens_tensor = torch.tensor(tokens).to(self.device)

        def decode_task():
            return self.tokenizer.decode(tokens_tensor.tolist())

        return await asyncio.get_running_loop().run_in_executor(
            self.executor, decode_task
        )

    async def sample(
        self,
        input_logits: np.ndarray,
        temp: float = DEFAULT_TEMP,
        top_k: int = DEFAULT_TOP_K,
    ) -> np.ndarray:
        """
        Samples the next tokens based on input logits.

        Args:
            input_logits (np.ndarray): Input logits for sampling.
            temp (float): Temperature for sampling. Defaults to 0.6.
            top_k (int): Top-k value for sampling. Defaults to 35.

        Returns:
            np.ndarray: Sampled tokens.
        """
        if DEBUG >= 4:
            logger.debug(f"Sampling with temp={temp}, top_k={top_k}")

        logits = torch.tensor(input_logits).to(self.device)

        def sample_task():
            sampling_noise = torch.empty(
                (
                    logits.size(0),
                    self.model_instance.model.tok_embeddings.num_embeddings,
                ),
                device=logits.device,
            ).exponential_(1, generator=self.rng)
            tokens = ttg.sample(
                logits.clone(), temperature=temp, top_k=top_k, q=sampling_noise
            )
            if DEBUG >= 4:
                logger.debug(f"Sampled tokens: {tokens}")
            return tokens.cpu().numpy()

        return await asyncio.get_running_loop().run_in_executor(
            self.executor, sample_task
        )

    async def infer_tensor(
        self,
        request_id: str,
        shard: Shard,
        input_data: np.ndarray,
        inference_state: Optional[dict] = None,
    ) -> tuple[np.ndarray, Optional[dict]]:
        """
        Performs inference on the input tensor.

        Args:
            request_id (str): Unique identifier for the inference request.
            shard (Shard): The model shard to use for inference.
            input_data (np.ndarray): Input data for inference.
            inference_state (Optional[dict]): Previous inference state, if any. Defaults to None.

        Returns:
            tuple[np.ndarray, Optional[dict]]: Inference result and updated state.

        Raises:
            RuntimeError: If inference fails due to model errors.
            torch.cuda.OutOfMemoryError: If GPU memory is exhausted.
        """
        await self.ensure_shard(shard)
        if inference_state:
            self.inference_state.from_dict(inference_state)
        self.current_request_id = request_id

        hidden_state = None
        input_tensor = None
        if input_data.ndim == 3:
            hidden_state = torch.tensor(input_data).to(
                device=self.device, dtype=self.model_config["torch_dtype"]
            )
        elif input_data.ndim == 2:
            input_tensor = torch.tensor(input_data).to(device=self.device)

        if self.enable_cache and not self.cache_initialized:
            batch_size = (
                input_tensor.size(0)
                if input_tensor is not None
                else self.inference_state.tokens.size(0)
            )
            token_length = (
                input_tensor.size(1)
                if input_tensor is not None
                else self.inference_state.tokens.size(1)
            )
            total_response_length = (
                token_length + self.model_instance.max_generated_tokens
            )
            self.setup_cache(batch_size, total_response_length)

        def infer_task():
            logger.info(f"Starting inference for request_id: {self.current_request_id}")
            model_cache_enabled = self.model_instance.model.caches_are_enabled()

            if (
                self.inference_state.tokens is not None
                and input_tensor is not None
                and input_tensor.size(-1) == 1
            ):
                self.inference_state.tokens = torch.cat(
                    [self.inference_state.tokens.to(self.device), input_tensor], dim=-1
                ).to(self.device)
            elif input_tensor is not None:
                self.inference_state.tokens = input_tensor

            try:
                in_tokens = self.inference_state.tokens.to(self.device)
                in_input_pos = self.inference_state.input_pos.to(self.device)
                in_mask = self.inference_state.mask.to(self.device)

                if hidden_state is not None:
                    model_hidden_state, model_logits = self.model_instance.generate(
                        tokens=in_tokens,
                        hidden_state=hidden_state,
                        input_pos=in_input_pos,
                        mask=in_mask,
                        curr_pos=self.inference_state.curr_pos,
                    )
                else:
                    if not model_cache_enabled:
                        model_hidden_state, model_logits = self.model_instance.generate(
                            tokens=in_tokens,
                            input_pos=in_input_pos,
                            mask=in_mask,
                            curr_pos=self.inference_state.curr_pos,
                        )
                    else:
                        model_hidden_state, model_logits = self.model_instance.generate(
                            tokens=input_tensor,
                            input_pos=in_input_pos,
                            mask=in_mask,
                            curr_pos=self.inference_state.curr_pos,
                        )
            except torch.cuda.OutOfMemoryError:
                logger.error("Out of memory on CUDA, clearing model and stopping")
                self.out_of_memory_count += 1
                self.clear_model()
                return None, None
            except RuntimeError as e:
                logger.error(f"Model runtime error: {e}")
                raise

            if model_hidden_state is not None:
                if model_hidden_state.dtype == torch.bfloat16:
                    model_hidden_state = model_hidden_state.float()
                if DEBUG >= 4:
                    logger.debug(f"Sending hidden states: {model_hidden_state.size()}")
                return model_hidden_state.cpu().numpy(), self.inference_state.to_dict()

            if self.inference_state.curr_pos == 0:
                self.inference_state.curr_pos = self.inference_state.tokens.size(-1)
            else:
                self.inference_state.curr_pos += 1

            if model_logits.dtype == torch.bfloat16:
                model_logits = model_logits.float()
            return model_logits[
                :, -1
            ].detach().cpu().numpy(), self.inference_state.to_dict()

        return await asyncio.get_running_loop().run_in_executor(
            self.executor, infer_task
        )

    async def ensure_shard(self, shard: Shard):
        if self.current_shard == shard:
            return

        self.current_shard = shard
        self.inference_state = ShardInferenceState()
        logger.info(f"Ensuring shard: {shard}")

        try:
            self.model_directory = await self.shard_downloader.ensure_shard(
                shard, self.__class__.__name__
            )
            logger.info(f"Model directory: {self.model_directory}")

            self.model_config = load_model_config(self.model_directory / "config.json")
            logger.info(f"Model config loaded: {self.model_config}")

            self.tokenizer = await _resolve_tokenizer(self.model_directory)
            logger.info(f"Tokenizer resolved: {self.tokenizer}")

            def initialize_model():
                logger.info("Initializing model...")
                self.model_instance = ShardedGeneralModel(
                    config=self.model_config,
                    shard=shard,
                    device=self.device,
                    dtype=self.model_config["torch_dtype"],
                    use_cache=self.enable_cache,
                )
                load_model_weights_torchtune(
                    cache_dir=self.model_directory,
                    shard=shard,
                    model=self.model_instance,
                    num_heads=self.model_config["num_heads"],
                    num_kv_heads=self.model_config["num_kv_heads"],
                    dim=self.model_config["embed_dim"],
                    head_dim=self.model_config["head_dim"],
                )
                logger.info(f"Model initialized for shard: {shard}")

            await asyncio.get_running_loop().run_in_executor(
                self.executor, initialize_model
            )
        except Exception as e:
            logger.error(f"Failed to ensure shard: {e}")
            raise

    async def load_checkpoint(self, shard: Shard, checkpoint_path: str):
        """
        Loads a checkpoint for the specified shard.

        Args:
            shard (Shard): The model shard to load the checkpoint for.
            checkpoint_path (str): Path to the checkpoint file.
        """
        await self.ensure_shard(shard)
        # Additional implementation may be required for checkpoint loading
        logger.info(
            f"Checkpoint loading not fully implemented for shard: {shard} at {checkpoint_path}"
        )
